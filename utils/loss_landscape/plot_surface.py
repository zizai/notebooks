"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
from ray.rllib import SampleBatch

from utils.loss_landscape import mpi4pytorch, net_plotter, plot_1D, plot_2D, projection, scheduler


def name_surface_file(data_split, dir_file, raw_data, split_idx, xmax, xmin, xnum, y, ymax, ymin, ynum):
    # use args.dir_file as the prefix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(xmin), str(xmax), int(xnum))
    if y:
        surf_file += 'x[%s,%s,%d]' % (str(ymin), str(ymax), int(ynum))

    # dataloder parameters
    if raw_data: # without data normalization
        surf_file += '_rawdata'
    if data_split > 1:
        surf_file += '_datasplit=' + str(data_split) + '_splitidx=' + str(split_idx)

    return surf_file + ".h5"


def setup_surface_file(dir_file, surf_file, xmax, xmin, xnum, y, ymax, ymin, ynum):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print("surface file is already set up: %s" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(xmin, xmax, num=xnum)
    f['xcoordinates'] = xcoordinates

    if y:
        ycoordinates = np.linspace(ymin, ymax, num=ynum)
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def crunch(trainer, dir_type, surf_file, w, s, d, loss_key, comm, rank):
    """
        Calculate the loss values of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses = []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
    else:
        losses = f[loss_key][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        net = trainer.get_policy().model
        assert dir_type in ['states', 'weights'], 'Wrong dir_type={}'.format(dir_type)
        if dir_type == 'weights':
            net_plotter.set_weights(net, w, d, coord)
        elif dir_type == 'states':
            net_plotter.set_states(net, s, d, coord)
        #trainer.set_weights(net.state_dict().items())

        # Record the time to compute the loss value
        loss_start = time.time()
        postprocessed_batch = trainer.evaluate()
        loss = - postprocessed_batch['rewards'].mean()
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss

        # Send updated plot data to the master node
        syc_start = time.time()
        losses = mpi4pytorch.reduce_max(comm, losses)
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                loss_compute_time, syc_time))

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi4pytorch.reduce_max(comm, losses)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()


def run(model_file,
        trainer,
        data_split=1,
        dir_type='states',
        idx=0,
        log=False,
        loss_max=5.,
        raw_data=False,
        same_dir=False,
        plot=False,
        proj_file='',
        seed=123,
        show=False,
        split_idx=0,
        use_mpi=False,
        vmin=0.1,
        vmax=10.,
        vlevel=0.5,
        x='-1:1:51',
        y=None,
        xignore='',
        xnorm='filter',
        yignore='',
        ynorm='filter'):
    torch.manual_seed(seed)
    ###############################################################
    # Environment setup
    ###############################################################
    if use_mpi:
        comm = mpi4pytorch.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    # if use_cuda:
    #     if not torch.cuda.is_available():
    #         raise Exception('User selected cuda option, but cuda is not available on this machine')
    #     gpu_count = torch.cuda.device_count()
    #     torch.cuda.set_device(rank % gpu_count)
    #     print('Rank %d use GPU %d of %d GPUs on %s' %
    #           (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    ###############################################################
    # Check plotting resolution
    ###############################################################
    try:
        xmin, xmax, xnum = [float(a) for a in x.split(':')]
        ymin, ymax, ynum = (None, None, None)
        if y:
            ymin, ymax, ynum = [float(a) for a in y.split(':')]
            assert ymin and ymax and ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    ###############################################################
    # Load models and extract parameters
    ###############################################################
    net = trainer.get_policy().model
    w = net_plotter.get_weights(net)
    s = copy.deepcopy(net.state_dict())

    ###############################################################
    # Setup the direction file and the surface file
    ###############################################################
    dir_file = net_plotter.name_direction_file(dir_type, idx, model_file, same_dir, xignore, xnorm, y, yignore, ynorm)
    if rank == 0:
        net_plotter.setup_direction(dir_file, dir_type, net, same_dir, xignore, xnorm, y, yignore, ynorm)

    surf_file = name_surface_file(data_split, dir_file, raw_data, split_idx, xmax, xmin, xnum, y, ymax, ymin, ynum)
    if rank == 0:
        setup_surface_file(dir_file, surf_file, xmax, xmin, xnum, y, ymax, ymin, ynum)

    # wait until master has setup the direction file and surface file
    mpi4pytorch.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the cosine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = projection.cal_angle(projection.nplist_to_tensor(d[0]), projection.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    mpi4pytorch.barrier(comm)

    ################################################################
    # Start the computation
    ###############################################################
    crunch(trainer, dir_type, surf_file, w, s, d, 'train_loss', comm, rank)

    ###############################################################
    # Plot figures
    ###############################################################
    if plot and rank == 0:
        if y and proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, proj_file, 'train_loss', show)
        elif y:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', vmin, vmax, vlevel, show)
        else:
            plot_1D.plot_1d_loss(surf_file, xmin, xmax, loss_max, log, show)


###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')
