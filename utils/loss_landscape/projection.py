"""
    Project a model or multiple models to a plane spaned by given directions.
"""

import numpy as np
import torch
import copy


def tensorlist_to_tensor(weights):
    """ Concatnate a list of tensors into one tensor.

        Args:
            weights: a list of parameter tensors, e.g. net_plotter.get_weights(net).

        Returns:
            concatnated 1D tensor
    """
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])


def nplist_to_tensor(nplist):
    """ Concatenate a list of numpy vectors into one tensor.

        Args:
            nplist: a list of numpy vectors, e.g., direction loaded from h5 file.

        Returns:
            concatnated 1D tensor
    """
    v = []
    for d in nplist:
        w = torch.tensor(d*np.float64(1.0))
        # Ignoreing the scalar values (w.dim() = 0).
        if w.dim() > 1:
            v.append(w.view(w.numel()))
        elif w.dim() == 1:
            v.append(w)
    return torch.cat(v)


def npvec_to_tensorlist(direction, params):
    """ Convert a numpy vector to a list of tensors with the same shape as "params".

        Args:
            direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
            base: a list of parameter tensors from net

        Returns:
            a list of tensors with the same shape as base
    """
    if isinstance(params, list):
        w2 = copy.deepcopy(params)
        idx = 0
        for w in w2:
            w.copy_(torch.tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return w2
    else:
        s2 = []
        idx = 0
        for (k, w) in params.items():
            s2.append(torch.Tensor(direction[idx:idx + w.numel()]).view(w.size()))
            idx += w.numel()
        assert(idx == len(direction))
        return s2



def cal_angle(vec1, vec2):
    """ Calculate cosine similarities between two torch tensors or two ndarraies
        Args:
            vec1, vec2: two tensors or numpy ndarraies
    """
    if isinstance(vec1, torch.Tensor) and isinstance(vec1, torch.Tensor):
        return torch.dot(vec1, vec2)/(vec1.norm()*vec2.norm()).item()
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        return np.ndarray.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def project_1D(w, d):
    """ Project vector w to vector d and get the length of the projection.

        Args:
            w: vectorized weights
            d: vectorized direction

        Returns:
            the projection scalar
    """
    assert len(w) == len(d), 'dimension does not match for w and '
    scale = torch.dot(w, d)/d.norm()
    return scale.item()


def project_2D(d, dx, dy, proj_method):
    """ Project vector d to the plane spanned by dx and dy.

        Args:
            d: vectorized weights
            dx: vectorized direction
            dy: vectorized direction
            proj_method: projection method
        Returns:
            x, y: the projection coordinates
    """

    if proj_method == 'cos':
        # when dx and dy are orthorgonal
        x = project_1D(d, dx)
        y = project_1D(d, dy)
    elif proj_method == 'lstsq':
        # solve the least squre problem: Ax = d
        A = np.vstack([dx.numpy(), dy.numpy()]).T
        [x, y] = np.linalg.lstsq(A, d.numpy())[0]

    return x, y
