import sys
sys.path.append("./")

import argparse
import time
import torch
from rdkit import Chem
from torch.utils.data import BatchSampler

from agents import Molda
from datasets import QM9
from utils import all_scores

'''
https://github.com/materialsvirtuallab/megnet
https://github.com/3dmol/3Dmol.js
https://github.com/lukasturcani/stk
https://github.com/mcs07/MolVS
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--log_interval', type=int, default=100)

    args = parser.parse_args()
    dataset = QM9('./.env/data/qm9')

    mol = dataset.graph2mol(dataset[10])
    print(dataset)
    print(len(dataset.smiles))
    print(Chem.MolToSmiles(mol))
    print(dataset.smiles[10])

    '''
    dataset = ZINC250K('./.env/data/zinc250k')
    mol = dataset.graph2mol(dataset[10])
    print(dataset)
    print(len(dataset.smiles))
    print(Chem.MolToSmiles(mol))
    print(dataset.smiles[10])
    '''

    train_size = int(len(dataset) * 0.8)
    eval_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size, test_size])

    model_path = args.model_path if args.model_path is not None else './Molda_{}.pt'.format(int(time.time()))

    molda = Molda('cpu', dataset.num_features, model_path=model_path, log_interval=args.log_interval)

    molda.share_memory()
    molda.train(train_dataset, args.n_steps)

    '''
    import torch.multiprocessing as mp
    processes = []
    for rank in range(args.n_workers):
        p = mp.Process(target=trainer.train, args=(rank, train_dataset, args.n_steps))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    '''

    # eval
    generated_data = molda.generate(dataset[:100])
    mols = [dataset.graph2mol(g, remove_Hs=True, sanitize=True) for g in generated_data]
    m0, m1 = all_scores(mols, eval_dataset, norm=True)
    print(m0, m1)

    # test
