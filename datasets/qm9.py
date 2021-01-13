import glob
import numpy as np
import os
import pandas
import shutil
import tarfile
import tempfile
from logging import getLogger
from rdkit import Chem
from tqdm import tqdm

from data import download
from data.parsers import CSVFileParser
from data.preprocessors import AtomicNumberPreprocessor
from datasets import MolecularDataset


url = 'https://ndownloader.figshare.com/files/3195389'
file_name = 'qm9.csv'

_root = 'qm9'

_target_column_names = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                'zpve', 'U0', 'U', 'H', 'G', 'Cv']
_smiles_column_names = ['SMILES1', 'SMILES2']


def get_qm9_label_names():
    """Returns label names of QM9 datasets."""
    return _target_column_names


def get_qm9(preprocessor=None, labels=None, return_smiles=False,
            target_index=None):
    """Downloads, caches and preprocesses QM9 dataset.

    Args:
        preprocessor (BasePreprocessor): Preprocessor.
            This should be chosen based on the network to be trained.
            If it is None, default `AtomicNumberPreprocessor` is used.
        labels (str or list): List of target labels.
        return_smiles (bool): If set to ``True``,
            smiles array is also returned.
        target_index (list or None): target index list to partially extract
            dataset. If None (default), all examples are parsed.

    Returns:
        dataset, which is composed of `features`, which depends on
        `preprocess_method`.

    """
    labels = labels or get_qm9_label_names()
    if isinstance(labels, str):
        labels = [labels, ]

    def postprocess_label(label_list):
        # This is regression task, cast to float value.
        return np.asarray(label_list, dtype=np.float32)

    if preprocessor is None:
        preprocessor = AtomicNumberPreprocessor()
    parser = CSVFileParser(preprocessor, postprocess_label=postprocess_label,
                           labels=labels, smiles_col='SMILES1')
    result = parser.parse(get_qm9_filepath(), return_smiles=return_smiles,
                          target_index=target_index)

    if return_smiles:
        return result['dataset'], result['smiles']
    else:
        return result['dataset']


def get_qm9_filepath(download_if_not_exist=True):
    """Construct a filepath which stores qm9 dataset for config_name

    This method check whether the file exist or not,  and downloaded it if
    necessary.

    Args:
        download_if_not_exist (bool): If `True` download dataset
            if it is not downloaded yet.

    Returns (str): file path for qm9 dataset (formatted to csv)

    """
    cache_path = _get_qm9_filepath()
    if not os.path.exists(cache_path):
        if download_if_not_exist:
            is_successful = download_and_extract_qm9(save_filepath=cache_path)
            if not is_successful:
                logger = getLogger(__name__)
                logger.warning('Download failed.')
    return cache_path


def _get_qm9_filepath():
    """Construct a filepath which stores QM9 dataset in csv

    This method does not check if the file is already downloaded or not.

    Returns (str): filepath for qm9 dataset

    """
    cache_root = download.get_dataset_directory(_root)
    cache_path = os.path.join(cache_root, file_name)
    return cache_path


def download_and_extract_qm9(save_filepath):
    logger = getLogger(__name__)
    logger.warning('Extracting QM9 dataset, it takes time...')
    download_file_path = download.cached_download(url)
    tf = tarfile.open(download_file_path, 'r')
    temp_dir = tempfile.mkdtemp()
    tf.extractall(temp_dir)
    file_re = os.path.join(temp_dir, '*.xyz')
    file_pathes = glob.glob(file_re)
    # Make sure the order is sorted
    file_pathes.sort()
    ls = []
    for path in tqdm(file_pathes):
        with open(path, 'r') as f:
            data = [line.strip() for line in f]

        num_atom = int(data[0])
        properties = list(map(float, data[1].split('\t')[1:]))
        smiles = data[3 + num_atom].split('\t')
        new_ls = smiles + properties
        ls.append(new_ls)

    df = pandas.DataFrame(ls, columns=_smiles_column_names + _target_column_names)
    df.to_csv(save_filepath)
    shutil.rmtree(temp_dir)
    return True


class QM9(MolecularDataset):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
    Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
    about 130,000 molecules with 13 regression targets.
    Each molecule includes complete spatial information for the single low
    energy conformation of the atoms in the molecule.
    In addition, we provide the atom features from the `"Neural Message
    Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    components = [Chem.Atom(i) for i in [6, 7, 8, 9]]
    max_atom_num = 9
    target_label_names = _target_column_names
    url = url

    def __init__(self, root, transform=None):
        super(QM9, self).__init__(root, smiles_col=_smiles_column_names[0], label_names=_target_column_names, transform=transform)

    @property
    def raw_file_names(self):
        return file_name

    def download(self):
        download_and_extract_qm9(self.raw_paths[0])
