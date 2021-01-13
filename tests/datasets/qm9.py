from neuroblast.datasets import QM9
from rdkit import Chem

dataset = QM9('./.env/data/qm9')


def test_length():
    assert len(dataset) == len(dataset.smiles)


def test_convert():
    mol = dataset.graph2mol(dataset[10])
    assert Chem.MolToSmiles(mol) == dataset.smiles[10]
