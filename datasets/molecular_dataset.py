import torch
import torch.nn.functional as F

from data.parsers import CSVFileParser
from data.preprocessors import MPNNPreprocessor
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset


class MolecularDataset(InMemoryDataset):

    components = [Chem.Atom(i + 1) for i in range(120)]
    max_atom_num = 120

    def __init__(self, root, smiles_col=None, label_names=None, transform=None):
        self.smiles_col = smiles_col
        self.label_names = label_names

        super(MolecularDataset, self).__init__(root, transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.smiles = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        return 'graphs.pt', 'smiles.pt'

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def atom_encoder(self, atom_nums):
        x = torch.tensor(atom_nums, dtype=torch.long)
        x = F.one_hot(x - 1, num_classes=self.max_atom_num)
        return x

    def atom_decoder(self, node_label):
        atom_num = node_label + 1
        atom = Chem.Atom(atom_num)
        return atom

    def bond_encoder(self, index, attr):
        edge_index = torch.tensor(index).transpose(0, 1)
        edge_attr = torch.tensor(attr)
        return edge_index, edge_attr

    def bond_decoder(self, edge_label):
        decoder = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
        return decoder[edge_label]

    def graph2mol(self, g, remove_Hs=True, sanitize=True):
        node_labels = torch.argmax(g.x, dim=-1).tolist()
        edges = g.edge_index.transpose(0, 1).tolist()
        edge_labels = torch.argmax(g.edge_attr, dim=-1).tolist()

        mol = Chem.RWMol()
        for node_label in node_labels:
            atom = self.atom_decoder(node_label)
            mol.AddAtom(atom)

        for idx, (start, end) in enumerate(edges):
            bond_type = self.bond_decoder(edge_labels[idx])
            mol.AddBond(int(start), int(end), bond_type)

        if remove_Hs:
            try:
                mol = Chem.RemoveHs(mol)
            except:
                mol = None

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def process(self):
        preprocessor = MPNNPreprocessor(add_Hs=True, kekulize=True)
        parser = CSVFileParser(preprocessor, labels=self.label_names, smiles_col=self.smiles_col)
        result = parser.parse(self.raw_paths[0], return_smiles=True)

        data_list = []
        for i, d in enumerate(result['dataset']):
            x = self.atom_encoder(d[0])
            edge_index, edge_attr = self.bond_encoder(d[1], d[2])
            if self.label_names is not None:
                y = torch.tensor(d[3])
                g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            else:
                g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(g)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(result['smiles'], self.processed_paths[1])
