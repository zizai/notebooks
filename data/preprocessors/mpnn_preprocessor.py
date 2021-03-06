from .common import construct_atomic_number_array, construct_discrete_edge_index, type_check_num_atoms
from .mol_preprocessor import MolPreprocessor


class MPNNPreprocessor(MolPreprocessor):
    """Molecular data preprocessor message-passing neural networks

    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.

    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=True,
                 kekulize=True):
        super(MPNNPreprocessor, self).__init__(
            add_Hs=add_Hs, kekulize=kekulize)
        if max_atoms >= 0 and out_size >= 0 and max_atoms > out_size:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size

    def get_input_features(self, mol):
        """get input features

        Args:
            mol (Mol): Molecule input

        Returns:

        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = construct_atomic_number_array(mol, out_size=self.out_size)
        edge_index, edge_attr = construct_discrete_edge_index(mol)
        return atom_array, edge_index, edge_attr
