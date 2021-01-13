"""Tests for CPyDock scoring function module"""

import os
import numpy as np
from lightdock.scoring.cpydock.driver import CPyDock, CPyDockAdapter
from lightdock.pdbutil.PDBIO import parse_complex_from_file
from lightdock.structure.complex import Complex


if __name__ == '__main__':
    data_path = os.path.expanduser('~/pdb')
    pydock = CPyDock()

    atoms, residues, chains = parse_complex_from_file(data_path + '1AY7_rec.pdb.H')
    receptor = Complex(chains, atoms, structure_file_name=(data_path + '1AY7_rec.pdb.H'))
    atoms, residues, chains = parse_complex_from_file(data_path + '1AY7_lig.pdb.H')
    ligand = Complex(chains, atoms, structure_file_name=(data_path + '1AY7_lig.pdb.H'))
    adapter = CPyDockAdapter(receptor, ligand)
    score = pydock(adapter.receptor_model, adapter.receptor_model.coordinates[0],
                   adapter.ligand_model, adapter.ligand_model.coordinates[0])
    print(score)
    np.testing.assert_almost_equal(- 15.923994756, score)
