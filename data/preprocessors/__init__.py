from .atomic_number_preprocessor import AtomicNumberPreprocessor  # NOQA
from .base_preprocessor import BasePreprocessor  # NOQA
from .common import construct_adj_matrix  # NOQA
from .common import construct_atomic_number_array  # NOQA
from .common import construct_discrete_edge_matrix  # NOQA
from .common import construct_supernode_feature  # NOQA
from .common import MolFeatureExtractionError  # NOQA
from .common import type_check_num_atoms  # NOQA
from .ecfp_preprocessor import ECFPPreprocessor  # NOQA
from .relgat_preprocessor import RelGATPreprocessor  # NOQA
from .ggnn_preprocessor import GGNNPreprocessor  # NOQA
from .gin_preprocessor import GINPreprocessor  # NOQA
from .mol_preprocessor import MolPreprocessor  # NOQA
from .mpnn_preprocessor import MPNNPreprocessor  # NOQA
from .nfp_preprocessor import NFPPreprocessor  # NOQA
from .relgcn_preprocessor import RelGCNPreprocessor  # NOQA
from .rsgcn_preprocessor import RSGCNPreprocessor  # NOQA
from .schnet_preprocessor import SchNetPreprocessor  # NOQA
from .weavenet_preprocessor import WeaveNetPreprocessor  # NOQA

preprocess_method_dict = {
    'ecfp': ECFPPreprocessor,
    'nfp': NFPPreprocessor,
    'ggnn': GGNNPreprocessor,
    'gin': GINPreprocessor,
    'schnet': SchNetPreprocessor,
    'weavenet': WeaveNetPreprocessor,
    'relgcn': RelGCNPreprocessor,
    'rsgcn': RSGCNPreprocessor,
    'relgat': RelGATPreprocessor,
}
