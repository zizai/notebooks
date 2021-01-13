from .base_splitter import BaseSplitter  # NOQA
from .random_splitter import RandomSplitter  # NOQA
from .scaffold_splitter import ScaffoldSplitter  # NOQA
from .stratified_splitter import StratifiedSplitter  # NOQA
from .time_splitter import TimeSplitter  # NOQA

split_method_dict = {
    'random': RandomSplitter,
    'stratified': StratifiedSplitter,
    'scaffold': ScaffoldSplitter,
    'time': TimeSplitter,
}
