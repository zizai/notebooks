from functools import partial

from .coalesce import coalesce
from .deep_update import deep_update
from .scatter import scatter_add
from .seq_to_graph import seq_to_graph


def add_mixins(base, mixins):
    """Returns a new class with mixins applied in priority order."""

    mixins = list(mixins or [])

    while mixins:

        class new_base(mixins.pop(), base):
            pass

        base = new_base

    return base


def force_list(elements=None, to_tuple=False):
    """
    Makes sure `elements` is returned as a list, whether `elements` is a single
    item, already a list, or a tuple.
    Args:
        elements (Optional[any]): The inputs as single item, list, or tuple to
            be converted into a list/tuple. If None, returns empty list/tuple.
        to_tuple (bool): Whether to use tuple (instead of list).
    Returns:
        Union[list,tuple]: All given elements in a list/tuple depending on
            `to_tuple`'s value. If elements is None,
            returns an empty list/tuple.
    """
    ctor = list
    if to_tuple is True:
        ctor = tuple
    return ctor() if elements is None else ctor(elements) \
        if type(elements) in [list, tuple] else ctor([elements])


force_tuple = partial(force_list, to_tuple=True)
