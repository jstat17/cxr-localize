from collections.abc import Iterable
from typing import Any

def intersect_iters(iter1: Iterable, iter2: Iterable) -> bool:
    """Determine if two iterables have any common entries

    Args:
        iter1 (Iterable): First iterable
        iter2 (Iterable): Second iterable

    Returns:
        bool: If the two iterables have common entries
    """
    intersection = set(iter1) & set(iter2)
    return bool(intersection)

def get_iter_to_iter_dict(iter1: Iterable[Any], iter2: Iterable[Any]) -> dict[Any, Any]:
    """Get a dictionary from iter1 values to iter2 values

    Args:
        iter1 (Iterable[Any]): First iterable (dictionary keys)
        iter2 (Iterable[Any]): Second iterable (dictionary values)

    Returns:
        dict[Any, Any]: Dictionary from iter1 values to iter2 values
    """
    mapping = dict()
    for iter1_val, iter2_val in zip(iter1, iter2):
        mapping[iter1_val] = iter2_val
    
    return mapping