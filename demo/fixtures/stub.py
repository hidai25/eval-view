from typing import Dict, List


def group_by_key(items: List[Dict], key: str) -> Dict[str, List[Dict]]:
    """Group a list of dicts by the value of a given key.

    Args:
        items: List of dictionaries to group.
        key: The key whose value determines the group.

    Returns:
        A dict mapping each unique value of ``key`` to the list of dicts
        that had that value.  Items missing ``key`` are silently skipped.

    Example:
        >>> items = [{"type": "a", "v": 1}, {"type": "b", "v": 2}, {"type": "a", "v": 3}]
        >>> group_by_key(items, "type")
        {'a': [{'type': 'a', 'v': 1}, {'type': 'a', 'v': 3}], 'b': [{'type': 'b', 'v': 2}]}
    """
    result: Dict[str, List[Dict]] = {}
    for item in items:
        if key not in item:
            continue
        group = item[key]
        if group not in result:
            result[group] = []
        result[group].append(item)
    return result


if __name__ == "__main__":
    items = [
        {"type": "fruit", "name": "apple"},
        {"type": "veggie", "name": "carrot"},
        {"type": "fruit", "name": "banana"},
    ]
    print(group_by_key(items, "type"))
