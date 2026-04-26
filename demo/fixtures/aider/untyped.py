def merge_dicts(a, b):
    result = dict(a)
    for key, value in b.items():
        result[key] = value
    return result


if __name__ == "__main__":
    print(merge_dicts({"x": 1}, {"y": 2}))
