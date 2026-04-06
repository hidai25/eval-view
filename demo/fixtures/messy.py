def process_fields(fields):
    result = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                result[key] = stripped
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    data = {"name": "  Alice  ", "age": 30, "bio": "", "city": None}
    print(process_fields(data))  # should print {'name': 'Alice', 'age': 30}
