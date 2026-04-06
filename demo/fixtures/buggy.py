def find_max(numbers):
    """Return the maximum value in a list of numbers."""
    if not numbers:
        return None
    max_val = numbers[0]
    for i in range(1, len(numbers)):  # fixed: include last element
        if numbers[i] > max_val:
            max_val = numbers[i]
    return max_val


if __name__ == "__main__":
    print(find_max([3, 1, 4, 1, 5, 9, 2, 6]))  # should print 9
