def get_mean(numbers):
    return sum(numbers) / len(numbers)


def get_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 0:
        return (sorted_numbers[n//2] + sorted_numbers[n//2 - 1]) / 2
    else:
        return sorted_numbers[n//2]


def get_mode(numbers):
    counts = {}
    for num in numbers:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1
    max_count = max(counts.values())
    mode = [num for num, count in counts.items() if count == max_count]
    return mode


def get_variance(number):
    mean = get_mean(number)
    variance = sum((x - mean) ** 2 for x in number) / len(number)
    return variance


def get_standard_deviation(number):
    variance = get_variance(number)
    return variance ** 0.5


sample = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(f"Mean: {get_mean(sample)}")
print(f"Median: {get_median(sample)}")
print(f"Mode: {get_mode(sample)}")
print(f"Variance: {get_variance(sample)}")
print(f"Standard Deviation: {get_standard_deviation(sample)}")
