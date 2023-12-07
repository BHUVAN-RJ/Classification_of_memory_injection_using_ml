def count_unique_elements(filename):
    unique_counts = []
    i = 0

    # Read the file and process each line
    with open(filename, 'r') as file:
        for line in file:
            i += 1
            line = line.strip()  # Remove leading/trailing whitespace
            if i % 6 == 4:
                words = line.split(',')
                unique_elements = set(word.strip() for word in words)
                unique_counts.append(len(unique_elements))

    return unique_counts


# Usage example
filename = '/Users/bhuvanrj/PycharmProjects/main_project/dataset.txt'  # Replace with the actual file path
unique_counts = count_unique_elements(filename)
total_unique_elements = sum(unique_counts)

print(f'Total number of unique elements: {total_unique_elements}')
