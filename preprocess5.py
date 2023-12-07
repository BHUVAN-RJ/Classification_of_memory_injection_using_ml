def modify_file(filename):
    modified_lines = []
    i = 0

    # Read the file and process each line
    with open(filename, 'r') as file:
        for line in file:
            i += 1
            line = line.strip()  # Remove leading/trailing whitespace
            if i % 6 == 4:
                words = line.split(',')
                quoted_words = ['"' + word.strip() + '"' for word in words]
                modified_line = ', '.join(quoted_words)
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)

    # Write the modified lines back to the file
    with open(filename, 'w') as file:
        file.write('\n'.join(modified_lines))


# Usage example
filename = '/Users/bhuvanrj/PycharmProjects/main_project/dataset.txt'  # Replace with the actual file path
modify_file(filename)
