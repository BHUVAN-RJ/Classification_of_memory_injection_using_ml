def update_file(filename):
    i = 0

    # Read the file and process each line
    with open(filename, 'r+') as file:
        lines = file.readlines()
        file.seek(0)  # Move the file pointer to the beginning

        for line in lines:
            i += 1
            line = line.strip()  # Remove leading/trailing whitespace
            if i % 6 == 4:
                if 'NULL' in line:
                    count = 0
                else:
                    elements = line.split(',')
                    count = len(elements)
                result = count / 6500
                line = str(result)
            file.write(line + '\n')

        file.truncate()  # Remove any remaining content

    file.close()


# Usage example
filename = '/Users/bhuvanrj/PycharmProjects/main_project/dataset.txt'  # Replace with the actual file path
update_file(filename)
