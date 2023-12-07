def process_file(file_path):
    # Open the file with a reader and writer
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process the lines and replace if line number mod 6 is equal to 4
    i = 0
    with open(file_path, 'w') as file:
        for line in lines:
            if i % 6 == 3:
                if len(line)<=1:
                    file.write(" NULL\n")
                else:
                    file.write(line)
            else:
                file.write(line)
            i += 1

            # File path of the text file
file_path = '/Users/bhuvanrj/PycharmProjects/main_project/dataset.txt'

            # Call the function to process the file
process_file(file_path)