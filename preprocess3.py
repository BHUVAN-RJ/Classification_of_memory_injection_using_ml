#This script goes through the line and replaces white spaces with commas so that we can create a list from the loaded DLL files
def process_file(file_path):
    # Open the file with a reader and writer
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process the lines and replace whitespaces with commas for line number mod 6 equals 4
    with open(file_path, 'w') as file:
        i = 0
        for line in lines:
            if i % 6 == 3:
                modified_line = ""
                j = 1
                while j < len(line):
                    if line[j] == ' ' and (j + 1) < len(line) and line[j + 1] != '(':
                        modified_line += ','
                    else:
                        modified_line += line[j]
                    j += 1
                file.write(modified_line)
            else:
                file.write(line)
            i += 1

# File path of the text file
file_path = '/Users/bhuvanrj/PycharmProjects/main_project/dataset.txt'

# Call the function to process the file
process_file(file_path)
