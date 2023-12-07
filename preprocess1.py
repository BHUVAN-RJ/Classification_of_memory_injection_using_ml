#This script deletes all the unnecessary words from the dataset

def delete_words(file_path, words_to_delete):
    # Read the contents of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Delete the specified words from the content
    for word in words_to_delete:
        content = content.replace(word, '')

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)

# File path of the text file
file_path = '/Users/bhuvanrj/PycharmProjects/main_project/dataset.txt'

# Words to delete
words_to_delete = ['Process Name', 'Process ID', 'Memory Usage (MB)', 'Loaded DLLs', 'Injection Technique','-----------------------------------------------------', ':']
# Call the function to delete the words from the file
delete_words(file_path, words_to_delete)
