def process_dataset(file_path):
    # Open the file with a reader
    with open(file_path, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    for i in range(0, len(lines), 6):
        # Combine six lines into a single line
        combined_line = ','.join([line.strip() for line in lines[i:i+6]])
        processed_lines.append(combined_line)

    # Write the processed lines to a new file
    output_file_path = 'processed_dataset_graph.csv'
    with open(output_file_path, 'w') as file:
        file.write('\n'.join(processed_lines))

    print(f"Processed dataset saved to '{output_file_path}'.")


# File path of the dataset
file_path = '/Users/bhuvanrj/PycharmProjects/main_project/testdataset.txt'

# Call the function to process the dataset
process_dataset(file_path)
