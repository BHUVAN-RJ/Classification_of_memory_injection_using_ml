import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('processed_dataset_graph.csv', names=['Process Name','Process ID', 'Memory Usage (MB)', 'Percent of Loaded DLLs', 'Injection'], index_col=False,header=0)

# Initialize variables
i = 1
j = 0

# Plot a graph for each row
for _, row in data.iterrows():
    print(row)
    # Increment i and j
    i += 1
    if i % 2 == 0:
        j += 1

    # Determine color and marker symbol
    color = 'green' if i % 2 == 1 else 'red'
    marker = ['o', 's', 'D', '^', 'v','+','*'][j % 7]

    # Extract data from the row
    memory_usage = row['Memory Usage (MB)']
    percent_loaded_dlls = row['Percent of Loaded DLLs']

    # Plot the graph
    plt.scatter(memory_usage, percent_loaded_dlls, color=color, marker=marker)

# Set labels and title
plt.xlabel('Memory Usage (MB)')
plt.ylabel('Percent of Loaded DLLs')
plt.title('Memory Usage vs. Percent of Loaded DLLs')
plt.legend()

# Show the graph
plt.show()
