import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

###############

def get_user_input():
    sub_systems = []
    interfaces = []

    # Get sub-systems from the user
    num_sub_systems = int(input("Enter the number of sub-systems: "))
    for i in range(num_sub_systems):
        sub_system = input(f"Enter sub-system {i+1}: ")
        sub_systems.append(sub_system)

    # Get interfaces from the user
    print("\nEnter interfaces between sub-systems (e.g., 'Subsystem1-Subsystem2'). Type 'done' when finished.")
    while True:
        interface = input("Enter interface: ")
        if interface.lower() == 'done':
            break
        interfaces.append(interface)

    return sub_systems, interfaces

sub_systems, interfaces = get_user_input()
print("\nSub-systems:", sub_systems)
print("Interfaces:", interfaces)


def create_dsm_matrix(sub_systems, interfaces):
    # Create a blank DSM matrix
    matrix_size = len(sub_systems)
    dsm_matrix = np.zeros((matrix_size, matrix_size))

    # Populate the DSM matrix based on interfaces
    for interface in interfaces:
        system_a, system_b = interface.split('-')
        a_index = sub_systems.index(system_a)
        b_index = sub_systems.index(system_b)
        dsm_matrix[a_index][b_index] = 1  # Mark the interaction

    return dsm_matrix

def perform_clustering(dsm_matrix):
    # Perform hierarchical clustering
    linked = linkage(dsm_matrix, 'single')
    
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=sub_systems, distance_sort='descending', show_leaf_counts=True)
    plt.show()

# Assuming sub_systems and interfaces are already collected from the user
dsm_matrix = create_dsm_matrix(sub_systems, interfaces)
print("DSM Matrix:\n", dsm_matrix)

# Perform clustering
perform_clustering(dsm_matrix)


#Assuming `dsm_matrix` is the DSM matrix that we have created earlier
# and `sub_systems` contains the list of sub-system names.


#-------------------------------------
# Visualisation
#-------------------------------------

# Create a figure and a subplot
fig, ax = plt.subplots()

# Use a colormap to represent the presence of an interface (black for presence)
cmap = plt.cm.get_cmap('Greys', 2)  # 2 distinct colors for 0 and 1 values

# Create the heatmap for the DSM matrix
cax = ax.matshow(dsm_matrix, interpolation='nearest', cmap=cmap)

# Set gridlines based on matrix dimensions
ax.set_xticks(np.arange(len(sub_systems)))
ax.set_yticks(np.arange(len(sub_systems)))

# Label the grid with sub-system names
ax.set_xticklabels(sub_systems, rotation=90)
ax.set_yticklabels(sub_systems)

# Draw blue dots and letters on the diagonal
for i in range(len(sub_systems)):
    ax.plot(i, i, marker='o', color='blue', markersize=10)  # Draw blue dot
    ax.text(i, i, sub_systems[i][0], color='white', ha='center', va='center')  # Draw first letter

# Show the plot
plt.show()