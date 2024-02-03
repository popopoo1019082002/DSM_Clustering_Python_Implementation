import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

# Step 1: Data Collection
def get_user_input():
    """
    Collects user input for sub-systems and interfaces in a system engineering project.
    Allows the user to correct entries before final confirmation.

    Returns:
        sub_systems (list): List of sub-system names.
        interfaces (list): List of interfaces between sub-systems.
    """
    sub_systems = []
    interfaces = []

    # Function to display current entries and ask for confirmation or correction
    def confirm_or_correct_entries(entries, entry_type):
        print(f"\nCurrent {entry_type}:")
        for i, entry in enumerate(entries, 1):
            print(f"{i}. {entry}")
        correct_entry = input(f"Do you want to correct any {entry_type}? Enter the number or 'no' to continue: ").lower()
        if correct_entry.isdigit():
            index_to_correct = int(correct_entry) - 1
            if 0 <= index_to_correct < len(entries):
                new_entry = input(f"Enter the new value for {entry_type} {correct_entry}: ")
                entries[index_to_correct] = new_entry
            else:
                print("Invalid number. Please try again.")
        elif correct_entry != 'no':
            print("Invalid input. Please enter a number or 'no'.")

    # Collect sub-systems with a correction feature
    while True:
        sub_system = input("Enter sub-system name (or 'done' if finished): ")
        if sub_system.lower() == 'done':
            if sub_systems:  # Ensure there is at least one sub-system before confirming
                confirm_or_correct_entries(sub_systems, "sub-systems")
            if input("Are all sub-system entries correct? (yes/no): ").lower() == 'yes':
                break
        else:
            sub_systems.append(sub_system)

    # Collect interfaces with a correction feature
    print("\nEnter interfaces between sub-systems (e.g., 'Subsystem1-Subsystem2').")
    while True:
        interface = input("Enter interface (or 'done' if finished): ")
        if interface.lower() == 'done':
            if interfaces:  # Ensure there is at least one interface before confirming
                confirm_or_correct_entries(interfaces, "interfaces")
            if input("Are all interface entries correct? (yes/no): ").lower() == 'yes':
                break
        else:
            interfaces.append(interface)

    return sub_systems, interfaces

# Step 2: Matrix Creation
def create_dsm_matrix(sub_systems, interfaces):
    """
    Creates a Design Structure Matrix (DSM) based on the defined sub-systems and their interfaces.
    This version accounts for asymmetric dependencies, where one sub-system's dependency on another
    does not necessarily imply the reverse.

    Args:
        sub_systems (list): List of sub-system names.
        interfaces (list): List of interfaces between sub-systems, specified as 'Subsystem1-Subsystem2'.

    Returns:
        numpy.ndarray: A DSM matrix representing the interfaces between sub-systems.
    """
    matrix_size = len(sub_systems)
    dsm_matrix = np.zeros((matrix_size, matrix_size))

    # Populate the DSM matrix based on provided interfaces
    for interface in interfaces:
        system_a, system_b = interface.split('-')
        a_index = sub_systems.index(system_a)
        b_index = sub_systems.index(system_b)
        # Mark the interaction as '1' only in the specified direction
        dsm_matrix[a_index][b_index] = 1

    return dsm_matrix

# Step 3: Clustering

def perform_clustering(dsm_matrix, sub_systems):
    """
    Performs clustering on the DSM matrix using various algorithms.

    Args:
        dsm_matrix (numpy.ndarray): The DSM matrix.
        sub_systems (list): List of sub-system names.

    Returns:
        dict: A dictionary containing clustering labels for each clustering method.
    """
    # Flatten the matrix to use as input for clustering algorithms
    X = dsm_matrix.flatten().reshape(len(sub_systems), -1)
    
    # Determine the optimal number of clusters
    wcss = calculate_wcss(X)
    n_clusters = min(optimal_number_of_clusters(wcss), len(sub_systems) - 1)
    print(f"Optimal number of clusters: {n_clusters}")

    # Initialize and run various clustering algorithms
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    spectral = SpectralClustering(n_clusters=n_clusters, n_neighbors=min(n_clusters, len(X)), random_state=42, affinity='nearest_neighbors')
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    
    # Store the clustering labels
    return {
        'kmeans': kmeans.fit_predict(X),
        'spectral': spectral.fit_predict(X),
        'agglomerative': agglomerative.fit_predict(X)
    }

# 3.5 Elbow method 
def calculate_wcss(dsm_matrix):
    """
    Calculates the within-cluster sum of squares (WCSS) for different numbers of clusters.

    Args:
        dsm_matrix (numpy.ndarray): The DSM matrix.

    Returns:
        list: A list of WCSS values for each number of clusters.
    """
    wcss = []
    num_samples = len(dsm_matrix)
    # Limit the maximum number of clusters to either the number of samples or 10, whichever is smaller
    max_clusters = min(num_samples, 10)

    # Compute WCSS for each possible number of clusters
    for n in range(1, max_clusters):
        kmeans = KMeans(n_clusters=n, random_state=42).fit(dsm_matrix)
        wcss.append(kmeans.inertia_)

    return wcss

def optimal_number_of_clusters(wcss):
    """
    Determines the optimal number of clusters using the Elbow method.

    Args:
        wcss (list): A list of WCSS values for each number of clusters.

    Returns:
        int: Optimal number of clusters determined by the Elbow method.
    """
    # Calculate the difference in WCSS between each number of clusters
    wcss_diff = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
    # The elbow point is where the WCSS decrease sharply changes
    return wcss_diff.index(max(wcss_diff)) + 2

def plot_elbow(wcss):
    """
    Plots the Elbow graph to visualize the optimal number of clusters.

    Args:
        wcss (list): A list of WCSS values for each number of clusters.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(wcss)+1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

# Step 4: Rearrangement
    
def rearrange_dsm_matrix(dsm_matrix, sub_systems, labels):
    """
    Rearranges the DSM matrix to form a block diagonal structure based on clustering labels.

    Args:
        dsm_matrix (numpy.ndarray): The DSM matrix before rearrangement.
        sub_systems (list): The list of sub-system names.
        labels (numpy.ndarray): An array of cluster labels for the sub-systems.

    Returns:
        numpy.ndarray: The DSM matrix rearranged to bring elements of the same cluster closer to the diagonal.
        list: The list of sub-system names reordered to match the rearranged DSM matrix.
    """
    # Pair each sub-system with its corresponding label and original index
    labeled_sub_systems = list(zip(sub_systems, labels, range(len(sub_systems))))

    # Sort the sub-systems by cluster labels and then by their original index
    # This will group sub-systems by clusters and maintain order within each cluster
    labeled_sub_systems.sort(key=lambda x: (x[1], x[2]))

    # Extract the new order of sub-systems after sorting
    sorted_sub_systems = [sub_sys[0] for sub_sys in labeled_sub_systems]
    sorted_indices = [sub_sys[2] for sub_sys in labeled_sub_systems]

    # Rearrange the DSM matrix according to the new sub-system order
    rearranged_matrix = dsm_matrix[sorted_indices, :]
    rearranged_matrix = rearranged_matrix[:, sorted_indices]

    return rearranged_matrix, sorted_sub_systems

# Step 5: Visualization
def visualize_dsm(dsm_matrix, sub_systems, ax):
    cmap = plt.cm.get_cmap('Greys', 2)
    cax = ax.matshow(dsm_matrix, interpolation='nearest', cmap=cmap)
    ax.set_xticks(np.arange(len(sub_systems)))
    ax.set_yticks(np.arange(len(sub_systems)))
    ax.set_xticklabels(sub_systems, rotation=90)
    ax.set_yticklabels(sub_systems)
    for i in range(len(sub_systems)):
        ax.plot(i, i, marker='o', color='blue', markersize=10)
        ax.text(i, i, sub_systems[i][0], color='white', ha='center', va='center')
    ax.set_title('Initial DSM')

def visualize_clustered_dsm(dsm_matrix, sub_systems, labels, method_name, ax):
    """
    Visualizes the DSM matrix after clustering, including cluster boundaries.

    Args:
        dsm_matrix (numpy.ndarray): The DSM matrix.
        sub_systems (list): List of sub-system names.
        labels (numpy.ndarray): Cluster labels for each sub-system.
        method_name (str): Name of the clustering method used.
        ax (matplotlib.axes.Axes): The Axes object to plot on.
    """
    rearranged_matrix, sorted_sub_systems = rearrange_dsm_matrix(dsm_matrix, sub_systems, labels)

    # Create the heatmap for the rearranged DSM matrix
    cmap = plt.cm.get_cmap('Greys', 2)
    ax.matshow(rearranged_matrix, interpolation='nearest', cmap=cmap)

    # Setting up the gridlines, labels, and titles
    ax.set_xticks(np.arange(len(sorted_sub_systems)))
    ax.set_yticks(np.arange(len(sorted_sub_systems)))
    ax.set_xticklabels(sorted_sub_systems, rotation=90)
    ax.set_yticklabels(sorted_sub_systems)

    # Draw cluster boundaries on the DSM matrix
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        if indices:
            ax.axhline(min(indices)-0.5, linestyle='--', color='red')
            ax.axvline(min(indices)-0.5, linestyle='--', color='red')

    ax.set_title(f"Clustered DSM using {method_name}")

# Main execution
sub_systems, interfaces = get_user_input()
dsm_matrix = create_dsm_matrix(sub_systems, interfaces)

# Perform clustering and get labels
cluster_labels = perform_clustering(dsm_matrix, sub_systems)

# Determine the number of subplots required: 1 for the initial DSM + number of clustering methods
num_methods = len(cluster_labels) + 1
fig, axs = plt.subplots(1, num_methods, figsize=(15, 5))  # Adjust figsize as needed

# Visualize initial DSM
visualize_dsm(dsm_matrix, sub_systems, axs[0])

# Visualize clustered DSM for each clustering method
for i, (method, labels) in enumerate(cluster_labels.items(), start=1):
    visualize_clustered_dsm(dsm_matrix, sub_systems, labels, method, axs[i])

plt.tight_layout()
plt.show()