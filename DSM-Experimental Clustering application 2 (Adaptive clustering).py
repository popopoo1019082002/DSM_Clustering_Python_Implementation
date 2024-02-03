import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

# Step 1: Data Collection
def get_user_input():
    """
    Collects user input for sub-systems and interfaces in a system engineering project.

    Returns:
        sub_systems (list): List of sub-system names.
        interfaces (list): List of interfaces between sub-systems.
    """
    sub_systems = []
    interfaces = []

    # Get the number of sub-systems
    num_sub_systems = int(input("Enter the number of sub-systems: "))

    # Collect the names of each sub-system
    for i in range(num_sub_systems):
        sub_system = input(f"Enter sub-system {i+1}: ")
        sub_systems.append(sub_system)

    # Collect interfaces between sub-systems
    print("\nEnter interfaces between sub-systems (e.g., 'Subsystem1-Subsystem2'). Type 'done' when finished.")
    while True:
        interface = input("Enter interface: ")
        if interface.lower() == 'done':
            break
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
    Performs clustering on the DSM matrix.

    Args:
        dsm_matrix (numpy.ndarray): The DSM matrix.
        sub_systems (list): List of sub-system names.

    Returns:
        dict: A dictionary containing the clustering results for each method.
    """
    clustering_results = {}
    X = dsm_matrix.flatten().reshape(len(sub_systems), -1)

    # Determine the maximum number of neighbors for Spectral Clustering
    # It cannot be more than the number of samples (sub-systems)
    max_neighbors = min(len(sub_systems) - 1, 10)  # 10 is arbitrary, adjust based on your requirements

    # Perform K-means clustering
    optimal_k = optimal_number_of_clusters(calculate_wcss(X))
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clustering_results['kmeans'] = kmeans.fit_predict(X)

    # Perform Spectral clustering
    optimal_spectral_clusters = min(optimal_k, max_neighbors)
    spectral = SpectralClustering(n_clusters=optimal_spectral_clusters, n_neighbors=max_neighbors, random_state=42, affinity='nearest_neighbors')
    clustering_results['spectral'] = spectral.fit_predict(X)

    # Perform Agglomerative clustering
    agglomerative = AgglomerativeClustering(n_clusters=optimal_k)
    clustering_results['agglomerative'] = agglomerative.fit_predict(X)

    return clustering_results

def estimate_optimal_clusters_spectral(X):
    # Implement your estimation method for the optimal number of clusters for Spectral Clustering
    # Placeholder for the actual method
    return int(np.sqrt(len(X)))

def estimate_optimal_clusters_agglomerative(X):
    # Implement your estimation method for the optimal number of clusters for Agglomerative Clustering
    # Placeholder for the actual method
    return int(np.sqrt(len(X)))

def print_cluster_details(labels, sub_systems, method_name):
    """
    Prints the details of the clusters formed by a clustering method.

    Args:
        labels (numpy.ndarray): Array of cluster labels for the sub-systems.
        sub_systems (list): List of sub-system names.
        method_name (str): The name of the clustering method used.
    """
    print(f"\n{method_name} Clustering Details:")
    print(f"{'Cluster ID':<12}{'Sub-systems':<50}{'Count'}")
    print("-" * 70)
    
    # Pair sub-systems with their respective cluster labels
    labeled_sub_systems = list(zip(sub_systems, labels))
    
    # Create a dictionary to hold cluster details
    clusters = {}
    for sub_system, label in labeled_sub_systems:
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sub_system)
    
    # Print out the details for each cluster
    for label, sub_systems in clusters.items():
        print(f"{label:<12}{', '.join(sub_systems):<50}{len(sub_systems)}")
    
    print(f"{'Total Clusters:':<12}{len(clusters)}")


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
    Rearranges the DSM matrix based on clustering labels and prints out the new order of sub-systems for debugging.

    Args:
        dsm_matrix (numpy.ndarray): The DSM matrix before rearrangement.
        sub_systems (list): The list of sub-system names.
        labels (numpy.ndarray): An array of cluster labels for the sub-systems.

    Returns:
        rearranged_matrix (numpy.ndarray): The DSM matrix rearranged to bring elements of the same cluster closer to the diagonal.
        sorted_sub_systems (list): The list of sub-system names reordered to match the rearranged DSM matrix.
    """
    # Pair each sub-system with its corresponding label and original index for sorting
    labeled_sub_systems = list(zip(sub_systems, labels, range(len(sub_systems))))

    # Sort the sub-systems by cluster labels and then by their original index
    labeled_sub_systems.sort(key=lambda x: (x[1], x[2]))

    # Extract the sorted order of sub-systems and their original indices
    sorted_sub_systems = [item[0] for item in labeled_sub_systems]
    sorted_indices = [item[2] for item in labeled_sub_systems]

    # Rearrange the DSM matrix according to the sorted indices to align with cluster blocks
    rearranged_matrix = dsm_matrix[sorted_indices, :]
    rearranged_matrix = rearranged_matrix[:, sorted_indices]

    # Debug: print the sorted sub-systems and their cluster labels
    print("\nSorted Sub-systems and their Labels:")
    for sys, label in zip(sorted_sub_systems, labels):
        print(f"Sub-system: {sys}, Label: {label}")

    return rearranged_matrix, sorted_sub_systems

# Step 5: Visualization
def visualize_dsm(dsm_matrix, sub_systems, ax):
    """
    Visualize the Design Structure Matrix (DSM).

    Args:
        dsm_matrix (numpy.ndarray): The DSM matrix to visualize.
        sub_systems (list): The list of subsystem names.
        ax (matplotlib.axes.Axes): The Axes object to plot on.
    """
    cmap = plt.cm.get_cmap('Greys', 2)  # Greyscale colormap for 0 and 1 values
    ax.matshow(dsm_matrix, interpolation='nearest', cmap=cmap)
    ax.set_xticks(np.arange(len(sub_systems)))
    ax.set_yticks(np.arange(len(sub_systems)))
    ax.set_xticklabels(sub_systems, rotation=90)
    ax.set_yticklabels(sub_systems)
    for i in range(len(sub_systems)):
        # Check if subsystem name is not empty before accessing the character
        label = sub_systems[i][0] if sub_systems[i] else ''
        ax.text(i, i, label, color='white', ha='center', va='center')
    ax.set_title('Initial DSM')


def visualize_clustered_dsm(dsm_matrix, sub_systems, labels, method_name, ax):
    """
    Visualizes the DSM matrix after clustering, with rectangles delineating the cluster boundaries.

    Args:
        dsm_matrix (numpy.ndarray): The original DSM matrix.
        sub_systems (list): List of sub-system names.
        labels (numpy.ndarray): Cluster labels for each sub-system.
        method_name (str): Name of the clustering method used.
        ax (matplotlib.axes.Axes): The Axes object to plot on.

    Note:
        This function uses 'ax' to draw on a specific subplot within a larger figure.
    """
    # Rearrange the DSM matrix to group sub-systems by their clusters
    rearranged_matrix, sorted_sub_systems = rearrange_dsm_matrix(dsm_matrix, sub_systems, labels)

    # Create a heatmap for the rearranged DSM matrix using a grayscale colormap
    cmap = plt.cm.get_cmap('Greys', 2)  # Two distinct colors for 0 (no connection) and 1 (connection)
    ax.matshow(rearranged_matrix, interpolation='nearest', cmap=cmap)

    # Configure the subplot with gridlines and labels
    ax.set_xticks(np.arange(len(sorted_sub_systems)))
    ax.set_yticks(np.arange(len(sorted_sub_systems)))
    ax.set_xticklabels(sorted_sub_systems, rotation=90)
    ax.set_yticklabels(sorted_sub_systems)

    # Add rectangles around the clusters to highlight them on the DSM
    start_idx = 0
    for label in sorted(set(labels)):
        # Find the range of indices for the current label (cluster)
        indices = [i for i, x in enumerate(labels) if x == label]
        # Calculate the size of the cluster and draw a rectangle around it
        cluster_size = len(indices)
        rect = plt.Rectangle((start_idx-0.5, start_idx-0.5), cluster_size, cluster_size, fill=False, edgecolor='red', linestyle='--')
        ax.add_patch(rect)
        start_idx += cluster_size

    # Title the subplot with the name of the clustering method
    ax.set_title(f"Clustered DSM using {method_name}")

def show_cluster_details_graphically(cluster_labels, sub_systems):
    """
    Shows a graphical representation of cluster details in a pop-up window.

    Args:
        cluster_labels (dict): A dictionary with clustering method names as keys and arrays of labels as values.
        sub_systems (list): List of sub-system names.
    """
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, len(cluster_labels), figsize=(5 * len(cluster_labels), 5))
    
    # If there's only one method, axs may not be an array
    if len(cluster_labels) == 1:
        axs = [axs]

    # Go through each clustering method and its labels
    for ax, (method_name, labels) in zip(axs, cluster_labels.items()):
        # Get cluster details
        clusters = {label: [] for label in labels}
        for sub_system, label in zip(sub_systems, labels):
            clusters[label].append(sub_system)

        # Create a table of cluster details
        cell_text = []
        for cluster_id, members in clusters.items():
            cell_text.append([cluster_id, ', '.join(members), len(members)])
        
        # Sort the table by cluster ID
        cell_text.sort(key=lambda x: x[0])

        # Add a table at the bottom of the axes
        table = ax.table(cellText=cell_text, colLabels=['Cluster ID', 'Sub-systems', 'Count'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Hide axes
        ax.axis('off')
        ax.axis('tight')

        # Set the title for the subplot
        ax.set_title(f"{method_name} Clustering Details")

    # Adjust layout and show the figure
    plt.tight_layout()
    plt.show()

# Main Execution Section

try:
    # Collect user input for sub-systems and interfaces
    sub_systems, interfaces = get_user_input()

    # Create the initial DSM matrix based on the user input
    dsm_matrix = create_dsm_matrix(sub_systems, interfaces)

    # Perform clustering on the DSM matrix
    cluster_labels = perform_clustering(dsm_matrix, sub_systems)

    # Determine the number of subplots required: 1 for the initial DSM + number of clustering methods
    num_methods = len(cluster_labels) + 1
    fig, axs = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5))  # Adjust figsize as needed

    # Visualize the initial DSM
    visualize_dsm(dsm_matrix, sub_systems, axs[0])
    axs[0].set_title('Initial DSM')

    # Visualize the clustered DSM for each clustering method
    for i, (method, labels) in enumerate(cluster_labels.items(), start=1):
        # Rearrange the matrix based on clustering results
        rearranged_matrix, sorted_sub_systems = rearrange_dsm_matrix(dsm_matrix, sub_systems, labels)
        # Visualize the rearranged matrix
        visualize_clustered_dsm(rearranged_matrix, sorted_sub_systems, labels, method, axs[i])

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()

    # Show cluster details graphically in a pop-up window
    show_cluster_details_graphically(cluster_labels, sub_systems)
except Exception as e:
    print(f"An error occurred: {e}")