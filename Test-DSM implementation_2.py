import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN

# Step 1: Data Collection
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

# Step 2: Matrix Creation
def create_dsm_matrix(sub_systems, interfaces):
    # Create a blank DSM matrix
    matrix_size = len(sub_systems)
    dsm_matrix = np.zeros((matrix_size, matrix_size))

    # Populate the DSM matrix based on interfaces
    for interface in interfaces:
        system_a, system_b = interface.split('-')
        a_index = sub_systems.index(system_a)
        b_index = sub_systems.index(system_b)
        # Ensure to mark interaction for both elements
        dsm_matrix[a_index][b_index] = 1
        dsm_matrix[b_index][a_index] = 1  # Assuming bidirectional interaction

    return dsm_matrix

# Step 3: Clustering
def perform_clustering(dsm_matrix, sub_systems):
    # Flatten the matrix to get the feature vectors
    X = dsm_matrix.flatten().reshape(len(sub_systems), -1)
    
    # Calculate WCSS for a range of number of clusters
    wcss = calculate_wcss(X)
    
    # Determine the optimal number of clusters, ensuring it's less than the number of samples
    n_clusters = min(optimal_number_of_clusters(wcss), len(sub_systems) - 1)
    print(f"Optimal number of clusters: {n_clusters}")

    # Initialize and run K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Initialize and run Spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, n_neighbors=min(n_clusters, len(X)), random_state=42, affinity='nearest_neighbors')
    spectral_labels = spectral.fit_predict(X)
    
    # Initialize and run Agglomerative clustering
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_labels = agglomerative.fit_predict(X)
    
    # Collect and return the clusters for each algorithm
    clusters = {
        'kmeans': {},
        'spectral': {},
        'agglomerative': {}
    }
    
    for i, label in enumerate(kmeans_labels):
        if label not in clusters['kmeans']:
            clusters['kmeans'][label] = []
        clusters['kmeans'][label].append(sub_systems[i])
    
    for i, label in enumerate(spectral_labels):
        if label not in clusters['spectral']:
            clusters['spectral'][label] = []
        clusters['spectral'][label].append(sub_systems[i])
    
    for i, label in enumerate(agglomerative_labels):
        if label not in clusters['agglomerative']:
            clusters['agglomerative'][label] = []
        clusters['agglomerative'][label].append(sub_systems[i])
    
    return {
        'kmeans': kmeans_labels,
        'spectral': spectral_labels,
        'agglomerative': agglomerative_labels
    }
# ========================================
# 3.5 Elbow method 

def calculate_wcss(dsm_matrix):
    wcss = []
    num_samples = len(dsm_matrix)
    max_clusters = min(num_samples, 10)  # Limit the maximum number of clusters

    for n in range(1, max_clusters):
        kmeans = KMeans(n_clusters=n, random_state=42).fit(dsm_matrix)
        wcss.append(kmeans.inertia_)
    return wcss

# chosing optimal number of clusters

def optimal_number_of_clusters(wcss):
    # This function calculates the difference between each WCSS and finds the largest one
    wcss_diff = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
    return wcss_diff.index(max(wcss_diff)) + 2  # +2 because index 0 corresponds to 1 cluster


def plot_elbow(wcss):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(wcss)+1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # Within-cluster sum of squares
    plt.show()

# =========================================

# Step 4: Rearrangement:
    
def rearrange_dsm_matrix(dsm_matrix, sub_systems, labels):
    # Create a mapping of sub-systems to their cluster labels
    cluster_mapping = {sub_system: str(label) for sub_system, label in zip(sub_systems, labels)}

    # Sort sub-systems by their cluster labels
    sorted_sub_systems = sorted(sub_systems, key=lambda x: cluster_mapping[x])

    # Rearrange the DSM matrix according to the sorted sub-systems
    sorted_indices = [sub_systems.index(sys) for sys in sorted_sub_systems]
    rearranged_matrix = dsm_matrix[sorted_indices, :]
    rearranged_matrix = rearranged_matrix[:, sorted_indices]

    return rearranged_matrix, sorted_sub_systems

def visualize_clustered_dsm(dsm_matrix, sub_systems, labels, method_name):
    rearranged_matrix, sorted_sub_systems = rearrange_dsm_matrix(dsm_matrix, sub_systems, labels)

    # Create a figure and a subplot
    fig, ax = plt.subplots()

    # Use a colormap to represent the presence of an interface (black for presence)
    cmap = plt.cm.get_cmap('Greys', 2)  # 2 distinct colors for 0 and 1 values

    # Create the heatmap for the rearranged DSM matrix
    ax.matshow(rearranged_matrix, interpolation='nearest', cmap=cmap)

    # Set gridlines based on matrix dimensions
    ax.set_xticks(np.arange(len(sorted_sub_systems)))
    ax.set_yticks(np.arange(len(sorted_sub_systems)))

    # Label the grid with sorted sub-system names
    ax.set_xticklabels(sorted_sub_systems, rotation=90)
    ax.set_yticklabels(sorted_sub_systems)

    # Draw cluster boundaries
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        if indices:
            ax.axhline(min(indices)-0.5, linestyle='--', color='red')
            ax.axvline(min(indices)-0.5, linestyle='--', color='red')

    plt.title(f"Clustered DSM using {method_name}")
    plt.show()

# =========================================

# Step 5: Visualization
def visualize_clustered_dsm(dsm_matrix, sub_systems, labels, method_name, ax):
    rearranged_matrix, sorted_sub_systems = rearrange_dsm_matrix(dsm_matrix, sub_systems, labels)

    # Use a colormap to represent the presence of an interface (black for presence)
    cmap = plt.cm.get_cmap('Greys', 2)  # 2 distinct colors for 0 and 1 values

    # Create the heatmap for the rearranged DSM matrix
    ax.matshow(rearranged_matrix, interpolation='nearest', cmap=cmap)

    # Set gridlines based on matrix dimensions
    ax.set_xticks(np.arange(len(sorted_sub_systems)))
    ax.set_yticks(np.arange(len(sorted_sub_systems)))

    # Label the grid with sorted sub-system names
    ax.set_xticklabels(sorted_sub_systems, rotation=90)
    ax.set_yticklabels(sorted_sub_systems)

    # Draw cluster boundaries
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        if indices:
            ax.axhline(min(indices)-0.5, linestyle='--', color='red')
            ax.axvline(min(indices)-0.5, linestyle='--', color='red')

    ax.set_title(f"Clustered DSM using {method_name}")


# ==================================
# Main execution

'''
sub_systems, interfaces = get_user_input()
dsm_matrix = create_dsm_matrix(sub_systems, interfaces)
clusters = perform_clustering(dsm_matrix, sub_systems)

for method, clustering in clusters.items():
    print(f"\n{method.upper()} Clustering:")
    for cluster_id, cluster_elements in clustering.items():
        print(f"Cluster {cluster_id}: {', '.join(cluster_elements)}")


print("DSM Matrix:\n", dsm_matrix)

wcss = calculate_wcss(dsm_matrix)
plot_elbow(wcss)


# Visualize initial DSM
visualize_dsm(dsm_matrix, sub_systems)

# Perform clustering
cluster_labels = perform_clustering(dsm_matrix, sub_systems)

# Visualize clustered DSM for each clustering method
for method, labels in cluster_labels.items():
    visualize_clustered_dsm(dsm_matrix, sub_systems, labels, method)

'''

# ======================================
# Main execution 2
# ======================================

# Main execution
sub_systems, interfaces = get_user_input()
dsm_matrix = create_dsm_matrix(sub_systems, interfaces)

# Perform clustering and get labels
cluster_labels = perform_clustering(dsm_matrix, sub_systems)

# Determine the number of subplots required: 1 for the initial DSM + number of clustering methods
num_methods = len(cluster_labels) + 1
fig, axs = plt.subplots(1, num_methods, figsize=(15, 5))  # Adjust figsize as needed

# Visualize initial DSM
visualize_clustered_dsm(dsm_matrix, sub_systems, axs[0])

# Visualize clustered DSM for each clustering method
for i, (method, labels) in enumerate(cluster_labels.items(), start=1):
    visualize_clustered_dsm(dsm_matrix, sub_systems, labels, method, axs[i])

plt.tight_layout()
plt.show()