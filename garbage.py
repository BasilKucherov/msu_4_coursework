import numpy as np
import matplotlib.pyplot as plt
import torch

'''
Clusters generator for validation loss functions
'''
# Generate synthetic data
np.random.seed(0)
n_samples = 300
cluster_radius = 10
center_distances = [0, 10, 20, 30]

# Generate random angles for the clusters
cluster1_angles = np.random.uniform(0, 2 * np.pi, n_samples // 2)
cluster2_angles = np.random.uniform(0, 2 * np.pi, n_samples // 2)

cluster1_radiuses = np.random.uniform(0, cluster_radius, n_samples // 2)
cluster2_radiuses = np.random.uniform(0, cluster_radius, n_samples // 2)

# Create labels (0 for the first cluster, 1 for the second cluster)
labels = torch.tensor([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Define a list of different margin values
margins = [0.1, 0.5, 1.0, 2, 5, 10]  # Add more margins if needed

# Create subplots for different margins in rows and different clusters in columns
num_rows = len(margins)
num_cols = len(center_distances)  # Two clusters

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i, margin in enumerate(margins):
    for j, center_distance in enumerate(center_distances):  # Loop through clusters
        # Create cluster 1 data points
        cluster1_x = cluster1_radiuses * np.cos(cluster1_angles)
        cluster1_y = cluster1_radiuses * np.sin(cluster1_angles)

        # Create cluster 2 data points
        cluster2_x = center_distance + cluster2_radiuses * np.cos(cluster2_angles)
        cluster2_y = cluster2_radiuses * np.sin(cluster2_angles)

        # Concatenate cluster data
        data_x = np.concatenate([cluster1_x, cluster2_x])
        data_y = np.concatenate([cluster1_y, cluster2_y])

        # Create PyTorch tensor from data
        data = torch.tensor(np.vstack((data_x, data_y)).T, dtype=torch.float32)

        loss_fn = TripletLoss(margin=margin)
        loss = loss_fn(data, labels)

        axs[i, j].scatter(data_x, data_y, c=labels)
        axs[i, j].set_title(f"loss {loss}")
        axs[i, j].set_xlabel("X-axis")
        axs[i, j].set_ylabel("Y-axis")
        axs[i, j].set_aspect('equal')

plt.tight_layout()
plt.show()


''' Plot some examples from dataset'''
import matplotlib.pyplot as plt
import torch

# Example batch
# batch = {
#     'input': [torch.randn((3, 3)) for _ in range(6)],  # List of 2D torch tensors
#     'target': [0,1,2,3,4,5],  # List of labels
# }

# Define the number of rows and columns for the subplot grid
n_rows = 8
n_cols = 4

# Create a figure and subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 20))
# Flatten the axes if it's a 1D array (for the case when there's only 1 row or 1 column)
# if n_rows == 1 or n_cols == 1:
#     axes = axes.reshape(-1)

# Plot each tensor-label pair
for i in range(n_rows):
    for j in range(n_cols):
        tensor = batch['input'][i * n_cols + j]
        label = train_dataset.get_class_from_idx(int(batch['target'][i * n_cols + j]))
        
        ax = axes[i][j]
        # print(f"{i}: {ax}")
        ax.imshow(tensor, cmap='viridis')  # You can use any colormap you prefer
        ax.set_title(label)
        ax.axis('off')  # Turn off axis labels and ticks
# for i, (tensor, label) in enumerate(zip(batch['input'], batch['target'])):
    

# Add spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
