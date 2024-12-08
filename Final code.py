import torch
import torch.nn as nn
import os
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog,Tk
from scipy.interpolate import interp1d
import numpy as np

# Function to upload the image
def upload_image():
    # Create a temporary Tkinter root window to handle the file dialog
    root = Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Ensure the dialog is on top
    root.update()    # Ensure the window processes events

    # Open the file dialog
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.gif")]
    )
    root.destroy()  # Destroy the temporary root window

    if file_path:
        print(f"Image uploaded: {file_path}")
        return file_path
    else:
        print("No file was selected.")
        return None

# Upload the image
file_path = upload_image()

if file_path:  # Proceed only if a file was uploaded
    # Load the YOLO model
    model = YOLO("segmentation_model.pt")  # Segmentation model

    # Predict the segmentation
    results = model.predict(
        source=file_path,  # Path to your test images
        conf=0.4  # Confidence threshold for predictions
    )

    # Save results to a text file
    with open("segmentation_output.txt", "w") as file:
        for result in results:
            if hasattr(result, "masks") and result.masks:
                for mask in result.masks.xy:
                    file.write("Polygon Coordinates:\n")
                    for x, y in mask:
                        file.write(f"{x}, {y}\n")
                    file.write("\n")
            else:
                file.write("No segmentation mask found.\n")

    print("Results saved to segmentation_output.txt.")
else:
    print("No file selected. Exiting.")
 
# Load the data
file_path = r"segmentation_output.txt"

# Initialize lists to store x and y coordinates
x_values = []
y_values = []

# Read and process the file
with open(file_path, 'r') as file:
    for line in file:
        # Check if the line contains numeric data
        if ',' in line:
            try:
                x, y = map(float, line.strip().split(','))  # Parse x, y values
                x_values.append(x)
                y_values.append(y)
            except ValueError:
                continue  # Skip lines that are not numeric

# Ensure there are coordinates to process
if not x_values or not y_values:
    raise ValueError("No valid coordinates found in the file.")

# Step 4: Calculate cumulative distances along the boundary
distances = np.sqrt(np.diff(x_values)**2 + np.diff(y_values)**2)
cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

# Step 5: Interpolate to generate 200 evenly spaced points
num_points = 200
even_distances = np.linspace(0, cumulative_distances[-1], num_points)

# Interpolate x and y based on cumulative distances
new_x = np.interp(even_distances, cumulative_distances, x_values)
new_y = np.interp(even_distances, cumulative_distances, y_values)

# Step 6: Combine into a DataFrame
interpolated_coordinates = pd.DataFrame({'x': new_x, 'y': new_y})

# Save the interpolated coordinates to a new CSV file
output_file = 'Segmentation_Preprocesses.csv'
interpolated_coordinates.to_csv(output_file, index=False)

print(f"Interpolated coordinates saved to {output_file}")

# Save the interpolated coordinates to an Excel file
excel_output_file = 'shape_for_gripping.xlsx'
interpolated_coordinates.to_excel(excel_output_file, index=False)

print(f"Interpolated coordinates saved to {output_file} and {excel_output_file}")

class GrippingPointGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GrippingPointGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # New layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))  # New layer
        x = global_mean_pool(x, batch)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

Predicting_shape = pd.read_excel('shape_for_gripping.xlsx').values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GrippingPointGNN(input_dim=2, hidden_dim=128, output_dim=8).to(device)

# Ensure the model is loaded
checkpoint_path = "grasping_model.pth"
checkpoint = torch.load(checkpoint_path)  # Load the checkpoint
model.load_state_dict(checkpoint["model_state_dict"])  # Load the model's state_dict
model.eval()  # Set the model to evaluation mode

# Convert the preprocessed shape data to a PyTorch-Geometric Data object
node_features = torch.tensor(Predicting_shape, dtype=torch.float)  # Node features from Excel
# Create edges (this is an example; adjust based on your shape's connectivity)
num_nodes = node_features.size(0)
edge_index = torch.tensor(
    [[i, (i + 1) % num_nodes] for i in range(num_nodes)] +  # Connect nodes in a ring
    [[(i + 1) % num_nodes, i] for i in range(num_nodes)],   # Reverse connections
    dtype=torch.long
).t().contiguous()

# Define the graph
first_graph = Data(x=node_features, edge_index=edge_index)

# Make the prediction
with torch.no_grad():
    predicted_gripping_points = model(first_graph.x, first_graph.edge_index, first_graph.batch)

# Reshape the predicted gripping points to (4, 2)
predicted_gripping_points = predicted_gripping_points.reshape(-1, 2).cpu().numpy()

# Extract boundary points for visualization
boundary_points = first_graph.x.cpu().numpy()  # Boundary points for visualization

# Ground truth gripping points (if available)
ground_truth = None
if first_graph.y is not None:
    ground_truth = first_graph.y.cpu().numpy()

# Map each predicted gripping point to the closest boundary point
adjusted_gripping_points = []
for predicted_point in predicted_gripping_points:
    # Compute distances to all boundary points
    distances = np.linalg.norm(boundary_points - predicted_point, axis=1)
    # Find the index of the closest boundary point
    closest_index = np.argmin(distances)
    # Add the closest boundary point to the adjusted list
    adjusted_gripping_points.append(boundary_points[closest_index])

# Convert to a NumPy array for easier handling
adjusted_gripping_points = np.array(adjusted_gripping_points)

# Visualize the data
plt.figure(figsize=(8, 6))

# Plot boundary points
plt.scatter(boundary_points[:, 0], boundary_points[:, 1], label="Boundary Points", color='blue', alpha=0.5)

# Plot predicted gripping points (before adjustment)
plt.scatter(predicted_gripping_points[:, 0], predicted_gripping_points[:, 1], label="Predicted Gripping Points (Raw)", color='green', marker='x')

# Plot adjusted gripping points (after mapping to boundary)
plt.scatter(adjusted_gripping_points[:, 0], adjusted_gripping_points[:, 1], label="Adjusted Gripping Points", color='orange', marker='o')

# If ground truth is available, plot it
if ground_truth is not None:
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label="Ground Truth Gripping Points", color='red')

# Set labels, title, and legend
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Grasping Points Prediction Visualization')
plt.legend()
plt.show()

