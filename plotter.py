import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load your HDF5 file
with h5py.File('data.h5', 'r') as file:
    # Assuming the dataset you want to plot is named 'dataset_name'
    # Replace 'dataset_name' with the actual name of your dataset
    data = file['dataset_name'][:]

# Check if data is 2D
if data.ndim != 2:
    raise ValueError("Data is not 2D. Adjust your dataset or the script accordingly.")

# Create the plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(data, cmap='viridis')  # You can choose different colormaps
plt.colorbar()  # To show the color scale
plt.title("2D Plot of HDF5 Data")
plt.xlabel("X-axis label")
plt.ylabel("Y-axis label")

# Save the plot as a PNG file
plt.savefig("plot.png", dpi=300)

# Optionally, display the plot
# plt.show()
