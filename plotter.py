import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# List all files in the current directory
all_files = os.listdir('.')

# Filter for .h5 files
h5_files = [file for file in all_files if file.endswith('.h5')]

for file_name in h5_files:

    # Load the HDF5 file
    with h5py.File(file_name, 'r') as file:
        data = file['data'][:]
        time = file['time'][()]

    # Print dimensions
    print(data.shape)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(data.transpose(), cmap='viridis')
    plt.colorbar()  
    plt.title(f"solution at time = "+str(time))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the plot as a PNG file with the same name as the .h5 file
    plt.savefig(f"{file_name.split('.')[0]}.png", dpi=300)

    # Close the plot to avoid overlapping issues
    plt.close()
