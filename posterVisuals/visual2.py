import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import random
import os
from scipy.ndimage import gaussian_filter

def load_image(image_path):
    """Load thermal image from file"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    return gray

def generate_thermal_sequence(base_img, num_frames=8, max_shift=3, temp_variation=1.0):
    """Generate simulated thermal image sequence with variations"""
    sequence = []
    height, width = base_img.shape
    
    for i in range(num_frames):
        # Start with a copy of the base image
        frame = base_img.copy()
        
        # Apply small random shift
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(frame, M, (width, height))
        
        # Apply temperature variations
        temp_offset = random.uniform(-temp_variation, temp_variation)
        frame = np.clip(frame.astype(float) + temp_offset, 0, 255).astype(np.uint8)
        
        # Add smooth thermal noise
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        smooth_noise = gaussian_filter(noise, sigma=2)
        frame = np.clip(frame.astype(np.int16) + smooth_noise, 0, 255).astype(np.uint8)
        
        # Add to sequence
        sequence.append(frame)
    
    return sequence

def pixel_to_temperature(pixel_value, vmin, vmax, min_temp=-4, max_temp=12):
    """Map pixel values to temperature range"""
    # Map from [vmin, vmax] to [min_temp, max_temp]
    return min_temp + (max_temp - min_temp) * (pixel_value - vmin) / (vmax - vmin)

def create_3d_pixel_cloud(thermal_sequence, downsample_factor=6, point_size=10, 
                         alpha=0.7, sample_rate=0.3, min_temp=-4, max_temp=12,
                         colormap='inferno'):
    """Create 3D pixel cloud visualization"""
    num_frames = len(thermal_sequence)
    height, width = thermal_sequence[0].shape
    
    # Downsample for visualization
    ds_height, ds_width = height // downsample_factor, width // downsample_factor
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get value range across all images
    vmin = min([np.min(img) for img in thermal_sequence])
    vmax = max([np.max(img) for img in thermal_sequence])
    
    # Create normalized temperature range
    norm = Normalize(min_temp, max_temp)
    cmap = plt.cm.get_cmap(colormap)
    
    # Collect all points for plotting
    x_coords, y_coords, z_coords, colors = [], [], [], []
    
    # Process each frame
    for t, img in enumerate(thermal_sequence):
        # Downsample the image
        ds_img = cv2.resize(img, (ds_width, ds_height), interpolation=cv2.INTER_AREA)
        
        # Sample points
        for i in range(ds_height):
            for j in range(ds_width):
                if random.random() < sample_rate:
                    # Get pixel value and convert to temperature
                    pixel_value = ds_img[i, j]
                    temp = pixel_to_temperature(pixel_value, vmin, vmax, min_temp, max_temp)
                    
                    # Store coordinates
                    x_coords.append(j)
                    y_coords.append(i)
                    z_coords.append(t)
                    
                    # Store color based on temperature
                    colors.append(cmap(norm(temp)))
    
    # Plot all points at once for efficiency
    ax.scatter(x_coords, y_coords, z_coords, c=colors, s=point_size, alpha=alpha)
    
    # Set labels and title
    ax.set_xlabel('X Pixel Coordinate', fontsize=12)
    ax.set_ylabel('Y Pixel Coordinate', fontsize=12)
    ax.set_zlabel('Time Frame', fontsize=12)
    ax.set_title('Multi-temporal Thermal Analysis:\nPixel Cloud Representation', 
                fontsize=14, fontweight='bold')
    
    # Add a color bar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.08, aspect=30)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Set view angle
    ax.view_init(30, 45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig

def create_voxel_visualization(thermal_sequence, downsample_factor=6, size_range=(5, 20), 
                              min_temp=-4, max_temp=12, colormap='inferno'):
    """Create a 3D voxel visualization with the same colormap as pixel cloud"""
    # Downsample
    height, width = thermal_sequence[0].shape
    ds_height, ds_width = height // downsample_factor, width // downsample_factor
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create 3D array for voxel data
    num_frames = len(thermal_sequence)
    voxel_data = np.zeros((ds_height, ds_width, num_frames))
    
    # Fill the voxel data
    for t, img in enumerate(thermal_sequence):
        ds_img = cv2.resize(img, (ds_width, ds_height), interpolation=cv2.INTER_AREA)
        voxel_data[:,:,t] = ds_img
    
    # Get value range across all images
    vmin = np.min(voxel_data)
    vmax = np.max(voxel_data)
    
    # Create normalized temperature range
    norm = Normalize(min_temp, max_temp)
    cmap = plt.cm.get_cmap(colormap)
    
    # Collect all points for plotting
    x_coords, y_coords, z_coords, colors, sizes = [], [], [], [], []
    
    # Process voxel data
    for x in range(ds_width):
        for y in range(ds_height):
            for z in range(num_frames):
                # Sample fewer points for clarity
                if random.random() < 0.4:
                    # Get pixel value and convert to temperature
                    pixel_value = voxel_data[y, x, z]
                    temp = pixel_to_temperature(pixel_value, vmin, vmax, min_temp, max_temp)
                    
                    # Store coordinates
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                    
                    # Store color based on temperature
                    colors.append(cmap(norm(temp)))
                    
                    # Size proportional to temperature
                    temp_ratio = (temp - min_temp) / (max_temp - min_temp)
                    size = size_range[0] + (size_range[1] - size_range[0]) * temp_ratio
                    sizes.append(size)
    
    # Plot all points at once for efficiency
    ax.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Time Frame', fontsize=12)
    ax.set_title('3D Thermal Voxel Representation\nFor Building Analysis', 
                fontsize=14, fontweight='bold')
    
    # Add a color bar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.08, aspect=30)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Set view angle
    ax.view_init(30, 45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig

def main():
    """Main function to run the visualization"""
    # Configuration
    image_path = '/home/fabio/University/research/SP2025_RABI/posterVisuals/Picture1.png'
    output_dir = '/home/fabio/University/research/SP2025_RABI/posterVisuals/'
    
    # Temperature range and colormap settings
    min_temp = -4
    max_temp = 12
    colormap = 'inferno'  # Use the same colormap for both visualizations
    
    try:
        # Load the image
        thermal_img = load_image(image_path)
        
        # Generate thermal sequence
        sequence = generate_thermal_sequence(thermal_img)
        
        # Create and save 3D pixel cloud visualization
        fig_cloud = create_3d_pixel_cloud(
            sequence, 
            min_temp=min_temp, 
            max_temp=max_temp,
            colormap=colormap
        )
        cloud_path = os.path.join(output_dir, "thermal_pixel_cloud.png")
        fig_cloud.savefig(cloud_path, dpi=300, bbox_inches='tight')
        plt.close(fig_cloud)
        
        # Create and save voxel visualization with the same colormap
        fig_voxel = create_voxel_visualization(
            sequence, 
            min_temp=min_temp, 
            max_temp=max_temp,
            colormap=colormap
        )
        voxel_path = os.path.join(output_dir, "thermal_voxel.png") 
        fig_voxel.savefig(voxel_path, dpi=300, bbox_inches='tight')
        plt.close(fig_voxel)
        
        print(f"Visualizations saved to {output_dir}")
        print(f"- Pixel cloud: {cloud_path}")
        print(f"- Voxel visualization: {voxel_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()