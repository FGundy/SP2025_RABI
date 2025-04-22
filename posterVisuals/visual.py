import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import random
import os
from scipy.ndimage import gaussian_filter

class ThermalSequenceVisualizer:
    """
    Class to generate and visualize multi-temporal thermal image data for scientific publications
    """
    
    def __init__(self, image_path, output_dir=None):
        """
        Initialize with the base thermal image path
        
        Args:
            image_path: Path to the original thermal image
            output_dir: Directory to save visualizations (defaults to same as input)
        """
        self.image_path = image_path
        if output_dir is None:
            self.output_dir = os.path.dirname(image_path)
        else:
            self.output_dir = output_dir
            
        # Load the thermal image
        self.base_img = self._load_image()
        
        # Convert to grayscale if needed
        if len(self.base_img.shape) == 3:
            self.base_gray = cv2.cvtColor(self.base_img, cv2.COLOR_BGR2GRAY)
        else:
            self.base_gray = self.base_img.copy()
            
        # Set default parameters
        self.thermal_sequence = None
        
    def _load_image(self):
        """Load and validate the thermal image"""
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read thermal image at {self.image_path}")
        return img
    
    def generate_sequence(self, num_frames=8, max_shift=3, temp_variation=1.0, 
                          add_noise=True, noise_level=3):
        """
        Generate a sequence of simulated thermal images based on the original
        
        Args:
            num_frames: Number of frames to generate
            max_shift: Maximum pixel shift for spatial transformations
            temp_variation: Maximum temperature variation in degrees
            add_noise: Whether to add Gaussian noise
            noise_level: Standard deviation of the Gaussian noise
        
        Returns:
            self (for method chaining)
        """
        # Store the sequence of images
        sequence = []
        height, width = self.base_gray.shape
        
        for i in range(num_frames):
            # Start with a copy of the base image
            frame = self.base_gray.copy()
            
            # 1. Apply small random shift
            shift_x = random.randint(-max_shift, max_shift)
            shift_y = random.randint(-max_shift, max_shift)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            frame = cv2.warpAffine(frame, M, (width, height))
            
            # 2. Apply slight temperature variations (global)
            temp_offset = random.uniform(-temp_variation, temp_variation)
            frame = np.clip(frame.astype(float) + temp_offset, 0, 255).astype(np.uint8)
            
            # 3. Apply local temperature variations (using Gaussian blurring)
            if add_noise:
                # Random noise
                noise = np.random.normal(0, noise_level, frame.shape).astype(np.int16)
                
                # Smooth the noise to create more realistic temperature patterns
                smooth_noise = gaussian_filter(noise, sigma=2)
                
                # Apply the noise
                frame = np.clip(frame.astype(np.int16) + smooth_noise, 0, 255).astype(np.uint8)
            
            # Add to sequence
            sequence.append(frame)
        
        self.thermal_sequence = sequence
        return self
    
    def create_3d_pixel_cloud(self, downsample_factor=6, point_size=10, alpha=0.7, sample_rate=0.3,
                             min_temp=-4, max_temp=12):
        """
        Create a 3D pixel cloud visualization for publication
        
        Args:
            downsample_factor: Factor to downsample the images for visualization
            point_size: Size of the 3D points
            alpha: Transparency of points
            sample_rate: Fraction of points to sample (for clarity)
            min_temp: Minimum temperature in Celsius
            max_temp: Maximum temperature in Celsius
            
        Returns:
            Matplotlib figure
        """
        if self.thermal_sequence is None:
            raise ValueError("Thermal sequence not generated. Call generate_sequence() first.")
            
        num_frames = len(self.thermal_sequence)
        height, width = self.thermal_sequence[0].shape
        
        # Downsample for visualization
        ds_height, ds_width = height // downsample_factor, width // downsample_factor
        
        # Create figure
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare color mapping
        vmin = min([np.min(img) for img in self.thermal_sequence])
        vmax = max([np.max(img) for img in self.thermal_sequence])
        
        # Create a function to map pixel values to temperatures
        def map_to_temperature(pixel_value):
            # Map from [vmin, vmax] to [min_temp, max_temp]
            return min_temp + (max_temp - min_temp) * (pixel_value - vmin) / (vmax - vmin)
        
        # Collect all points for plotting
        x_coords, y_coords, z_coords, colors, temps = [], [], [], [], []
        
        # Process each frame
        for t, img in enumerate(self.thermal_sequence):
            # Downsample the image
            ds_img = cv2.resize(img, (ds_width, ds_height), interpolation=cv2.INTER_AREA)
            
            # Sample points
            for i in range(ds_height):
                for j in range(ds_width):
                    if random.random() < sample_rate:
                        pixel_value = ds_img[i, j]
                        temp_value = map_to_temperature(pixel_value)
                        temps.append(temp_value)
                        
                        x_coords.append(j)
                        y_coords.append(i)
                        z_coords.append(t)
        
        # Create a normalized temperature range for colormapping
        norm = Normalize(min_temp, max_temp)
        cmap = plt.cm.inferno
        
        # Map temperatures to colors
        colors = [cmap(norm(temp)) for temp in temps]
        
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
        
        # Set view angle for better visualization
        ax.view_init(30, 45)
        
        # Add grid for better spatial reference
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_voxel_visualization(self, downsample_factor=6, size_range=(5, 20), cmap='viridis',
                                  min_temp=-4, max_temp=12):
        """
        Create a 3D voxel visualization similar to the example image
        
        Args:
            downsample_factor: Factor to downsample the images
            size_range: Range for point sizes based on temperature
            cmap: Colormap to use
            min_temp: Minimum temperature in Celsius
            max_temp: Maximum temperature in Celsius
            
        Returns:
            Matplotlib figure
        """
        if self.thermal_sequence is None:
            raise ValueError("Thermal sequence not generated. Call generate_sequence() first.")
            
        # Downsample
        height, width = self.thermal_sequence[0].shape
        ds_height, ds_width = height // downsample_factor, width // downsample_factor
        
        # Create figure
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create 3D array for voxel data
        num_frames = len(self.thermal_sequence)
        voxel_data = np.zeros((ds_height, ds_width, num_frames))
        
        # Fill the voxel data
        for t, img in enumerate(self.thermal_sequence):
            ds_img = cv2.resize(img, (ds_width, ds_height), interpolation=cv2.INTER_AREA)
            voxel_data[:,:,t] = ds_img
            
        # Normalize data for visualization
        vmin = np.min(voxel_data)
        vmax = np.max(voxel_data)
        
        # Create mapping function from pixel values to temperature
        def map_to_temperature(pixel_value):
            # Map from [vmin, vmax] to [min_temp, max_temp]
            return min_temp + (max_temp - min_temp) * (pixel_value - vmin) / (vmax - vmin)
        
        # Create normalized temperature range
        norm = Normalize(min_temp, max_temp)
        color_map = plt.cm.get_cmap(cmap)
        
        # Collect points for plotting
        x_coords, y_coords, z_coords, colors, sizes = [], [], [], [], []
        
        # Plot points with variable size based on temperature
        for x in range(ds_width):
            for y in range(ds_height):
                for z in range(num_frames):
                    # Sample fewer points for clarity
                    if random.random() < 0.4:
                        pixel_value = voxel_data[y, x, z]
                        temp_value = map_to_temperature(pixel_value)
                        
                        x_coords.append(x)
                        y_coords.append(y)
                        z_coords.append(z)
                        
                        # Color based on temperature
                        colors.append(color_map(norm(temp_value)))
                        
                        # Size proportional to temperature (warmer = larger)
                        temp_ratio = (temp_value - min_temp) / (max_temp - min_temp)
                        size = size_range[0] + (size_range[1] - size_range[0]) * temp_ratio
                        sizes.append(size)
        
        # Plot all points at once
        ax.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes, alpha=0.7)
        
        # Add colorbar
        sm = cm.ScalarMappable(cmap=color_map, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.08, aspect=30)
        cbar.set_label('Temperature (°C)', fontsize=12)
        
        # Set labels
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Time Frame', fontsize=12)
        ax.set_title('3D Thermal Voxel Representation\nFor Building Analysis', 
                    fontsize=14, fontweight='bold')
        
        # Set view angle
        ax.view_init(30, 45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
        
    def create_panel_visualization(self, min_temp=-4, max_temp=12):
        """
        Create a panel visualization showing all frames for comparison
        
        Args:
            min_temp: Minimum temperature in Celsius
            max_temp: Maximum temperature in Celsius
            
        Returns:
            Matplotlib figure
        """
        if self.thermal_sequence is None:
            raise ValueError("Thermal sequence not generated. Call generate_sequence() first.")
            
        num_frames = len(self.thermal_sequence)
        rows = (num_frames + 3) // 4  # Ceiling division for number of rows
        cols = min(4, num_frames)     # Max 4 columns
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows), dpi=150)
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
        
        # Get pixel value range across all images
        vmin = min([np.min(img) for img in self.thermal_sequence])
        vmax = max([np.max(img) for img in self.thermal_sequence])
        
        # Create mapping function
        def map_to_temperature(img):
            # Map from [vmin, vmax] to [min_temp, max_temp]
            return min_temp + (max_temp - min_temp) * (img.astype(float) - vmin) / (vmax - vmin)
        
        # Create temperature-normalized colormap
        norm = Normalize(min_temp, max_temp)
        
        for i, img in enumerate(self.thermal_sequence):
            if i < len(axes):
                # Convert pixel values to temperatures
                temp_img = map_to_temperature(img)
                
                # Display with consistent temperature scale
                im = axes[i].imshow(temp_img, cmap='inferno', norm=norm)
                axes[i].set_title(f'Frame {i+1}', fontsize=10)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(num_frames, len(axes)):
            axes[i].axis('off')
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Temperature (°C)', fontsize=12)
        
        plt.suptitle('Multi-temporal Thermal Image Sequence', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.9, 0.96])
        
        return fig
    
    def save_visualizations(self, prefix='thermal', dpi=300, min_temp=-4, max_temp=12):
        """
        Save all visualizations to the output directory
        
        Args:
            prefix: Prefix for filenames
            dpi: Resolution for saved figures
            min_temp: Minimum temperature in Celsius
            max_temp: Maximum temperature in Celsius
            
        Returns:
            List of saved file paths
        """
        saved_files = []
        
        if self.thermal_sequence is None:
            self.generate_sequence()
        
        # Create and save pixel cloud
        fig_cloud = self.create_3d_pixel_cloud(min_temp=min_temp, max_temp=max_temp)
        cloud_path = os.path.join(self.output_dir, f"{prefix}_pixel_cloud.png")
        fig_cloud.savefig(cloud_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_cloud)
        saved_files.append(cloud_path)
        
        # Create and save voxel visualization
        fig_voxel = self.create_voxel_visualization(min_temp=min_temp, max_temp=max_temp)
        voxel_path = os.path.join(self.output_dir, f"{prefix}_voxel.png")
        fig_voxel.savefig(voxel_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_voxel)
        saved_files.append(voxel_path)
        
        # Create and save panel visualization
        fig_panel = self.create_panel_visualization(min_temp=min_temp, max_temp=max_temp)
        panel_path = os.path.join(self.output_dir, f"{prefix}_panel.png")
        fig_panel.savefig(panel_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_panel)
        saved_files.append(panel_path)
        
        print(f"Saved {len(saved_files)} visualizations to {self.output_dir}")
        return saved_files

def main():
    """Main function to run the visualization"""
    # Configuration
    image_path = '/home/fabio/University/research/SP2025_RABI/posterVisuals/Picture1.png'
    output_dir = '/home/fabio/University/research/SP2025_RABI/posterVisuals/'
    
    # Temperature range for building materials in winter conditions
    # Based on ambient temperature of about 28°F (-2°C) with wind
    min_temp = -4  # Coldest parts (metal, glass) can be below freezing
    max_temp = 12  # Warmest parts (brick, potentially heated areas)
    
    # Create visualizer
    try:
        visualizer = ThermalSequenceVisualizer(image_path, output_dir)
        
        # Generate temporal sequence with realistic variations
        visualizer.generate_sequence(
            num_frames=8,           # 8 frames as requested
            max_shift=3,            # Small spatial shifts
            temp_variation=1.0,     # Slight temperature variations
            add_noise=True,         # Add realistic thermal noise
            noise_level=3           # Moderate noise level
        )
        
        # Save all visualizations
        saved_files = visualizer.save_visualizations(
            prefix='thermal_building',
            min_temp=min_temp,
            max_temp=max_temp
        )
        
        print("Visualizations completed successfully!")
        print(f"Files saved: {saved_files}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()