import pandas as pd
import folium
from folium import plugins
import branca.colormap as cm
from datetime import datetime

def fix_datetime_format(datetime_str):
    """Convert datetime from 'YYYY:MM:DD HH:MM:SS' to pandas-compatible format"""
    try:
        # Split into date and time
        date_part, time_part = datetime_str.split(' ')
        # Replace colons with dashes in date part
        date_part = date_part.replace(':', '-')
        return f"{date_part} {time_part}"
    except:
        return datetime_str

def fix_nj_coordinates(df):
    """Fix coordinates to be in New Jersey instead of Kyrgyzstan"""
    df = df.copy()
    
    # Convert coordinates to float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Fix longitudes for New Jersey
    df['longitude'] = df['longitude'].apply(lambda x: -abs(x) if 73 < abs(x) < 75 else x)
    
    # Fix datetime format
    df['datetime'] = df['datetime'].apply(fix_datetime_format)
    
    return df

def visualize_nj_drone_data(known_csv_path, unknown_csv_path, output_path='nj_drone_imagery_map.html'):
    """Create an interactive map visualization of drone imagery locations in New Jersey"""
    
    # Read and fix coordinates in both CSVs
    known_df = fix_nj_coordinates(pd.read_csv(known_csv_path))
    unknown_df = fix_nj_coordinates(pd.read_csv(unknown_csv_path))
    
    # Calculate center point for the map
    all_lats = pd.concat([known_df['latitude'], unknown_df['latitude']])
    all_lons = pd.concat([known_df['longitude'], unknown_df['longitude']])
    center_lat = all_lats.mean()
    center_lon = all_lons.mean()
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=16,
                  tiles='OpenStreetMap')
    
    # Create color schemes for different buildings
    known_buildings = known_df['building'].unique()
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue',
              'darkpurple', 'pink', 'lightred', 'lightblue', 'lightgreen', 'gray', 'black']
    building_colors = dict(zip(known_buildings, colors[:len(known_buildings)]))
    
    # Add known building points
    known_building_groups = {}
    
    for idx, row in known_df.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            building = row['building']
            if building not in known_building_groups:
                known_building_groups[building] = folium.FeatureGroup(name=f'Known - {building}')
            
            # Create popup text
            popup_text = f"""
            <b>Building:</b> {building}<br>
            <b>DateTime:</b> {row['datetime']}<br>
            <b>File:</b> {row['filename']}<br>
            <b>Type:</b> {row['image_type']}<br>
            <b>Coordinates:</b> {row['latitude']:.6f}, {row['longitude']:.6f}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color=building_colors.get(building, 'blue'),
                fill=True,
                fill_color=building_colors.get(building, 'blue'),
                fill_opacity=0.7,
                weight=2
            ).add_to(known_building_groups[building])
    
    # Add unknown points
    unknown_group = folium.FeatureGroup(name='Unknown Locations')
    
    for idx, row in unknown_df[unknown_df['building'] == 'Unknown'].iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            # Create popup text
            popup_text = f"""
            <b>DateTime:</b> {row['datetime']}<br>
            <b>File:</b> {row['filename']}<br>
            <b>Type:</b> {row['image_type']}<br>
            <b>Coordinates:</b> {row['latitude']:.6f}, {row['longitude']:.6f}
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                weight=2
            ).add_to(unknown_group)
    
    # Add all feature groups to map
    for group in known_building_groups.values():
        group.add_to(m)
    unknown_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add measure tool
    plugins.MeasureControl(position='topright').add_to(m)
    
    # Add fullscreen option
    plugins.Fullscreen(position='topright').add_to(m)
    
    # Add mini map
    minimap = plugins.MiniMap()
    m.add_child(minimap)
    
    # Save map
    m.save(output_path)
    print(f"Map saved to: {output_path}")
    
    # Save fixed coordinates back to CSVs
    known_df.to_csv(known_csv_path.replace('.csv', '_fixed.csv'), index=False)
    unknown_df.to_csv(unknown_csv_path.replace('.csv', '_fixed.csv'), index=False)
    print(f"\nFixed coordinates saved to:")
    print(f"- {known_csv_path.replace('.csv', '_fixed.csv')}")
    print(f"- {unknown_csv_path.replace('.csv', '_fixed.csv')}")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total known points: {len(known_df)}")
    print(f"Total unknown points: {len(unknown_df[unknown_df['building'] == 'Unknown'])}")
    print("\nKnown buildings:")
    for building in sorted(known_df['building'].unique()):
        count = len(known_df[known_df['building'] == building])
        print(f"- {building}: {count} points")

if __name__ == "__main__":
    # Replace these paths with your actual CSV file paths
    known_csv = "known_buildings_metadata.csv"
    unknown_csv = "unknown_buildings_metadata.csv"
    
    visualize_nj_drone_data(known_csv, unknown_csv)