"""
Process A13 highway data from 2024-09-09 to 2024-09-13
Concatenate multiple days of space-time speed diagrams into a single matrix
Format aligned with NGSIM velocity data structure
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator

def read_speed_data(filepath):
    """
    Read space-time speed data from a text file
    
    Args:
        filepath: path to the data file
        
    Returns:
        numpy array with shape (n_locations, n_timesteps)
    """
    data = np.loadtxt(filepath)
    return data

def concatenate_daily_data(data_dir, start_date, end_date, date_format="%Y-%m-%d"):
    """
    Concatenate multiple days of speed data along time dimension
    
    Args:
        data_dir: directory containing the data files
        start_date: start date string (e.g., "2024-09-09")
        end_date: end date string (e.g., "2024-09-13")
        date_format: date format string
        
    Returns:
        numpy array with concatenated data
        list of dates processed
    """
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, date_format)
    end = datetime.strptime(end_date, date_format)
    
    data_list = []
    dates_processed = []
    
    current = start
    while current <= end:
        date_str = current.strftime(date_format)
        filepath = os.path.join(data_dir, f"{date_str}.txt")
        
        if os.path.exists(filepath):
            print(f"Reading {filepath}...")
            data = read_speed_data(filepath)
            data_list.append(data)
            dates_processed.append(date_str)
        else:
            print(f"Warning: {filepath} not found, skipping...")
        
        current += timedelta(days=1)
    
    if not data_list:
        raise ValueError("No data files found in the specified date range")
    
    # Concatenate along time dimension (axis=1, columns)
    concatenated_data = np.hstack(data_list)
    
    print(f"\nProcessed {len(dates_processed)} days of data:")
    print(f"  Dates: {dates_processed}")
    print(f"  Final shape: {concatenated_data.shape} (locations × timesteps)")
    
    return concatenated_data, dates_processed

def save_velocity_data(data, output_filepath, format_type='space_delimited'):
    """
    Save velocity data in the same format as NGSIM data
    
    Args:
        data: numpy array to save
        output_filepath: path to save the file
        format_type: 'space_delimited' or 'csv'
    """
    if format_type == 'space_delimited':
        # Save with scientific notation, space-delimited (like NGSIM format)
        np.savetxt(output_filepath, data, fmt='%15.7e')
    else:
        # Save as CSV
        pd.DataFrame(data).to_csv(output_filepath, index=False, header=False)
    
    print(f"\nSaved to {output_filepath}")
    print(f"  Shape: {data.shape}")
    print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")


if __name__ == "__main__":
    # Configuration
    data_dir = "data"
    start_date = "2024-09-06"
    end_date = "2024-09-13"
    output_file = f"data/A13_Velocity_Data_{start_date[5:7]}{start_date[8:10]}-{end_date[5:7]}{end_date[8:10]}.txt"
    
    print("=" * 60)
    print("A13 Highway Space-Time Speed Data Processing")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}\n")
    
    # Read and concatenate data
    concatenated_data, dates = concatenate_daily_data(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date
    )
    
    # Save in NGSIM-compatible format
    save_velocity_data(
        data=concatenated_data,
        output_filepath=output_file,
        format_type='space_delimited'
    )
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    # Display summary statistics
    print("\nData Statistics:")
    print(f"  Number of spatial locations: {concatenated_data.shape[0]}")
    print(f"  Number of time steps: {concatenated_data.shape[1]}")
    print(f"  Time steps per day: {concatenated_data.shape[1] // len(dates)}")
    
    # Only compute stats on valid data (> 0)
    valid_data = concatenated_data[concatenated_data > 0]
    if len(valid_data) > 0:
        print(f"  Mean velocity: {np.mean(valid_data):.2f} km/h")
        print(f"  Min velocity (excluding -1): {np.min(valid_data):.2f} km/h")
        print(f"  Max velocity: {np.max(valid_data):.2f} km/h")
    print(f"  Number of missing values (-1): {np.sum(concatenated_data == -1)}")
    
    # ========== VISUALIZATION ==========
    print("\n" + "=" * 60)
    print("Generating Space-Time Diagram...")
    print("=" * 60)
    
    # Create spatial and temporal coordinates
    n_locations = concatenated_data.shape[0]
    n_timesteps = concatenated_data.shape[1]
    
    x = np.arange(n_locations)  # Spatial index
    t = np.arange(n_timesteps)  # Temporal index (time steps)
    X, T = np.meshgrid(x, t)
    
    # Prepare data for visualization (transpose to get time on y-axis, space on x-axis)
    velocity_data = concatenated_data.T  # Shape: (n_timesteps, n_locations)
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.88, top=0.95, bottom=0.1)
    
    ax = fig.add_subplot(gs[0])
    
    # Create heatmap
    im = ax.imshow(velocity_data, 
                   cmap='rainbow',
                   aspect='auto',
                   origin='lower',
                   interpolation='nearest',
                   extent=[x.min(), x.max(), t.min(), t.max()])
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Velocity (km/h)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Labels and title
    ax.set_xlabel('Spatial Location Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Time Step Index', fontsize=13, fontweight='bold')
    ax.set_title(f'A13 Highway Space-Time Speed Diagram\n({start_date} to {end_date})',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(False)
    ax.tick_params(labelsize=11)
    
    # Add text annotation with data info
    textstr = f'Locations: {n_locations} | Time steps: {n_timesteps} | Avg: {np.mean(valid_data):.1f} km/h'
    ax.text(0.5, -0.12, textstr, transform=ax.transAxes, 
            fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    plt.savefig('figures/a13_spacetime_diagram.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figures/a13_spacetime_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Space-time diagram saved to figures/a13_spacetime_diagram.pdf/png")
    
    # Display figure
    plt.show()
    
    print("=" * 60)

