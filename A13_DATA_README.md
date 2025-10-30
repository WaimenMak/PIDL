# A13 Highway Data Processing

This document describes the processing of A13 highway space-time speed data for use with Physics-Informed Neural Networks (PINNs).

## Data Structure

### Input Data Files
- **Location**: `data/2024-09-*.txt`
- **Format**: Space-delimited numeric matrix
- **Structure**:
  - Each **row** represents a spatial location (30 locations total)
  - Each **column** represents a time step (96 time steps per day, ~15 minute intervals)
  - Values represent vehicle speed in **km/h**
  - Missing or invalid data marked as `-1.0`

### Date Range
- **Start**: 2024-09-09
- **End**: 2024-09-13
- **Total**: 5 days of data

## Processing Pipeline

### 1. Data Concatenation (`process_a13_data.py`)

This script concatenates multiple days of data into a single matrix:

```bash
python process_a13_data.py
```

**Output**: `data/A13_Velocity_Data_0909-0913.txt`
- Shape: (30 locations, 480 time steps)
- Format: Space-delimited scientific notation (compatible with NGSIM format)

**Statistics**:
- Number of spatial locations: 30
- Number of time steps: 480 (96 per day × 5 days)
- Mean velocity: ~92 km/h
- Velocity range: 13-123 km/h (excluding missing values)

### 2. PINN Training (`ojits03_a13_pytorch.py`)

This script applies Physics-Informed Deep Learning to the A13 data:

```bash
python ojits03_a13_pytorch.py
```

**Features**:
- Uses PyTorch implementation
- Trains both PINN and regular NN models for comparison
- Handles missing data by:
  - Excluding `-1` values from training
  - Replacing with mean velocity for visualization
- Generates comparison plots

**Output**:
- Trained models for traffic speed estimation
- Visualization plots: `figures/a13_pidl_dl_pytorch_*.pdf/eps`
- Performance metrics (L2 error)

## Data Format Alignment

The processed A13 data follows the same format as NGSIM data:

| Aspect | NGSIM Data | A13 Data |
|--------|-----------|----------|
| Format | Space-delimited matrix | Space-delimited matrix |
| Units | m/s | km/h |
| Missing values | - | -1.0 |
| Structure | (locations × time) | (30 × 480) |
| Loading | `pd.read_table(..., delim_whitespace=True)` | Same |

## Usage Example

```python
import pandas as pd
import numpy as np

# Load the concatenated A13 data
vel = pd.read_table('data/A13_Velocity_Data_0909-0913.txt', 
                    delim_whitespace=True, 
                    header=None)

print(f"Data shape: {vel.shape}")  # (30, 480)
print(f"Locations: {vel.shape[0]}")  # 30
print(f"Time steps: {vel.shape[1]}")  # 480

# Handle missing values
data = vel.values
valid_mask = data > 0
valid_data = data[valid_mask]
print(f"Valid data range: {valid_data.min():.2f} - {valid_data.max():.2f} km/h")
```

## Key Differences from NGSIM

1. **Units**: A13 uses km/h, NGSIM uses m/s (convert with factor ~3.6)
2. **Missing Data**: A13 explicitly marks with -1, NGSIM has no gaps
3. **Scale**: A13 covers 5 days, NGSIM covers shorter duration
4. **Resolution**: A13 has 96 time steps/day (~15 min intervals)

## Files Overview

```
data/
├── 2024-09-09.txt          # Day 1 raw data
├── 2024-09-10.txt          # Day 2 raw data
├── 2024-09-11.txt          # Day 3 raw data
├── 2024-09-12.txt          # Day 4 raw data
├── 2024-09-13.txt          # Day 5 raw data
└── A13_Velocity_Data_0909-0913.txt  # Concatenated output

process_a13_data.py         # Data processing script
ojits03_a13_pytorch.py      # PINN training script for A13
ojits02_ngsim_pytorch.py    # PINN training script for NGSIM (reference)
```

## Customization

To process different date ranges, edit `process_a13_data.py`:

```python
# Change these variables:
start_date = "2024-09-09"  # Your start date
end_date = "2024-09-13"    # Your end date
output_file = "data/A13_Velocity_Data_custom.txt"
```

## Notes

- Data quality: First row contains all `-1` values (sensor placement or boundary condition)
- Spatial resolution: 30 detection points along highway section
- Temporal resolution: 96 measurements per day (15-minute intervals)
- The PINN model uses traffic flow PDEs with parameters that may need tuning for A13 characteristics

