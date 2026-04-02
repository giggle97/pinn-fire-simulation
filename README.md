# PINN Fire Simulation Project

Physics-Informed Neural Networks (PINN) for fire simulation and visualization.

## Project Structure

```
├── basic_pinn/            # Basic PINN implementation
│   └── test_pinn.py      # Basic PINN test script
├── fire_simulation/       # Fire simulation code
│   ├── advanced/          # Advanced fire simulation (terrain and wind)
│   │   ├── README.md      # Advanced simulation documentation
│   │   ├── pinns_fire_terrain_wind.py          # CPU version
│   │   └── pinns_fire_terrain_wind_gpu.py      # GPU version
│   └── basic/             # Basic fire simulation
│       ├── rothermel_pinn_v1.py   # Rothermel fire model with PINN
│       ├── test.py                # Test script
│       └── wui_pinn_demo.py       # WUI fire demo
├── results/               # Results (excluded from version control)
├── visualization/         # Visualization code
│   └── three/             # Three.js visualization
│       ├── dy.html        # Dynamic visualization
│       ├── index.html     # Main visualization
│       └── wui_pinn_three.py  # Visualization script
├── .gitignore             # Git ignore file
├── LICENSE                # License file
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/giggle97/pinn-fire-simulation.git
   cd pinn-fire-simulation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic PINN

```bash
python basic_pinn/test_pinn.py
```

### Basic Fire Simulation

```bash
python fire_simulation/basic/rothermel_pinn_v1.py
python fire_simulation/basic/wui_pinn_demo.py
```

### Advanced Fire Simulation

```bash
# CPU version
python fire_simulation/advanced/pinns_fire_terrain_wind.py

# GPU version (if CUDA available)
python fire_simulation/advanced/pinns_fire_terrain_wind_gpu.py
```

### Visualization

Open the HTML files in a web browser:
- `visualization/three/index.html`
- `visualization/three/dy.html`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
