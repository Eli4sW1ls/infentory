"""
Simple test to load paths from load_pp using infretis path module.
"""

import sys
from pathlib import Path

# Add infretis to path
infretis_path = Path("/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS/infretis")
if infretis_path.exists():
    sys.path.insert(0, str(infretis_path))
else:
    print(f"Error: Infretis not found at {infretis_path}")
    sys.exit(1)

# Import infretis modules
try:
    from infretis.classes.path import load_paths_from_disk
    print("✓ Successfully imported load_paths_from_disk from infretis")
except ImportError as e:
    print(f"❌ Failed to import from infretis: {e}")
    print("\nTrying alternate import...")
    try:
        # Try importing the whole module and checking what's available
        from infretis.classes import path as path_module
        print(f"Available functions in path module: {dir(path_module)}")
        # Look for any load function
        load_funcs = [f for f in dir(path_module) if 'load' in f.lower()]
        print(f"Functions with 'load' in name: {load_funcs}")
    except Exception as e2:
        print(f"❌ Could not import path module: {e2}")
    sys.exit(1)

# Test loading paths
print("\n" + "="*70)
print("Testing path loading from load_pp/")
print("="*70 + "\n")

load_dir = Path("load_pp")
if not load_dir.exists():
    print(f"❌ ERROR: {load_dir} directory not found!")
    sys.exit(1)

# Get ensemble directories
ensemble_dirs = sorted([int(d.name) for d in load_dir.iterdir() if d.is_dir() and d.name.isdigit()])
print(f"Found ensembles: {ensemble_dirs}")

# Load toml configuration
import tomli
toml_path = Path("test_infretis.toml")
if not toml_path.exists():
    print(f"❌ ERROR: {toml_path} not found!")
    sys.exit(1)

with open(toml_path, "rb") as f:
    config = tomli.load(f)

print(f"Loaded configuration from {toml_path}")
print(f"Load directory in config: {config['simulation']['load_dir']}")

# Try to load paths
try:
    print(f"\nAttempting to load paths using load_paths_from_disk...")
    paths = load_paths_from_disk(config)
    
    print(f"\n✓ SUCCESS! Loaded {len(paths)} paths")
    
    # Print some info about each path
    for i, path in enumerate(paths):
        print(f"\nPath {i}:")
        print(f"  Type: {type(path)}")
        if hasattr(path, 'phasepoints'):
            print(f"  Phase points: {len(path.phasepoints)}")
        if hasattr(path, 'get_length'):
            print(f"  Length: {path.get_length()}")
        if hasattr(path, 'phasepoints'):
            order_vals = [php.order for php in path.phasepoints[:5]]
            print(f"  First order values: {order_vals}")
            print(f"  Order range: [{min([php.order for php in path.phasepoints])}, {max([php.order for php in path.phasepoints])}]")
        if hasattr(path, 'generated'):
            print(f"  Status: {path.generated}")
        if hasattr(path, 'path_number'):
            print(f"  Path Number: {path.path_number}")


        
    print("\n" + "="*70)
    print("✓ Path loading test PASSED!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ ERROR loading paths: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
