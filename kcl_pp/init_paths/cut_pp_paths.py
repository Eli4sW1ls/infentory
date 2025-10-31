"""
Script to modify existing paths in a load/ folder based on interface sets.

This script takes a load/ folder containing paths with specific IDs and modifies
each path to only exist between a set of three interfaces (A, I, B), like in REPPTIS.

EW, Oct 2025
"""

import glob
import os
import sys
import shutil

import numpy as np
import MDAnalysis as mda
import tomli


def get_interfaces_from_toml(toml_path):
    """Load interfaces from infretis.toml file."""
    with open(toml_path, "rb") as toml_file:
        tdict = tomli.load(toml_file)
    return tdict["simulation"]["interfaces"]


def create_aib_sets(interfaces):
    """
    Create AIB (A, I, B) interface sets from the interface list.
    
    For each interface I (except first and last):
    - A is the previous interface (or first interface if I is the second)
    - I is the current interface
    - B is the next interface
    
    Returns: List of AIB tuples [(A0, I0, B0), (A1, I1, B1), ...]
    """
    aib_sets = []
    
    # Create AIB sets for all interfaces except the last
    for idx in range(len(interfaces) - 1):
        if idx == 0:
            A = interfaces[0]
        else:
            A = interfaces[idx - 1]
        I = interfaces[idx]
        B = interfaces[idx + 1]
        aib_sets.append((A, I, B))
    
    return aib_sets


def validate_path_ids(load_dir, path_ids):
    """
    Validate that the given path IDs exist in the load directory.
    
    Args:
        load_dir: Path to load/ directory
        path_ids: List of integer path IDs to validate
    
    Returns:
        List of validated path IDs
    
    Raises:
        FileNotFoundError: If any path ID doesn't exist
    """
    missing_paths = []
    
    for path_id in path_ids:
        path_dir = os.path.join(load_dir, str(path_id))
        if not os.path.isdir(path_dir):
            missing_paths.append(path_id)
    
    if missing_paths:
        raise FileNotFoundError(
            f"The following path IDs were not found in {load_dir}: {missing_paths}"
        )
    
    return path_ids


def find_aib_region(order_values, aib):
    """
    Find the indices that define a continuous region between interfaces A and B
    that crosses interface I, starting near A and ending near B (or reverse).
    
    The function finds the first continuous segment where:
    - All points are between A and B (inclusive)
    - The segment crosses interface I (has points on both sides of I)
    - The segment is continuous (no gaps)
    - The segment starts at or beyond A and ends at or beyond B (forward path)
      OR starts at or beyond B and ends at or beyond A (reverse path)
    
    Returns: (start_idx, end_idx) inclusive indices, or (None, None) if no valid segment
    """
    A, I, B = aib
    
    # Find all points between A and B
    in_region = np.logical_and(order_values >= A, order_values <= B)
    
    if not np.any(in_region):
        return None, None
    
    # Find continuous segments in the AIB region
    # A segment is continuous if consecutive indices differ by 1
    in_region_indices = np.where(in_region)[0]
    
    # Split into continuous segments
    segments = []
    current_segment = [in_region_indices[0]]
    
    for i in range(1, len(in_region_indices)):
        if in_region_indices[i] == in_region_indices[i-1] + 1:
            # Continuous
            current_segment.append(in_region_indices[i])
        else:
            # Gap found, save current segment and start new one
            segments.append(current_segment)
            current_segment = [in_region_indices[i]]
    
    # Don't forget the last segment
    segments.append(current_segment)
    
    # Find the first segment that:
    # 1. Crosses interface I
    # 2. Starts near A and ends near B (or vice versa)
    best_segment = None
    
    for segment in segments:
        segment_values = order_values[segment]
        
        # Check if this segment crosses I
        crosses_I = np.any(segment_values < I) and np.any(segment_values > I)
        
        # Special case: if A == I, we don't need to cross I
        if aib[0] == aib[1]:
            crosses_I = True
        
        if not crosses_I:
            continue
        
        # Check if segment starts near A and ends near B (forward)
        # or starts near B and ends near A (backward)
        start_val = segment_values[0]
        end_val = segment_values[-1]
        
        # Forward path: starts closer to A, ends closer to B
        forward_path = (abs(start_val - A) < abs(start_val - B)) and (abs(end_val - B) < abs(end_val - A))
        
        # Reverse path: starts closer to B, ends closer to A
        reverse_path = (abs(start_val - B) < abs(start_val - A)) and (abs(end_val - A) < abs(end_val - B))
        
        if forward_path or reverse_path:
            best_segment = segment
            break
    
    if best_segment is None:
        return None, None
    
    # Get start and end indices
    start_idx = best_segment[0]
    end_idx = best_segment[-1]
    
    # Extend to include boundary crossings
    # Include one point before if it exists and is outside [A, B]
    if start_idx > 0 and (order_values[start_idx - 1] < A or order_values[start_idx - 1] > B):
        start_idx -= 1
    else:
        return None, None
    
    # Include one point after if it exists and is outside [A, B]
    if end_idx < len(order_values) - 1 and (order_values[end_idx + 1] < A or order_values[end_idx + 1] > B):
        end_idx += 1
    else:
        return None, None

    return start_idx, end_idx


def modify_path(load_dir, output_dir, path_id, ensemble_id, aib):
    """
    Cut a single path to only exist between the AIB interfaces and save to output directory.
    
    Args:
        load_dir: Base load directory (source)
        output_dir: Output directory (load_pp)
        path_id: Path ID (integer) from source
        ensemble_id: Ensemble ID (integer) for output folder
        aib: Tuple of (A, I, B) interfaces
    """
    path_dir = os.path.join(load_dir, str(path_id))
    output_path_dir = os.path.join(output_dir, str(ensemble_id))
    
    print(f"\nProcessing path {path_id} -> ensemble {ensemble_id} with interfaces A={aib[0]:.4f}, I={aib[1]:.4f}, B={aib[2]:.4f}")
    
    # Check that source directory exists
    if not os.path.isdir(path_dir):
        raise FileNotFoundError(f"Path directory not found: {path_dir}")
    
    # Create output directory structure
    os.makedirs(output_path_dir, exist_ok=True)
    output_accepted_dir = os.path.join(output_path_dir, "accepted")
    os.makedirs(output_accepted_dir, exist_ok=True)
    
    # Load order.txt
    order_file = os.path.join(path_dir, "order.txt")
    if not os.path.exists(order_file):
        raise FileNotFoundError(f"order.txt not found in {path_dir}")
    
    # Read order.txt header separately
    order_header_lines = []
    with open(order_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                order_header_lines.append(line)
            else:
                break
    
    # Read order parameter values (skip header lines starting with #)
    order_data = np.loadtxt(order_file)
    if order_data.ndim == 1:
        order_data = order_data.reshape(1, -1)
    
    time_values = order_data[:, 0]
    order_values = order_data[:, 1]
    
    # Find the region between A, I, B
    start_idx, end_idx = find_aib_region(order_values, aib)
    
    if start_idx is None or end_idx is None:
        raise ValueError(
            f"Path {path_id} does not have a valid continuous segment in the AIB region "
            f"that crosses interface I (A={aib[0]:.4f}, I={aib[1]:.4f}, B={aib[2]:.4f}). "
            f"Path must be continuous and have points on both sides of I."
        )
    
    print(f"  Cutting path from index {start_idx} to {end_idx} (length: {end_idx - start_idx + 1})")
    print(f"  Order parameter range: {order_values[start_idx]:.4f} to {order_values[end_idx]:.4f}")
    
    # Read traj.txt
    traj_file = os.path.join(path_dir, "traj.txt")
    if not os.path.exists(traj_file):
        raise FileNotFoundError(f"traj.txt not found in {path_dir}")
    
    # Read trajectory info (skip header)
    with open(traj_file, 'r') as f:
        lines = f.readlines()
    
    header_lines = [l for l in lines if l.startswith('#')]
    data_lines = [l for l in lines if not l.startswith('#')]
    
    traj_data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 4:
            traj_data.append(parts)
    
    # Cut the trajectory data
    cut_traj_data = traj_data[start_idx:end_idx + 1]
    
    # Get unique trajectory files referenced
    trr_files = set()
    for entry in cut_traj_data:
        trr_files.add(entry[1])
    
    # Check that TRR files exist in accepted folder
    accepted_dir = os.path.join(path_dir, "accepted")
    if not os.path.isdir(accepted_dir):
        raise FileNotFoundError(f"accepted/ folder not found in {path_dir}")
    
    existing_trr = glob.glob(os.path.join(accepted_dir, "*.trr"))
    if len(existing_trr) == 0:
        raise FileNotFoundError(f"No TRR files found in {accepted_dir}")
    
    print(f"  Found {len(existing_trr)} TRR files in accepted/")
    
    # Build mapping of file -> list of (new_idx, old_idx, vel)
    trr_frame_map = {}
    for new_idx, entry in enumerate(cut_traj_data):
        trr_file = entry[1]
        old_frame_idx = int(entry[2])
        vel = int(entry[3])
        
        if trr_file not in trr_frame_map:
            trr_frame_map[trr_file] = []
        trr_frame_map[trr_file].append((new_idx, old_frame_idx, vel))
    
    # Create a combined trajectory file in output directory with unique name
    # Format: path{path_id}_ens{ensemble_id}_cut.trr
    output_trr_name = f"path{path_id}_ens{ensemble_id}_cut.trr"
    output_trr = os.path.join(output_accepted_dir, output_trr_name)
    
    # Use MDAnalysis to read and write frames from source directory
    first_trr = os.path.join(accepted_dir, list(trr_files)[0])
    temp_universe = mda.Universe(first_trr)
    n_atoms = temp_universe.atoms.n_atoms
    
    # Collect all frames in order
    frames_to_write = [None] * len(cut_traj_data)
    
    for trr_file, frame_list in trr_frame_map.items():
        trr_path = os.path.join(accepted_dir, trr_file)
        
        if not os.path.exists(trr_path):
            print(f"  Warning: TRR file not found: {trr_file}, skipping")
            continue
        
        u = mda.Universe(trr_path)
        
        for new_idx, old_idx, vel in frame_list:
            u.trajectory[old_idx]
            frames_to_write[new_idx] = (u.atoms.positions.copy(), 
                                       u.atoms.velocities.copy() if hasattr(u.atoms, 'velocities') else None,
                                       vel)
    
    # Write combined trajectory
    with mda.Writer(output_trr, n_atoms) as W:
        for frame_data in frames_to_write:
            if frame_data is None:
                continue
            positions, velocities, vel = frame_data
            temp_universe.atoms.positions = positions
            if velocities is not None:
                temp_universe.atoms.velocities = velocities
            W.write(temp_universe.atoms)
    
    print(f"  Created combined trajectory: accepted/{output_trr_name} with {len(frames_to_write)} frames")
    
    # Create order.txt in output directory
    new_order_data = order_data[start_idx:end_idx + 1].copy()
    # Renumber time column
    new_order_data[:, 0] = np.arange(len(new_order_data))
    
    output_order_file = os.path.join(output_path_dir, "order.txt")
    with open(output_order_file, 'w') as f:
        # Write original order.txt header if it exists
        if order_header_lines:
            for header in order_header_lines:
                f.write(header)
        else:
            f.write(f"#{'Time':>10} {'Orderp':>15}\n")
        
        for i in range(len(new_order_data)):
            f.write(f"{int(new_order_data[i, 0]):>11} {new_order_data[i, 1]:>15.6f}\n")
    
    print(f"  Created order.txt")
    
    # Create traj.txt in output directory
    output_traj_file = os.path.join(output_path_dir, "traj.txt")
    with open(output_traj_file, 'w') as f:
        # Write original traj.txt header if it exists
        if header_lines:
            for header in header_lines:
                f.write(header)
        else:
            f.write(f"#{'Step':>10} {'Filename':>20} {'index':>10} {'vel':>5}\n")
        
        for new_idx in range(len(cut_traj_data)):
            f.write(f"{new_idx:>11} {output_trr_name:>20} {new_idx:>10} {cut_traj_data[new_idx][3]:>5}\n")
    
    print(f"  Created traj.txt")
    
    # Check if energy.txt exists and create it in output directory
    energy_file = os.path.join(path_dir, "energy.txt")
    if os.path.exists(energy_file):
        energy_data = np.loadtxt(energy_file)
        if energy_data.ndim == 1:
            energy_data = energy_data.reshape(1, -1)
        
        new_energy_data = energy_data[start_idx:end_idx + 1].copy()
        new_energy_data[:, 0] = np.arange(len(new_energy_data))
        
        output_energy_file = os.path.join(output_path_dir, "energy.txt")
        np.savetxt(output_energy_file, new_energy_data, 
                  fmt=['%11d'] + ['%15.6f'] * (new_energy_data.shape[1] - 1))
        print(f"  Created energy.txt")


def main(load_dir, toml_path, path_ids, output_dir="load_pp"):
    """
    Cut paths from load directory and save to output directory.
    
    Args:
        load_dir: Path to load/ directory containing source path folders
        toml_path: Path to infretis.toml file with interface definitions
        path_ids: List of integer path IDs to process
        output_dir: Output directory name (default: "load_pp")
    """
    print(f"Loading interfaces from {toml_path}")
    interfaces = get_interfaces_from_toml(toml_path)
    print(f"Interfaces: {interfaces}")
    
    print(f"\nCreating AIB sets...")
    aib_sets = create_aib_sets(interfaces)
    for i, aib in enumerate(aib_sets):
        print(f"  Ensemble {i}: A={aib[0]:.4f}, I={aib[1]:.4f}, B={aib[2]:.4f}")
    
    print(f"\nValidating path IDs: {path_ids}")
    path_ids = validate_path_ids(load_dir, path_ids)
    print(f"All path IDs exist in {load_dir}")
    
    # Check that number of paths matches number of AIB sets
    if len(path_ids) != len(aib_sets):
        raise ValueError(
            f"Number of paths ({len(path_ids)}) does not match "
            f"number of AIB sets ({len(aib_sets)}). "
            f"Expected {len(aib_sets)} paths."
        )
    
    # Create output directory
    print(f"\nCreating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting path cutting and copying to {output_dir}...")
    print(f"{'='*60}")
    
    # Process each path with corresponding ensemble ID
    for ensemble_id, (path_id, aib) in enumerate(zip(path_ids, aib_sets)):
        try:
            modify_path(load_dir, output_dir, path_id, ensemble_id, aib)
        except Exception as e:
            print(f"\n{'!'*60}")
            print(f"ERROR processing path {path_id}: {e}")
            print(f"{'!'*60}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"All paths cut and saved to {output_dir} successfully!")
    print(f"Original paths in {load_dir} remain unchanged.")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
================================================================================
                    PATH CUTTING SCRIPT FOR INFRETIS
================================================================================

DESCRIPTION:
    This script cuts and processes trajectory paths from a load/ directory based
    on interface definitions from infretis.toml. Each path is trimmed to only
    contain the continuous segment between AIB (A-I-B) interfaces that crosses
    the middle interface I.
    
    The script creates a new output directory (default: load_pp/) with processed
    paths, leaving the original load/ directory completely unchanged.

USAGE:
    python modify_paths.py <load_dir> <toml_path> <path_id1> <path_id2> ... [output_dir]

ARGUMENTS:
    load_dir    Path to the source load/ directory containing path folders
                Each path folder should contain:
                  - order.txt      : Order parameter values
                  - traj.txt       : Trajectory file references
                  - accepted/      : Directory with .trr trajectory files
                  - energy.txt     : (Optional) Energy values
    
    toml_path   Path to infretis.toml file containing interface definitions
                Must have [simulation.interfaces] section
    
    path_ids    Space-separated list of path IDs (integers) to process
                Number of paths must match number of AIB sets (N-1 interfaces)
                Order matters: first path_id -> ensemble 0, etc.
    
    output_dir  (Optional) Output directory name (default: 'load_pp')
                If specified, must be last argument and non-integer

OUTPUT STRUCTURE:
    output_dir/
        0/                          # Ensemble 0
            order.txt               # Cut and renumbered order parameters
            traj.txt                # References to cut trajectory
            energy.txt              # Cut and renumbered energies (if exists)
            accepted/
                path{X}_ens0_cut.trr   # Combined cut trajectory
        1/                          # Ensemble 1
            ...
        N/                          # Ensemble N

AIB SETS (Interface Logic):
    For N interfaces, creates N-1 AIB sets:
    - Ensemble 0: A=I[0], I=I[0], B=I[1]
    - Ensemble 1: A=I[0], I=I[1], B=I[2]
    - Ensemble 2: A=I[1], I=I[2], B=I[3]
    - ...
    
    Each path is cut to the longest continuous segment where:
    1. All points are between A and B (inclusive)
    2. The segment crosses interface I (has points on both sides)
    3. No gaps in the trajectory

EXAMPLES:
    # Process 9 paths for 10 interfaces (creates load_pp/)
    python modify_paths.py load ../infretis.toml 0 1 2 3 4 5 6 7 8
    
    # Process 4 paths with custom output directory
    python modify_paths.py load ../infretis.toml 10 11 12 13 custom_output
    
    # Process paths from current directory
    python modify_paths.py ./load ./infretis.toml 5 6 7 8
    
    # Get this help message
    python modify_paths.py -h

ERROR HANDLING:
    The script will exit with an error if:
    - Any path ID directory doesn't exist
    - Number of path IDs doesn't match number of AIB sets
    - Required files (order.txt, traj.txt) are missing
    - No .trr files found in accepted/ folder
    - Path doesn't have a continuous segment crossing interface I
    - Interface definitions missing in toml file

REQUIREMENTS:
    - Python 3.7+
    - numpy
    - MDAnalysis
    - tomli

NOTES:
    - Original paths in load/ are NEVER modified
    - Output paths are renumbered starting from 0
    - TRR files are combined into single file per ensemble
    - Unique TRR naming: path{source_id}_ens{ensemble_id}_cut.trr
    - Headers from original files are preserved
    
================================================================================
        """)
        sys.exit(0)
    
    # Parse command line arguments
    # Usage: python modify_paths.py <load_dir> <toml_path> <path_id1> <path_id2> ... [output_dir]
    # Example: python modify_paths.py load ../infretis.toml 0 1 2 3 4 5 6 7 8 9
    # Example: python modify_paths.py load ../infretis.toml 0 1 2 3 4 5 6 7 8 9 load_custom
    
    if len(sys.argv) < 4:
        print("Error: Insufficient arguments")
        print("\nUsage: python modify_paths.py <load_dir> <toml_path> <path_id1> <path_id2> ... [output_dir]")
        print("\nExamples:")
        print("  python modify_paths.py load ../infretis.toml 0 1 2 3 4 5 6 7 8 9")
        print("  python modify_paths.py load ../infretis.toml 0 1 2 3 load_custom")
        print("\nArguments:")
        print("  load_dir    : Path to the source load/ directory containing path folders")
        print("  toml_path   : Path to infretis.toml file with interface definitions")
        print("  path_ids    : Space-separated list of path IDs (integers) to process")
        print("  output_dir  : (Optional) Output directory name (default: 'load_pp')")
        print("\nNote: Output directory will be created if it doesn't exist.")
        print("      Original paths will NOT be modified.")
        sys.exit(1)
    
    load_dir = sys.argv[1]
    toml_path = sys.argv[2]
    
    # Parse path IDs from remaining arguments (all except possibly the last one if it's a string)
    # Check if last argument might be output_dir (non-integer)
    output_dir = "load_pp"  # default
    args_for_path_ids = sys.argv[3:]
    
    # Try to parse the last argument - if it fails, it might be the output_dir
    if len(args_for_path_ids) > 0:
        try:
            int(args_for_path_ids[-1])
            # Last arg is an integer, all are path IDs
            path_ids = [int(arg) for arg in args_for_path_ids]
        except ValueError:
            # Last arg is not an integer, treat it as output_dir
            output_dir = args_for_path_ids[-1]
            try:
                path_ids = [int(arg) for arg in args_for_path_ids[:-1]]
            except ValueError as e:
                print(f"Error: All path IDs must be integers. Invalid argument: {e}")
                sys.exit(1)
    else:
        path_ids = []
    
    if not path_ids:
        print("Error: At least one path ID must be provided")
        sys.exit(1)
    
    if not os.path.isdir(load_dir):
        print(f"Error: Load directory not found: {load_dir}")
        sys.exit(1)
    
    if not os.path.exists(toml_path):
        print(f"Error: TOML file not found: {toml_path}")
        sys.exit(1)
    
    main(load_dir, toml_path, path_ids, output_dir)
