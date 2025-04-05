import sys
import re
from collections import defaultdict
import numpy as np


def parse_cea_output(file_content):
    # Find the pressures from the P, BAR line - now matching any number of float values
    pressure_pattern = r'P, BAR\s+([\d\.\s]+)'
    pressure_match = re.search(pressure_pattern, file_content)
    if pressure_match:
        # Split on whitespace and convert to float, handling any number of values
        pressures = [float(p) for p in pressure_match.group(1).split()]
    
    # Find the MOLE FRACTIONS section
    mole_fraction_section = re.search(r'MOLE FRACTIONS\n\n(.*?)(?=\n\n|\Z)', 
                                    file_content, 
                                    re.DOTALL)
    
    if not mole_fraction_section:
        return None, None
    
    # Parse each species line
    species_data = defaultdict(list)
    for line in mole_fraction_section.group(1).split('\n'):
        # Skip empty lines
        if not line.strip():
            continue
            
        # Remove leading * if present and split the line
        line = line.replace('*', '').strip()
        parts = line.split()

        species = parts[0]
        fractions = [float(x) for x in parts[1:]]
        species_data[species] = fractions
    
    return pressures, dict(species_data)



def find_cases(file_content):
    # Split the file into individual cases
    # We can use "CASE = " as our delimiter
    cases = file_content.split('CASE = ')[1:]  # Skip empty first split
    return cases

def parse_multiple_cases(file_content):
    cases = find_cases(file_content)
    results = []
    
    for case in cases:
        # Extract PHI value
        phi_pattern = r'PHI,EQ\.RATIO=\s*(\d+\.\d+)'
        phi_match = re.search(phi_pattern, case)
        phi = float(phi_match.group(1)) if phi_match else None
        
        # Get pressures and species data using our existing function
        pressures, species_data = parse_cea_output(case)
        
        if phi is not None and pressures and species_data:
            results.append({
                'phi': phi,
                'pressures': pressures,
                'species_data': species_data
            })
    
    return results

def print_multi_case_results(results):
    for i, case in enumerate(results, 1):
        print(f"\nCase {i} (PHI = {case['phi']:.3f}):")
        print(f"Pressures (bar): {case['pressures']}")
        print("\nMole Fractions:")
        for species, fractions in case['species_data'].items():
            print(f"{species:5} {' '.join(f'{x:.5f}' for x in fractions)}")
        print("-" * 50)


def combine_results(results):
    # Check that all results contain the same pressure data
    phi = [result["phi"] for result in results]
    pressures = results[0]["pressures"]
    for result in results:
        if result["pressures"] != pressures:
            raise ValueError("All results must contain the same pressure data")
        
    # Find all unique species and ensure they are present in all results
    species = set(sum([list(result["species_data"].keys()) for result in results], []))
    for result in results:
        for sp in species:
            if sp not in result["species_data"]:
                result["species_data"][sp] = [0] * len(pressures)

    # Create a 2D array of species data
    composition = {}
    for sp in species:
        composition[sp] = np.array([result["species_data"][sp] for result in results])

    return phi, pressures, composition


if __name__ == "__main__":
    # The CEA filename should be the first argument
    filename = sys.argv[1]

    with open(filename, "r") as file:
        content = file.read()
        results = parse_multiple_cases(content)

    print_multi_case_results(results)

    # lookup table will be by (pressures, phi)
    phi, pressures, composition = combine_results(results)

    # Filter out any species that are never greater than 1%
    composition = {sp: comp for sp, comp in composition.items() if np.max(comp) > 0.01}

    print(f"\nFiltered composition: {composition.keys()}")

    filename_out = filename.replace(".out", ".npz")
    np.savez(filename_out, phi=phi, pressure=pressures, composition=composition)