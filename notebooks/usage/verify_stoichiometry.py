"""Verify stoichiometric consistency of the chemical network.

This script checks that all reactions in the network conserve mass and charge
by examining the stoichiometric matrix.
"""

import numpy as np

from carbox.parsers import NetworkNames, parse_chemical_network

# Define standard astrophysical elements to track
ELEMENTS = [
    "H",
    "He",
    "C",
    "N",
    "O",
    "S",
    "Si",
    "Fe",
    "Mg",
    "Na",
    "Cl",
    "P",
    "F",
    "charge",
]

print("=" * 80)
print("Stoichiometry Verification Script")
print("=" * 80)
print()

# Load network
print("Loading chemical network...")
network_file = "data/uclchem_gas_phase_only.csv"
network = parse_chemical_network(network_file, format_type=NetworkNames.uclchem)
print(f"✓ Loaded {len(network.species)} species, {len(network.reactions)} reactions")
print()

# Create elemental composition matrix manually with better parsing
print("=" * 80)
print("SPECIES ELEMENTAL COMPOSITION (Manual Verification)")
print("=" * 80)
print("Format: Species -> Elements")
print()

elemental_matrix = np.zeros((len(ELEMENTS), len(network.species)))

for i, species in enumerate(network.species):
    name = species.name
    composition = {}

    # Parse species name to extract elemental composition
    # Handle special cases
    if name == "E-" or name == "ELECTR":
        composition["charge"] = -1
    else:
        # Count charge from + or -
        if "+" in name:
            charge_count = name.count("+")
            composition["charge"] = charge_count
            name = name.replace("+", "")
        elif "-" in name and name != "E-":
            charge_count = name.count("-")
            composition["charge"] = -charge_count
            name = name.replace("-", "")

        # Parse atoms (simple approach: look for element symbols followed by optional digits)
        j = 0
        while j < len(name):
            # Try two-letter elements first
            if j + 1 < len(name):
                two_letter = name[j : j + 2]
                if two_letter in ELEMENTS:
                    # Found element, check for digit
                    count = 1
                    if j + 2 < len(name) and name[j + 2].isdigit():
                        count = int(name[j + 2])
                        j += 3
                    else:
                        j += 2
                    composition[two_letter] = composition.get(two_letter, 0) + count
                    continue

            # Try single-letter elements
            one_letter = name[j]
            if one_letter in ELEMENTS:
                count = 1
                if j + 1 < len(name) and name[j + 1].isdigit():
                    count = int(name[j + 1])
                    j += 2
                else:
                    j += 1
                composition[one_letter] = composition.get(one_letter, 0) + count
            else:
                j += 1

    # Fill in the elemental matrix
    for elem_idx, element in enumerate(ELEMENTS):
        if element in composition:
            elemental_matrix[elem_idx, i] = composition[element]

    # Print composition for manual verification
    if composition:
        comp_str = ", ".join(
            [f"{elem}:{int(count)}" for elem, count in sorted(composition.items())]
        )
        print(f"{species.name:15s} -> {comp_str}")

print()
print(f"✓ Parsed elemental composition for {len(network.species)} species")
print()

# Get incidence matrix (Species x Reactions)
print("=" * 80)
print("STOICHIOMETRIC MATRIX CHECK")
print("=" * 80)
print()

incidence = network.incidence
# Convert sparse matrix to dense if needed
if hasattr(incidence, "todense"):
    incidence_dense = incidence.todense()
    incidence = np.array(incidence_dense)
else:
    incidence = np.array(incidence)

print(f"Incidence matrix shape: {incidence.shape} (Species x Reactions)")
print(f"Elemental matrix shape: {elemental_matrix.shape} (Elements x Species)")
print()

# Calculate stoichiometric matrix: Elements x Reactions
stoich_matrix = elemental_matrix @ incidence
print(f"Stoichiometric matrix shape: {stoich_matrix.shape} (Elements x Reactions)")
print()

# Check for violations (non-zero entries indicate mass/charge not conserved)
violations = []
for reaction_idx in range(stoich_matrix.shape[1]):
    reaction_balance = stoich_matrix[:, reaction_idx]
    if not np.allclose(reaction_balance, 0, atol=1e-10):
        # Find which elements are unbalanced
        unbalanced_elements = []
        for elem_idx, delta in enumerate(reaction_balance):
            if abs(delta) > 1e-10:
                unbalanced_elements.append(f"{ELEMENTS[elem_idx]}:{delta:+.1f}")

        reaction = network.reactions[reaction_idx]
        reactants_str = " + ".join(reaction.reactants)
        products_str = " + ".join(reaction.products)

        violations.append(
            {
                "index": reaction_idx,
                "reaction": f"{reactants_str} -> {products_str}",
                "unbalanced": ", ".join(unbalanced_elements),
            }
        )

# Print results
if violations:
    print(f"⚠ FOUND {len(violations)} STOICHIOMETRY VIOLATIONS:")
    print()
    for v in violations[:10]:  # Show first 10
        print(f"Reaction {v['index']:4d}: {v['reaction']}")
        print(f"                Unbalanced: {v['unbalanced']}")
        print()

    if len(violations) > 10:
        print(f"... and {len(violations) - 10} more violations")
else:
    print("✓ ALL REACTIONS CONSERVE MASS AND CHARGE!")

print()
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
