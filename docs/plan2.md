# Verification Plan: Stoichiometry and Solver Accuracy

## Objective
Verify the stoichiometric consistency of the chemical network and the accuracy of the solver by checking elemental mass conservation during the simulation.

## Task 1: Stoichiometry Verification Script
**File:** `notebooks/usage/verify_stoichiometry.py`

This script will statically analyze the network definition to ensure all reactions balance mass and charge.

### Steps:
1.  **Load Network:**
    *   Load `data/uclchem_gas_phase_only.csv` using `carbox.parsers.parse_chemical_network`.
2.  **Define Elements:**
    *   Use a standard list of astrophysical elements: `["H", "He", "C", "N", "O", "S", "Si", "Fe", "Mg", "Na", "Cl", "P", "F"]` plus `"charge"`.
3.  **Construct Elemental Matrix ($E$):**
    *   Dimensions: [Number of Elements] $\times$ [Number of Species]
    *   Iterate through all species and determine the count of each element (e.g., "H2O" -> 2 H, 1 O).
    *   **Manual Verification Output:** Print the inferred composition of each species to the console so the user can spot check for parser errors (e.g., ensuring "CO" is Carbon+Oxygen, not Cobalt).
4.  **Construct Incidence Matrix ($N$):**
    *   Dimensions: [Number of Species] $\times$ [Number of Reactions]
    *   This is already available via `network.incidence`.
5.  **Calculate Stoichiometric Check Matrix ($S$):**
    *   Compute $S = E \times N$.
    *   Dimensions: [Number of Elements] $\times$ [Number of Reactions].
    *   Each entry $S_{i,j}$ represents the net change of element $i$ in reaction $j$.
6.  **Validation:**
    *   Check if $S$ is a zero matrix (all values $\approx 0$).
    *   Identify and print any reactions where $\sum |S_{i,j}| > 0$ (i.e., mass is created or destroyed).

## Task 2: Simulation Accuracy Verification
**File:** `notebooks/usage/uclchem_complex_parameters.py`

This script will be modified to check if the solver preserves elemental abundances over time.

### Steps:
1.  **Post-Processing:**
    *   Insert code after the `solve_network` call.
2.  **Calculate Total Elemental Abundances:**
    *   Use the same Elemental Matrix ($E$) from Task 1.
    *   For each time step $t$, calculate the vector of total elemental abundances $A(t)$:
        $$A(t) = E \times Y(t)$$
        Where $Y(t)$ is the vector of species abundances at time $t$.
3.  **Calculate Deviation:**
    *   Compute the percentage change relative to the initial conditions ($t=0$):
        $$\Delta\%(t) = \frac{A(t) - A(0)}{A(0)} \times 100$$
4.  **Output:**
    *   Print the maximum percentage change for each element across the entire simulation time.
    *   This confirms that the numerical integration did not "drift" and violate mass conservation.
