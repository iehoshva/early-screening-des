import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import itertools
import time

time_start = time.time()
# =============================================================================
# 1. CANONICAL CONSTANTS & HOFTZER-VAN KREVELEN PARAMETERS
# =============================================================================
R_GAS    = 8.314    # J/(mol·K) — universal gas constant
T        = 298.15   # K (25 °C) — reference temperature
V_RING_CORRECTION = 16.0  # Van Krevelen ring-closure correction (cm³/mol), SUBTRACTED per ring

# Eyring Hole Theory constants for viscosity
N_A       = 6.02214e23   # Avogadro's number (mol⁻¹)
H_PLANCK  = 6.626e-34    # Planck's constant (J·s)

# K_VISC: ratio dG*/dE_vap
# Recommended: ~0.30 for non-polar DES (e.g. menthol/fatty acid)
#              ~0.45–0.55 for polar/H-bonded NADES
# A single value is used here for comparability; override per system if needed.
K_VISC = 0.45

# ─── HVK Group-Contribution Database ────────────────────────────────────────
# Format: { group_name: [SMARTS, F_di, F_pi², E_hi, V_mi] }
# SMARTS patterns are used so RDKit treats each entry as a substructure query.
hvk_database = {
    '-CH3':             ['[CX4H3]',               420.0,   0.0,      0.0,  33.5],
    '-CH2-':            ['[CX4H2]',               270.0,   0.0,      0.0,  16.1],
    '>CH-':             ['[CX4H1]',                80.0,   0.0,      0.0,  -1.0],
    '>C<':              ['[CX4H0]',                70.0,   0.0,      0.0, -19.2],
    '=CH-':             ['[CX3H1]',               200.0,   0.0,      0.0,  13.5],
    '=C<':              ['[CX3H0]',                70.0,   0.0,      0.0,  -5.5],
    '-COOH':            ['[CX3](=O)[OX2H1]',      530.0, 420.0,  10000.0,  28.5],
    '-COO- (Ester)':    ['[CX3](=O)[OX2H0]',      390.0, 490.0,   7000.0,  18.0],
    '-OH (Alcohol)':    ['[OX2H1;!$(OC=O)]',       210.0, 500.0,  20000.0,  10.0],
    '-O- (Ether/Sugar)':['[OX2H0;!$(OC=O)]',      100.0, 400.0,   3000.0,   3.8],
    '>C=O (Ketone)':    ['[CX3H0](=O)',            290.0, 770.0,   2000.0,  10.8],
    'Aromatic C':       ['[c]',                    211.6,  18.3,      0.0,   8.7],
    '-Cl (Chloride)':   ['[Cl]',                   450.0, 550.0,    400.0,  24.0],
    '-NH2':             ['[NX3H2]',                280.0,   0.0,   8400.0,  19.2],
    '-NH- (Amide)':     ['[NX3H1]',                160.0, 210.0,   3100.0,   4.5],
    '>N- (Tertiary)':   ['[NX3H0]',                 20.0, 800.0,   5000.0,  -9.0],
}

# =============================================================================
# 2. MOLECULE DATABASE (SMILES)
# =============================================================================
# IMPORTANT NOTE ON IONIC COMPOUNDS
# ──────────────────────────────────
# Several HBA components are salts (Choline Chloride, Betaine Chloride,
# Sodium Propionate, N4444Br). Their charged atoms ([N+], [Cl-], [Br-], [Na+])
# are NOT represented in the HVK database above, so their group-sum contributions
# will be underestimated. For rigorous results, these species require either:
#   (a) dedicated ionic HVK parameters (e.g. from Zhao et al. 2015), or
#   (b) replacement with their neutral equivalents as a conservative approximation.
# This is flagged here so results for ionic HBAs are interpreted with caution.


component_db = {
    'Choline_Chloride':   'C[N+](C)(C)CCO.[Cl-]',      # ionic — see note above
    'Betaine_Chloride':   'C[N+](C)(C)CC(=O)O.[Cl-]',  # ionic — see note above
    'Sodium_Propionate':  'CCC(=O)[O-].[Na+]',          # ionic — see note above
    'L-Proline':          'OC(=O)[C@@H]1CCCN1',
    'Betaine':            'C[N+](C)(C)CC(=O)[O-]',      # zwitterion — see note above
    'Urea':               'C(=O)(N)N',
    'Glycerol':           'C(C(CO)O)O',
    'Ethylene_Glycol':    'C(CO)O',
    '1,2_Butanediol':     'CCC(CO)O',
    '1,3_Butanediol':     'CC(CCO)O',
    '1,4_Butanediol':     'C(CCO)CO',
    '2,3_Butanediol':     'CC(C(C)O)O',
    '1,6_Hexanediol':     'C(CCCO)CCO',
    'Malic_Acid':         'C(C(C(=O)O)O)C(=O)O',
    'Malonic_Acid':       'C(C(=O)O)C(=O)O',
    'Citric_Acid':        'C(C(=O)O)C(CC(=O)O)(C(=O)O)O',
    'Levulinic_Acid':     'CC(=O)CCC(=O)O',
    'Lactic_Acid':        'CC(C(=O)O)O',
    'Oxalic_Acid':        'C(=O)(C(=O)O)O',
    'Fructose':           'C1[C@H]([C@H]([C@@H](C(O1)(CO)O)O)O)O',
    'Glucose':            'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O',
    'Sucrose':            ('C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)'
                           'O[C@]2([C@H]([C@@H]([C@H](O2)CO)O)O)CO)O)O)O)O'),
    'Sorbitol':           'C([C@H]([C@H]([C@@H]([C@H](CO)O)O)O)O)O',
    'Menthol':            'CC1CCC(C(C1)C(C)C)O',
    'Thymol':             'CC1=CC(=C(C=C1)C(C)C)O',
    'N4444Br':            'CCCC[N+](CCCC)(CCCC)CCCC.[Br-]',  # ionic — see note above
    'Lauric_Acid':        'CCCCCCCCCCCC(=O)O',
    'Dodecanol':          'CCCCCCCCCCCCO',
    'Octanoic_Acid':      'CCCCCCCC(=O)O',
}

# =============================================================================
# 3. THERMODYNAMIC FUNCTIONS & HOLE THEORY
# =============================================================================
def calculate_pure_properties(smiles_string):
    """
    Calculates HSP (dD, dP, dH), molar volume (Vm), molecular weight (MW),
    and density (rho) from a SMILES string using Hoftyzer-Van Krevelen
    group-contribution method.

    Returns None if the SMILES cannot be parsed or if Vm ≤ 0.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if not mol:
        print(f"  [WARNING] Could not parse SMILES: {smiles_string}")
        return None

    # Add explicit hydrogens so that HVK SMARTS patterns (e.g. [CX4H3]) match correctly
    mol = Chem.AddHs(mol)
    num_rings = mol.GetRingInfo().NumRings()

    sum_Fdi  = 0.0
    sum_Fpi2 = 0.0
    sum_Ehi  = 0.0
    sum_Vmi  = 0.0
    matched_atoms: set = set()

    for group_name, data in hvk_database.items():
        pattern = Chem.MolFromSmarts(data[0])
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            for match in mol.GetSubstructMatches(pattern):
                # Skip this match if any of its atoms have already been
                # assigned to a previously matched group (prevents double-counting).
                if not any(atom_idx in matched_atoms for atom_idx in match):
                    sum_Fdi  += data[1]
                    sum_Fpi2 += data[2]
                    sum_Ehi  += data[3]
                    sum_Vmi  += data[4]
                    matched_atoms.update(match)

    # FIX: ring closure REDUCES molar volume (more compact packing) — subtract correction
    sum_Vmi -= (num_rings * V_RING_CORRECTION)

    if sum_Vmi <= 0:
        print(f"  [DEBUG] Vm={sum_Vmi:.3f} → None for: {smiles_string}")
        return None

    # Hansen Solubility Parameters (MPa^0.5)
    dD = sum_Fdi / sum_Vmi
    dP = np.sqrt(sum_Fpi2) / sum_Vmi
    dH = np.sqrt(sum_Ehi  / sum_Vmi)

    # Molecular weight and density from RDKit (exact, not estimated)
    MW  = Descriptors.ExactMolWt(mol)   # g/mol
    rho = MW / sum_Vmi                  # g/cm³

    return {'dD': dD, 'dP': dP, 'dH': dH, 'Vm': sum_Vmi, 'MW': MW, 'rho': rho}


def calculate_hole_theory_viscosity(Vm_cm3: float, delta_total: float, T_kelvin: float) -> float:
    """
    Estimates dynamic viscosity (cP) using Eyring's Hole Theory of Liquids.

    Parameters
    ----------
    Vm_cm3      : molar volume (cm³/mol)
    delta_total : total Hildebrand solubility parameter (MPa^0.5)
    T_kelvin    : temperature (K)

    Returns
    -------
    Viscosity in cP (mPa·s)
    """
    Vm_m3 = Vm_cm3 * 1e-6  # convert cm³/mol → m³/mol

    # Cohesive energy density (MPa = J/cm³)
    CED = delta_total ** 2

    # Molar energy of vaporisation (J/mol)
    E_vap = CED * Vm_cm3

    # Activation free energy for viscous flow (hole creation)
    dG_star = K_VISC * E_vap

    # Eyring pre-exponential factor (N_A · h / Vm)
    pre_exp = (N_A * H_PLANCK) / Vm_m3

    # Dynamic viscosity (Pa·s → cP)
    eta_Pa_s = pre_exp * np.exp(dG_star / (R_GAS * T_kelvin))
    return eta_Pa_s * 1000.0  # Pa·s → cP


# =============================================================================
# 4. DES SCREENING ENGINE  (molar ratios 1:5 through 5:1)
# =============================================================================
db_pure   = {name: calculate_pure_properties(smiles) for name, smiles in component_db.items()}

# All nine molar ratios: 1:5, 1:4, 1:3, 1:2, 1:1, 2:1, 3:1, 4:1, 5:1
ratios = [(1,5), (1,4), (1,3), (1,2), (1,1), (2,1), (3,1), (4,1), (5,1)]

# ─── HBA / HBD pairs to screen ──────────────────────────────────────────────
des_pairs = [
    ('Choline_Chloride', 'Urea'),
    ('Choline_Chloride', 'Glycerol'),
    ('Choline_Chloride', 'Ethylene_Glycol'),
    ('Choline_Chloride', '1,2_Butanediol'),
    ('Choline_Chloride', '1,3_Butanediol'),
    ('Choline_Chloride', '1,4_Butanediol'),
    ('Choline_Chloride', '2,3_Butanediol'),
    ('Choline_Chloride', '1,6_Hexanediol'),
    ('Choline_Chloride', 'Malic_Acid'),
    ('Choline_Chloride', 'Malonic_Acid'),
    ('Choline_Chloride', 'Citric_Acid'),
    ('Choline_Chloride', 'Levulinic_Acid'),
    ('Choline_Chloride', 'Lactic_Acid'),
    ('Choline_Chloride', 'Oxalic_Acid'),
    ('Choline_Chloride', 'Fructose'),
    ('Choline_Chloride', 'Glucose'),
    ('Choline_Chloride', 'Sucrose'),
    ('Choline_Chloride', 'Sorbitol'),
    ('Betaine_Chloride', 'Urea'),
    ('Betaine_Chloride', 'Glycerol'),
    ('Betaine_Chloride', 'Ethylene_Glycol'),
    ('Betaine_Chloride', '1,2_Butanediol'),
    ('Betaine_Chloride', '1,3_Butanediol'),
    ('Betaine_Chloride', '1,4_Butanediol'),
    ('Betaine_Chloride', '2,3_Butanediol'),
    ('Betaine_Chloride', '1,6_Hexanediol'),
    ('Betaine_Chloride', 'Malic_Acid'),
    ('Betaine_Chloride', 'Malonic_Acid'),
    ('Betaine_Chloride', 'Citric_Acid'),
    ('Betaine_Chloride', 'Levulinic_Acid'),
    ('Betaine_Chloride', 'Lactic_Acid'),
    ('Betaine_Chloride', 'Oxalic_Acid'),
    ('Betaine_Chloride', 'Fructose'),
    ('Betaine_Chloride', 'Glucose'),
    ('Betaine_Chloride', 'Sucrose'),
    ('Betaine_Chloride', 'Sorbitol'),
    ('Sodium_Propionate', 'Urea'),
    ('Sodium_Propionate', 'Glycerol'),
    ('Sodium_Propionate', 'Ethylene_Glycol'),
    ('Sodium_Propionate', '1,2_Butanediol'),
    ('Sodium_Propionate', '1,3_Butanediol'),
    ('Sodium_Propionate', '1,4_Butanediol'),
    ('Sodium_Propionate', '2,3_Butanediol'),
    ('Sodium_Propionate', '1,6_Hexanediol'),
    ('Sodium_Propionate', 'Malic_Acid'),
    ('Sodium_Propionate', 'Malonic_Acid'),
    ('Sodium_Propionate', 'Citric_Acid'),
    ('Sodium_Propionate', 'Levulinic_Acid'),
    ('Sodium_Propionate', 'Lactic_Acid'),
    ('Sodium_Propionate', 'Oxalic_Acid'),
    ('Sodium_Propionate', 'Fructose'),
    ('Sodium_Propionate', 'Glucose'),
    ('Sodium_Propionate', 'Sucrose'),
    ('Sodium_Propionate', 'Sorbitol'),
    ('L-Proline', 'Urea'),
    ('L-Proline', 'Glycerol'),
    ('L-Proline', 'Ethylene_Glycol'),
    ('L-Proline', '1,2_Butanediol'),
    ('L-Proline', '1,3_Butanediol'),
    ('L-Proline', '1,4_Butanediol'),
    ('L-Proline', '2,3_Butanediol'),
    ('L-Proline', '1,6_Hexanediol'),
    ('L-Proline', 'Malic_Acid'),
    ('L-Proline', 'Malonic_Acid'),
    ('L-Proline', 'Citric_Acid'),
    ('L-Proline', 'Levulinic_Acid'),
    ('L-Proline', 'Lactic_Acid'),
    ('L-Proline', 'Oxalic_Acid'),
    ('L-Proline', 'Fructose'),
    ('L-Proline', 'Glucose'),
    ('L-Proline', 'Sucrose'),
    ('L-Proline', 'Sorbitol'),
    ('Betaine', 'Urea'),
    ('Betaine', 'Glycerol'),
    ('Betaine', 'Ethylene_Glycol'),
    ('Betaine', '1,2_Butanediol'),
    ('Betaine', '1,3_Butanediol'),
    ('Betaine', '1,4_Butanediol'),
    ('Betaine', '2,3_Butanediol'),
    ('Betaine', '1,6_Hexanediol'),
    ('Betaine', 'Malic_Acid'),
    ('Betaine', 'Malonic_Acid'),
    ('Betaine', 'Citric_Acid'),
    ('Betaine', 'Levulinic_Acid'),
    ('Betaine', 'Lactic_Acid'),
    ('Betaine', 'Oxalic_Acid'),
    ('Betaine', 'Fructose'),
    ('Betaine', 'Glucose'),
    ('Betaine', 'Sucrose'),
    ('Betaine', 'Sorbitol'),
    ('Menthol',  'Lauric_Acid'),
    ('Menthol',  'Dodecanol'),
    ('Menthol',  'Octanoic_Acid'),
    ('Thymol',   'Lauric_Acid'),
    ('Thymol',   'Dodecanol'),
    ('Thymol',   'Octanoic_Acid'),
    ('N4444Br',  'Lauric_Acid'),
    ('N4444Br',  'Dodecanol'),
    ('N4444Br',  'Octanoic_Acid'),
    ('N4444Br',  'Urea'),
    ('N4444Br',  'Glycerol'),
    ('N4444Br',  'Ethylene_Glycol'),
    ('N4444Br',  '1,2_Butanediol'),
    ('N4444Br',  '1,3_Butanediol'),
    ('N4444Br',  '1,4_Butanediol'),
    ('N4444Br',  '2,3_Butanediol'),
    ('N4444Br',  '1,6_Hexanediol'),
    ('N4444Br',  'Malic_Acid'),
    ('N4444Br',  'Malonic_Acid'),
    ('N4444Br',  'Citric_Acid'),
    ('N4444Br',  'Levulinic_Acid'),
    ('N4444Br',  'Lactic_Acid'),
    ('N4444Br',  'Oxalic_Acid'),
    ('N4444Br',  'Fructose'),
    ('N4444Br',  'Glucose'),
    ('N4444Br',  'Sucrose'),
    ('N4444Br',  'Sorbitol'),
]

screening_results = []

for hba_name, hbd_name in des_pairs:
    props_hba = db_pure.get(hba_name)
    props_hbd = db_pure.get(hbd_name)
    if props_hba is None or props_hbd is None:
        continue

    for r_hba, r_hbd in ratios:
        x_hba = r_hba / (r_hba + r_hbd)   # mole fraction of HBA
        x_hbd = r_hbd / (r_hba + r_hbd)   # mole fraction of HBD

        # ── Ideal mixing rules ────────────────────────────────────────────
        Vm_mix  = (x_hba * props_hba['Vm']) + (x_hbd * props_hbd['Vm'])   # cm³/mol
        phi_hba = (x_hba * props_hba['Vm']) / Vm_mix   # volume fraction HBA
        phi_hbd = (x_hbd * props_hbd['Vm']) / Vm_mix   # volume fraction HBD

        MW_mix  = (x_hba * props_hba['MW'])  + (x_hbd * props_hbd['MW'])  # g/mol
        rho_mix = MW_mix / Vm_mix                                          # g/cm³

        # ── Mixed HSP (volume-fraction weighted) ─────────────────────────
        dD_mix = (phi_hba * props_hba['dD']) + (phi_hbd * props_hbd['dD'])
        dP_mix = (phi_hba * props_hba['dP']) + (phi_hbd * props_hbd['dP'])
        dH_mix = (phi_hba * props_hba['dH']) + (phi_hbd * props_hbd['dH'])

        # Total Hildebrand parameter of the mixture
        delta_mix = np.sqrt(dD_mix**2 + dP_mix**2 + dH_mix**2)

        # ── Mixture viscosity via Eyring Hole Theory ──────────────────────
        eta_mix = calculate_hole_theory_viscosity(Vm_mix, delta_mix, T)

        row = {
            'HBA':               hba_name,
            'HBD':               hbd_name,
            'Ratio':             f"{r_hba}:{r_hbd}",
            'Density_g_cm3':     round(rho_mix,    3),
            'Visc_HoleTheory_cP':round(eta_mix,    3),
            'Vm_cm3_mol':        round(Vm_mix,      2),
        }

        screening_results.append(row)

# =============================================================================
# 5. FILTERING & OUTPUT
# =============================================================================
df_screening = pd.DataFrame(screening_results)

df_sorted = df_screening.sort_values(by='Visc_HoleTheory_cP', ascending=True)

output_file = 'Screening_DES_HoleTheory_Physicochemical.csv'
df_sorted.to_csv(output_file, index=False)

print(f"Thermodynamic screening complete. Results saved to '{output_file}'.")
print("\nSample Output (Top 5 Lowest Viscosity Predictions):")
print(
    df_sorted[[
        'HBA', 'HBD', 'Ratio',
        'Density_g_cm3',
        'Visc_HoleTheory_cP',
        'Vm_cm3_mol',
    ]].head(5).to_string(index=False)
)
end_time = time.time()
total_time = end_time - time_start
print(f"Total Time for {len(df_sorted)} DES: {round(total_time, 2)} sec")