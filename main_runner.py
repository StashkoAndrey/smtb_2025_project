# ======= main_runner.py =======
import numpy as np
import pandas as pd
import csv
import pickle
from grid_5 import SimulationGrid
from diffusion_cpp import diffuse

# Simulation configuration
GRID_SIZE = 10
MAX_STEPS = 1000
REPLICATES = 10

# Load parameter sets
param_table = pd.read_csv("parameters_2.csv")

# ───────────── Nutrient splitter ─────────────
def _distribute_meal(total, major_frac):
    major = total * major_frac
    remainder = total - major
    split = np.random.rand()
    return major, remainder * split, remainder * (1 - split)

# ───────────── One replicate ─────────────
def run_replicate(param_row):
    # Extract required params
    globals().update(param_row.to_dict())
    
    g = SimulationGrid(GRID_SIZE, CHANNEL_COST)
    centre = GRID_SIZE // 2
    g.place_bacterium(centre, centre)

    pop_counts = []
    energy_levels = []
    phage_totals = []
    survival_time = MAX_STEPS

    for step in range(MAX_STEPS):
        g.step(step)

        # diffusion
        g.nutrients_A = diffuse(g.nutrients_A, DIFFUSION_RATE)
        g.nutrients_B = diffuse(g.nutrients_B, DIFFUSION_RATE)
        g.nutrients_C = diffuse(g.nutrients_C, DIFFUSION_RATE)
        g.phages_A = diffuse(g.phages_A, PHAGE_A_DIFFUSION_RATE)
        g.phages_B = diffuse(g.phages_B, PHAGE_B_DIFFUSION_RATE)
        g.phages_C = diffuse(g.phages_C, PHAGE_C_DIFFUSION_RATE)

        # nutrient refills
        if step and step % REFILL_INTERVAL_A == 0:
            maj, mB, mC = _distribute_meal(MEAL_TOTAL_A, MAJOR_MEAL_FRACTION)
            g.refill_nutrient_A(maj)
            g.refill_nutrient_B(mB)
            g.refill_nutrient_C(mC)
        if step >= DELTA_T and (step - DELTA_T) % REFILL_INTERVAL_B == 0:
            maj, mA, mC = _distribute_meal(MEAL_TOTAL_B, MAJOR_MEAL_FRACTION)
            g.refill_nutrient_B(maj)
            g.refill_nutrient_A(mA)
            g.refill_nutrient_C(mC)
        if step >= 2 * DELTA_T and (step - 2 * DELTA_T) % REFILL_INTERVAL_C == 0:
            maj, mA, mB = _distribute_meal(MEAL_TOTAL_C, MAJOR_MEAL_FRACTION)
            g.refill_nutrient_C(maj)
            g.refill_nutrient_A(mA)
            g.refill_nutrient_B(mB)

        alive = [cell for row in g.grid for cell in row if cell]
        pop_counts.append(len(alive))
        energy_levels.append(sum(cell.energy for cell in alive))
        phage_totals.append(g.total_phage_A() + g.total_phage_B() + g.total_phage_C())

        if len(alive) == 0:
            survival_time = step
            break

    return {
        'avg_bacteria': np.mean(pop_counts),
        'avg_phage': np.mean(phage_totals),
        'avg_energy': np.mean(energy_levels),
        'survival_time': survival_time
    }

# ───────────── Run all parameter sets ─────────────
with open('simulation_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['number_of_simulation', 'avg_bacteria', 'avg_phage', 'avg_energy', 'survival_time'])

    for i, row in param_table.iterrows():
        metrics = [run_replicate(row) for _ in range(REPLICATES)]
        A = np.mean([m['avg_bacteria'] for m in metrics])
        P = np.mean([m['avg_phage'] for m in metrics])
        E = np.mean([m['avg_energy'] for m in metrics])
        S = np.mean([m['survival_time'] for m in metrics])

        writer.writerow([i + 1, A, P, E, S])
        print(f"✓ Sim {i + 1} done", flush=True)
