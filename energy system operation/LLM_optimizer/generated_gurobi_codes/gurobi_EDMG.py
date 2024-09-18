import gurobipy as gp
from gurobipy import GRB

# Parameters
N = 1  # Number of buses
T = 3  # Time periods (hours)

# Costs of generator for each bus ($/kWh)
C_gen = [0.05, 0.06, 0.04]

# Max power of solar panel for each bus (kW)
P_solar_max = [1, 1.2, 0.8]

# Max power of generator for each bus (kW)
P_gen_max = [2, 1.5, 2.5]

# Load demand for each bus at each time period (kW)
D = [
    [1.2, 0.8, 1.0],  # Bus 1
    [0.9, 1.1, 1.3],  # Bus 2
    [1.0, 1.2, 1.1]   # Bus 3
]

D1 = [
    [1.5, 1.1, 1.3],  # Bus 1
    [0.9, 1.1, 1.3],  # Bus 2
    [1.0, 1.2, 1.1]   # Bus 3
]

D2 = [
    [1.8, 1.4, 1.6],  # Bus 1
    [0.9, 1.1, 1.3],  # Bus 2
    [1.0, 1.2, 1.1]   # Bus 3
]

D3 = [
    [2.1, 1.7, 1.9],  # Bus 1
    [0.9, 1.1, 1.3],  # Bus 2
    [1.0, 1.2, 1.1]   # Bus 3
]

# Create a new model
m = gp.Model("microgrid_dispatch_multiple_buses")

# Decision variables
P_solar = m.addVars(N, T, lb=0, name="P_solar")
P_gen = m.addVars(N, T, lb=0, name="P_gen")

# Set upper bounds for solar and generator power
for i in range(N):
    for t in range(T):
        P_solar[i, t].UB = P_solar_max[i]
        P_gen[i, t].UB = P_gen_max[i]

# Objective: Minimize total generation cost
m.setObjective(gp.quicksum(C_gen[i] * P_gen[i, t] for i in range(N) for t in range(T)), GRB.MINIMIZE)

# Constraints
for i in range(N):
    for t in range(T):
        # Energy balance constraint
        m.addConstr(P_solar[i, t] + P_gen[i, t] == D3[i][t], name=f"energy_balance_{i}_{t}")

# Optimize the model
m.optimize()

# Output results
results = {
    "P_solar": [[P_solar[i, t].X for t in range(T)] for i in range(N)],
    "P_gen": [[P_gen[i, t].X for t in range(T)] for i in range(N)]
}

import pandas as pd

# Create a dataframe for displaying results
df = pd.DataFrame()

for i in range(N):
    for t in range(T):
        df = df.append({
            "Bus": i+1,
            "Time": t+1,
            "P_solar (kW)": results["P_solar"][i][t],
            "P_gen (kW)": results["P_gen"][i][t]
        }, ignore_index=True)

print(df)
