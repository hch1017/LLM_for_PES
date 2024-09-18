from gurobipy import Model, GRB, quicksum

# Parameters
T = range(12)  # Time periods, for example, 1 through 24 hours
c = {t: (25 + 10 * abs(t - 6)) for t in T}  # Simulated electricity prices
P = 50  # Minimum total power consumption
p_lower = {t: 2 for t in T}  # Minimum power consumption at time t
p_upper = {t: 10 for t in T}  # Maximum power consumption at time t
r = 5  # Maximum rate of change in power consumption between periods

# Initialize model
m = Model("Demand Response")

# Variables
p = m.addVars(T, lb=[p_lower[t] for t in T], ub=[p_upper[t] for t in T], name="p")

# Objective
m.setObjective(quicksum(c[t] * p[t] for t in T), GRB.MINIMIZE)

# Constraints
m.addConstr(quicksum(p[t] for t in T) >= P, name="min_total_power")

# Rate of change constraints
for t in T:
    if t > 1:
        m.addConstr(p[t] - p[t-1] <= r, name=f"rate_increase_{t}")
        m.addConstr(p[t-1] - p[t] <= r, name=f"rate_decrease_{t}")

# Solve model
m.optimize()

# Output solution
if m.status == GRB.OPTIMAL:
    print("Optimal value:", m.objVal)
    for t in T:
        print(f"Power at time {t}: {p[t].X:.2f} kW")
else:
    print("No optimal solution found.")
