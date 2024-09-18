from gurobipy import Model, GRB, quicksum

# Example parameters
num_generators = 3
num_periods = 24
demand = [50, 60, 70] + [80]*5 + [100]*12 + [80]*5
min_output = [20, 20, 30]
max_output = [50, 60, 90]
startup_cost = [10, 15, 20]
min_up_time = [3, 4, 2]
min_down_time = [2, 1, 1]

# Initialize the model
m = Model("Unit Commitment")

# Define variables
x = m.addVars(num_generators, num_periods, vtype=GRB.BINARY, name="x")
p = m.addVars(num_generators, num_periods, name="p")
# Auxiliary variable for the product of x and p^2
pp = m.addVars(num_generators, num_periods, name="pp")

# Quadratic cost coefficients
a = [0.5, 0.3, 0.2]  # Example quadratic coefficients

# Objective function: Minimize total cost
m.setObjective(
    quicksum(a[g] * pp[g, t] for g in range(num_generators) for t in range(num_periods)) +
    quicksum(startup_cost[g] * x[g, t] * (1 - (x[g, t-1] if t > 0 else 0)) for g in range(num_generators) for t in range(num_periods)),
    GRB.MINIMIZE)

# Constraints
# Power balance constraint
m.addConstrs((quicksum(p[g, t] * x[g, t] for g in range(num_generators)) >= demand[t] for t in range(num_periods)), "Demand")

# Generator output limits
for g in range(num_generators):
    for t in range(num_periods):
        m.addConstr(p[g, t] >= min_output[g] * x[g, t], f"MinPower_{g}_{t}")
        m.addConstr(p[g, t] <= max_output[g] * x[g, t], f"MaxPower_{g}_{t}")

# Link pp and p variables
for g in range(num_generators):
    for t in range(num_periods):
        m.addConstr(pp[g, t] <= max_output[g] * max_output[g] * x[g, t], f"ppUB_{g}_{t}")
        m.addConstr(pp[g, t] >= p[g, t] * p[g, t], f"ppLB_{g}_{t}")

# Minimum up time constraint
for g in range(num_generators):
    for t in range(num_periods):
        if t >= min_up_time[g]:
            m.addConstr(
                x[g, t] - x[g, t-1] <= quicksum(x[g, s] for s in range(t - min_up_time[g] + 1, t + 1)),
                f"MinUp_{g}_{t}"
            )

# Minimum down time constraint
for g in range(num_generators):
    for t in range(num_periods):
        if t >= min_down_time[g]:
            m.addConstr(
                x[g, t-1] - x[g, t] <= quicksum(1 - x[g, s] for s in range(t - min_down_time[g] + 1, t + 1)),
                f"MinDown_{g}_{t}"
            )

# Solve the model
m.optimize()

# Output results
if m.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    for g in range(num_generators):
        for t in range(num_periods):
            print(f"Generator {g}, Period {t}, Status: {x[g,t].X}, Output: {p[g,t].X if x[g,t].X > 0 else 0}")
else:
    print("No optimal solution.")
