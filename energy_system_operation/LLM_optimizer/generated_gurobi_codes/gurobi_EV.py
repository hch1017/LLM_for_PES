from gurobipy import Model, GRB, quicksum

def solve_ev_charging(V, T, P, x_depart, t_arrival, t_depart, u_min, u_max, x_initial, delta):
    # Create the model
    m = Model("EV_Charging")

    # Define decision variables
    u = m.addVars(V, T, name="u")
    x = m.addVars(V, T, name="x")

    # Define auxiliary variables for absolute value
    abs_diff = m.addVars(V, name="abs_diff")

    # Set bounds for the charging power variables after creation
    for j in V:
        for t in T:
            if t >= t_arrival[j] and t < t_depart[j]:
                u[j, t].lb = u_min[j]
                u[j, t].ub = u_max[j]
            else:
                u[j, t].lb = 0
                u[j, t].ub = 0

    # Objective function: Minimize the sum of squares of differences from desired final state
    # m.setObjective(quicksum( (x[j, t_depart[j]] - x_depart[j])**2 for j in V), GRB.MINIMIZE)
    m.setObjective(quicksum(abs_diff[j] for j in V), GRB.MINIMIZE)
    # Constraints for absolute value representation
    for j in V:
        m.addConstr(abs_diff[j] >= x[j, t_depart[j]] - x_depart[j])
        m.addConstr(abs_diff[j] >= x_depart[j] - x[j, t_depart[j]])

    # Constraints
    # Total power constraint at each time step
    m.addConstrs((quicksum(u[j, t] for j in V) <= P[t] for t in T), "power_capacity")

    # Initial state and battery update equations
    for j in V:
        m.addConstr(x[j, 0] == x_initial[j], "initial_state_{}".format(j))
        for t in range(1, len(T)):
            m.addConstr(x[j, t] == x[j, t-1] + delta * u[j, t], "state_update_{}_{}".format(j, t))

    # Solve the model
    m.optimize()

    # Output results
    if m.status == GRB.OPTIMAL:
        print("Optimal solution found:")
        print("Final Objective Value:", m.objVal)  # Display the final objective value
        for j in V:
            print(f"Vehicle {j} charging schedule:")
            for t in T:
                print(f"  Time {t}: Charge {u[j, t].X} kW")  # Display charging power at every time step
    else:
        print(f"Model status: {m.status}, No optimal solution found")

# Example data
V = [1, 2, 3]  # Vehicles
T = range(6)  # Time periods
P = {t: 60 for t in T}  # Power capacity limits at each time period
x_depart = {1: 80, 2: 80, 3: 80}  # Desired departure state of charge
x_initial = {1: 20, 2: 20, 3: 20}  # Initial state of charge
t_arrival = {1: 0, 2: 0, 3: 0}  # Arrival times
t_depart = {1: 5, 2: 4, 3: 3}  # Departure times
u_min = {1: 0, 2: 0, 3: 0}  # Minimum charging power
u_max = {1: 22, 2: 22, 3: 22}  # Maximum charging power
delta = 0.95  # Charging efficiency

# Call the solve function
solve_ev_charging(V, T, P, x_depart, t_arrival, t_depart, u_min, u_max, x_initial, delta)
