from gurobipy import Model, GRB, quicksum


# load_demand = 410  # MW
# units = 5  

# load_demand = 600  # MW
# units = 10  

load_demand = 900  # MW
units = 15  


# minimum and maximum dispatch power（MW）
# output_min = [28, 90, 68, 76, 19]
# output_max = [206, 284, 189, 266, 53]
# output_min = [28, 90, 68, 76, 19, 28, 90, 68, 76, 19]
# output_max = [206, 284, 189, 266, 53, 206, 284, 189, 266, 53]
output_min = [28, 90, 68, 76, 19, 28, 90, 68, 76, 19, 28, 90, 68, 76, 19]
output_max = [206, 284, 189, 266, 53, 206, 284, 189, 266, 53, 206, 284, 189, 266, 53]

# cost coefficient
# coef_A = [3, 4.05, 4.05, 3.99, 3.88]
# coef_B = [20, 18.07, 15.55, 19.21, 26.18]
# coef_C = [100, 98.87, 104.26, 107.21, 95.31]

# coef_A = [3, 4.05, 4.05, 3.99, 3.88, 3, 4.05, 4.05, 3.99, 3.88]
# coef_B = [20, 18.07, 15.55, 19.21, 26.18, 20, 18.07, 15.55, 19.21, 26.18]
# coef_C = [100, 98.87, 104.26, 107.21, 95.31, 100, 98.87, 104.26, 107.21, 95.31]

coef_A = [3, 4.05, 4.05, 3.99, 3.88, 3, 4.05, 4.05, 3.99, 3.88, 3, 4.05, 4.05, 3.99, 3.88]
coef_B = [20, 18.07, 15.55, 19.21, 26.18, 20, 18.07, 15.55, 19.21, 26.18, 20, 18.07, 15.55, 19.21, 26.18]
coef_C = [100, 98.87, 104.26, 107.21, 95.31, 100, 98.87, 104.26, 107.21, 95.31, 100, 98.87, 104.26, 107.21, 95.31]

# build model
model = Model("OPF")

# decision variables
thermal_output = model.addVars(units, lb=output_min, ub=output_max, name="Power")

# objective: minimize total cost
cost = quicksum(coef_A[i] * thermal_output[i] * thermal_output[i] + coef_B[i] * thermal_output[i] + coef_C[i] for i in range(units))
model.setObjective(cost, GRB.MINIMIZE)

# constraints
# load demand constraint
model.addConstr(quicksum(thermal_output[i] for i in range(units)) >= load_demand, "LoadDemand")

model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for i in range(units):
        print(f"Unit {i+1} output: {thermal_output[i].X} MW")
        print(f"Optimal objective value (total cost): {model.ObjVal}")
else:
    print("No optimal solution found.")
