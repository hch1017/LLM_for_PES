from gurobipy import Model, GRB, quicksum


T = 24  # time slots
E_max = 200  # maximum battery capacity MWh
P_charge_max, P_discharge_max = 50, 50  # maximum charging/discharging power MW
efficiency = 0.9  # charging/discharging rate
buy_price = [0.03 if i < 12 else 0.10 for i in range(T)]  # electricity price USD/kWh
sell_price = [0.05 if i < 12 else 0.15 for i in range(T)]  # feed-in tariff USD/kWh

# build model
model = Model("Battery_Scheduling")

# decision variables
charge_power = model.addVars(T, lb=0, ub=P_charge_max, name="Charge")
discharge_power = model.addVars(T, lb=0, ub=P_discharge_max, name="Discharge")
battery_level = model.addVars(T, lb=0, ub=E_max, name="Energy")

# objective function: maximize revenue
profit = quicksum((sell_price[t] * discharge_power[t] - buy_price[t] * charge_power[t]) for t in range(T))
model.setObjective(profit, GRB.MAXIMIZE)

# constraints
model.addConstrs((battery_level[t] == battery_level[t-1] + (charge_power[t] * efficiency - discharge_power[t] / efficiency) if t > 0 else battery_level[t] == charge_power[t] * efficiency for t in range(T)), "EnergyBalance")

# do optimization
model.optimize()

# print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for t in range(T):
        print(f"Hour {t}: Charge = {charge_power[t].X} MW, Discharge = {discharge_power[t].X} MW, Battery Level = {battery_level[t].X} MWh")
else:
    print("No optimal solution found.")
