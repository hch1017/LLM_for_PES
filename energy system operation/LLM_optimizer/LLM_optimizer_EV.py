import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys 
import torch
from openai import OpenAI
import time
import json

start_time = time.time()
print(torch.__version__)
print(torch.cuda.is_available())

########################### problem setup #####################

# Example data
V = [1, 2, 3]  # Vehicles
T = range(4)  # Time periods
P = {t: 60 for t in T}  # Power capacity limits at each time period
x_depart = {1: 80, 2: 80, 3: 80}  # Desired departure state of charge
x_initial = {1: 20, 2: 20, 3: 20}  # Initial state of charge
t_arrival = {1: 0, 2: 0, 3: 0}  # Arrival times
t_depart = {1: 3, 2: 2, 3: 1}  # Departure times
u_min = {1: 0, 2: 0, 3: 0}  # Minimum charging power
u_max = {1: 22, 2: 22, 3: 22}  # Maximum charging power
delta = 0.95  # Charging efficiency

power_demand = x_depart[1] - x_initial[1]

num_var = len(V) * len(T)

def calculate_objective(x, V, t_depart, x_depart):
    objective_value = sum(np.abs(x[j][t_depart[j]] - x_depart[j]) for j in V)
    return objective_value

def check_constraints(u, x, V, T, P, t_arrival, t_depart, u_min, u_max, x_initial, delta):
    # Check initial state and battery state update constraints
    # for j in V:
        # if x[j][0] != x_initial[j]:
        #     print('inital')
        #     return False  # Initial state constraint violated
        # for t in range(1, len(T)):
        #     if x[j][t] != x[j][t-1] + delta * u[j][t-1]:
        #         print('battery')
        #         return False  # Battery state update constraint violated
    
    # Check power constraints at each time step
    for t in T:
        # print(u)
        total_power_at_t = sum(u[j][t] for j in V if t >= t_arrival[j] and t <= t_depart[j])
        if total_power_at_t > P[t]:
            print('total')
            return False  # Power capacity constraint violated

    # Check bounds for charging power
    for j in V:
        for t in T:
            if not (u_min[j] <= u[j][t] <= u_max[j]):
                print('bound')
                return False  # Charging power bounds violated

    return True  # No constraints violated


# # Example solutions for charging power and state of charge
# u = {1: [0]*6 + [10]*14 + [0]*4}  # Charging profile for vehicle 1
# x = {1: [20] + [20 + 0.95*i*10 for i in range(23)]}  # State of charge for vehicle 1

# # Calculate objective value
# objective_value = calculate_objective(x, V, t_depart, x_depart)
# print("Objective Value:", objective_value)

# # Check if constraints are violated
# constraints_ok = check_constraints(u, x, V, T, P, t_arrival, t_depart, u_min, u_max, x_initial, delta)
# print("Constraints Violated:", not constraints_ok)

# sys.exit()


########################### prompt engineering and output process #####################

def is_number_isdigit(s): # function for parsing str response from LLM
    s_ = ''
    for i in range(len(s)):
        s_ *= s[i].replace('.','',1).replace('-','',1).strip().isdigit()
    return s_

def check_last_solutions(loss_list, last_nums): # function that stops optimization when the last 4 values of the loss function < 1
    if len(loss_list) >= last_nums:
        last = loss_list[-last_nums:]
        return all(num < 1 for num in last)

def create_prompt(num_var, num_sol, num_illegal_solutions, df, df_illegal, power_d, x_initial, x_depart, t_depart, t_arrival, P, delta, is_decimal): # create prompt
    var = ''
    value = ''
    for j in range(df.shape[1]-1):
        var += "p{},".format(j+1)
        value += "<value{}>,".format(j+1)
    var = var[:-1]
    value = value[:-1]

    x_ = {key: x_depart[key] - x_initial[key] for key in x_depart}
    t_ = {key: t_depart[key] - t_arrival[key] for key in t_depart}

    meta_prompt_start = f'''You need assistance in solving an EV charging scheduling optimization problem. This problem involves {num_var} optimization variables, \
     namely {var}. These variables are subject to constraints defined by their minimum and maximum values: p_min=0 and p_max=22. \
    Besides, the sum of {var} at each time slot has to be lower than {P[0]}.\
    The optimization objective is try to satisfy the terminal enery demand of each EV: {x_} within parking period {t_}.\
     Your objective is to provide values for {var} that satisfy the constraints and minimize the optimization objective. \
     BTW, the charging efficiency is {delta}. \
     Below are some previous solution and their objective value pairs. The pairs are arranged in descending order based on their function values, where lower values are better.\n\n'''

    solutions = ''
    if num_sol > len(df.loss):
        num_sol = len(df.loss)

    for i in range(num_sol):
        solution_str = 'input:\n'
        for j in range(df.shape[1]-1):
            solution_str += 'p{}={}'.format(j+1, df.iloc[-num_sol + i, j])
        solution_str += '\n function value:\n{}\n\n'.format(df.loss.iloc[-num_sol + i])
        solutions += solution_str
         
    
    assist_prompt = f'''The following solutions are illegal, which violate constraints. Thus, please do not give solutions same as them:'''
    df_illegal = df_illegal.drop_duplicates()
    if num_illegal_solutions > len(df_illegal):
        num_illegal_solutions = len(df_illegal)
    for i in range(num_illegal_solutions):
        col = df_illegal.iloc[-i,:].tolist()
        prompt = var + ':' + str(col)[1:-1] + '\n'
        assist_prompt += prompt

    if is_decimal:
        meta_prompt_end = f'''Now, without producing any additional text, please give me a new ({var}) pair that is different from all pairs above, and has a function value lower than
any of the above. The form of response must stritly follow the example: {var} = {value} where all values must be floating-point number with one decimal place.'''
    else:
        meta_prompt_end = f'''Now, without producing any additional text, please give me a new ({var}) pair that is different from all pairs above, and has a function value lower than
any of the above. The form of response must stritly follow the example: {var} = {value}, where all values must be integer.'''
    return meta_prompt_start + solutions + assist_prompt + meta_prompt_end



def get_x(output, x_initial):
    x_accumulated = {}
    for vehicle in output:
        initial_value = x_initial[vehicle]
        charge_values = output[vehicle]
        
        accumulated_values = []
        current_value = initial_value
        
        for value in charge_values:
            current_value += value
            accumulated_values.append(current_value)
        
        x_accumulated[vehicle] = accumulated_values

    return x_accumulated



################################ initialization #########################
client = OpenAI()



# 24h
# output = {1:[0,20,15,20,0,0],
#           2:[0,0,20,20,0,0],
#           3:[0,20,15,0,20,0],
#           4:[0,0,0,20,20,0]}
output = {1:[0,0,20,10],
          2:[20,9,20,0],
          3:[20,20,0,0]}

output_ = [value for sublist in output.values() for value in sublist]


x = get_x(output, x_initial)
# print(x)
loss = calculate_objective(x, V, t_depart, x_depart)

d = {}
d['loss'] = [loss]
for i in range(num_var):
    d['p{}'.format(i+1)] = [output_[i]]
loss_list = [loss] # collect all losses for plotting at the end
sol_list = []
for i in range(num_var):
    sol_list.append([output_[i]])
df = pd.DataFrame(data=d) # dataset to store all the proposed weights (w, b) and calculated loss
df.sort_values(by=['loss'], ascending=False, inplace=True)
df_illegal = pd.DataFrame()

num_sol = 40
num_solutions = 90 # number of observations to feed into the prompt
num_illegal_solutions = 20 # number of illegal solutions, like a rolling buffer in RL, only show the latest N illegal solutions
epochs = 100
temp = 1
num_samples = 10
model = 'gpt-4o'


var = ''
for j in range(df.shape[1]-1):
    var += "p{},".format(j+1)
var = var[:-1]
print(var)


################################ main simulation ####################

## simulation runner
## return df, df_illegal




def runner(model, df, df_illegal, loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, 
           V, T, x_depart, x_initial, t_arrival, t_depart, u_max, u_min,
           is_decimal=False, initialization=True, relax=False):
    for load_demand in loads:
        if initialization:
            output = {1:[0,0,20,10],
          2:[20,9,20,0],
          3:[0,20,20,0]}
            x = get_x(output, x_initial)
            loss = calculate_objective(x, V, t_depart, x_depart)
            output_ = [value for sublist in output.values() for value in sublist]

            d = {}
            d['loss'] = [loss]
            for i in range(num_var):
                d['p{}'.format(i+1)] = [output_[i]]
            loss_list = [loss] # collect all losses for plotting at the end
            sol_list = []
            for i in range(num_var):
                sol_list.append([output_[i]])
            df = pd.DataFrame(data=d) # dataset to store all the proposed weights (w, b) and calculated loss
            df.sort_values(by=['loss'], ascending=False, inplace=True)
            df_illegal = pd.DataFrame()
        else:
            loss_list = []
            sol_list = []
            for i in range(num_var):
                sol_list.append([])
        
        solution_record = {}
        
        for i in range(epochs):
            text = create_prompt(num_var, num_sol, num_illegal_solutions, df, df_illegal, power_demand, x_initial, x_depart, t_depart, t_arrival, P, delta, is_decimal)

            chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            temperature=temp,
            n=num_samples
            )

            solution_record[i] = {}

            for j in range(num_samples):
                output = chat_completion.choices[j].message.content

                try:
                    response = output.split(var+' =')[1].strip()
                except Exception as e:
                    continue
                #print('response:',response)
                
                if "\n" in response:
                    response = response.split("\n")[0].strip()
                    
                if "," in response:
                    numbers = response.split(',')
                # print('numbers',i,numbers)
                

                tmp_loss_list = []
                illegal_counter = 0
                sum_counter = 0
                minmax_counter = 0
                if len(numbers)==num_var:
                    # print(numbers)
                    # print(num_var)
                    # print(is_number_isdigit(numbers))
                    # if is_number_isdigit(numbers):
                    if True:
                        output = [float(numbers[i].strip()) for i in range(num_var)]
                        output_ = np.array(output)
                        
                        x_reconstructed = {}
                        index = 0
                        list_length = 4
                        for vehicle in V:
                            x_reconstructed[vehicle] = output_[index:index + list_length]
                            index += list_length
                        output = x_reconstructed
                        # rate_of_change_constraint(output_, T, r)
                        x = get_x(output, x_initial)
                        if check_constraints(output, x, V, T, P, t_arrival, t_depart, u_min, u_max, x_initial, delta):
                            print('legal, then record')
                            #print('thermal_output',thermal_)
                            loss = calculate_objective(x, V, t_depart, x_depart)
                            tmp_loss_list.append(loss)
                            for k in range(num_var):
                                sol_list.append(output_[k])
                            new_row = {'loss': loss}
                            for k in range(num_var):
                                new_row['p{}'.format(k+1)] = output_[k]
                            new_row_df = pd.DataFrame(new_row, index=[0])
                            df = pd.concat([df, new_row_df], ignore_index=True)
                            df.sort_values(by='loss', ascending=False, inplace=True)
                            print(f'loss={loss:.3f}')
                            solution_record[i][j] = np.append(output_, 0).tolist()
                        else:
                            illegal_counter += 1
                            new_row = {}
                            for k in range(num_var):
                                new_row['p{}'.format(k+1)] = output_[k]
                            new_row_df = pd.DataFrame(new_row, index=[0])
                            df_illegal = pd.concat([df_illegal, new_row_df], ignore_index=True)
                            solution_record[i][j] = np.append(output_, 1).tolist()
                # print('illegal', illegal_counter)
                if len(tmp_loss_list) != 0:
                    loss_list.append(np.min(tmp_loss_list))
                else:
                    loss_list.append(0)
            # print(solution_record)
            # sys.exit()

            # early stopping, not necessary here
            # if check_last_solutions(loss_list, 3):
                # break


        iterations = range(1, len(loss_list) + 1)
        torch.save(np.array(loss_list),'.\ev_results\loss_{}.pt'.format(load_demand[0]))
        for i in range(num_var):
            torch.save(np.array(sol_list[i]),'.\ev_results\p{}_{}.pt'.format(i+1, load_demand[0]))

        df.to_csv('.\ev_results\evgpt_{}.csv'.format(load_demand[0]), index=False)
        df_illegal.to_csv('.\ev_results\evgpt_{}_illegal.csv'.format(load_demand[0]), index=False)
        
        with open('ev_results/solution_{}.json'.format(load_demand[0]), 'w') as json_file:
            json.dump(solution_record, json_file)

        end_time = time.time()
        execution_time = end_time - start_time

        print("Program execution time:", execution_time, "seconds")

    return df, df_illegal


is_decimal = False
initialization = True
df = None
df_illegal = None
relax = False

loads = [P]

df, df_illegal = runner(model, df, df_illegal, loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, 
                        V, T, x_depart, x_initial, t_arrival, t_depart, u_max, u_min,
                        is_decimal=is_decimal, initialization=initialization, relax=relax)


################################ tight simulation ####################

initialization = False
relax = False
tight_loads = [{key: value - 10 for key, value in P.items()},
               {key: value - 20 for key, value in P.items()},
               {key: value - 30 for key, value in P.items()}]

df, df_illegal = runner(model, df, df_illegal, tight_loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, 
                        V, T, x_depart, x_initial, t_arrival, t_depart, u_max, u_min,
                        is_decimal=is_decimal, initialization=initialization, relax=relax)