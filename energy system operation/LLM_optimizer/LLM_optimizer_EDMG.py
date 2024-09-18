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
# print(torch.__version__)
# print(torch.cuda.is_available())

########################### problem setup #####################

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
    [1.2, 0.8, 1.0]  # Bus 1
]

    # [0.9, 1.1, 1.3],  # Bus 2
    # [1.0, 1.2, 1.1]   # Bus 3
D1 = [
    [1.5, 1.1, 1.3],  # Bus 1
]

D2 = [
    [1.8, 1.4, 1.6],  # Bus 1
]

D3 = [
    [2.1, 1.7, 1.9],  # Bus 1
]

# p_solar and p_gen should be in the shape of N*T

def objective_function(P_gen, C_gen, N, T):
    total_cost = 0
    for i in range(N):
        for t in range(T):
            total_cost += C_gen[i] * P_gen[i][t]
    return total_cost

def energy_balance_constraints(P_solar, P_gen, D, N, T):
    for i in range(N):
        for t in range(T):
            if not (P_solar[i][t] + P_gen[i][t] >= D[i][t]):
                return False
    return True

def power_output_constraints(P_solar, P_gen, P_solar_max, P_gen_max, N, T):
    for i in range(N):
        for t in range(T):
            if P_solar[i][t] < 0 or P_solar[i][t] > P_solar_max[i]:
                return False
            if P_gen[i][t] < 0 or P_gen[i][t] > P_gen_max[i]:
                return False
    return True

def generate_initial_solutions(P_solar, P_gen, P_solar_max, P_gen_max, D, N, T, num_solutions=10, perturbation=0.1):
    initial_solutions = []
    while len(initial_solutions) < num_solutions:
        new_P_solar = P_solar + np.random.uniform(-perturbation, perturbation, (N, T))
        new_P_gen = P_gen + np.random.uniform(-perturbation, perturbation, (N, T))
        
        new_P_solar = np.clip(new_P_solar, 0, P_solar_max)
        new_P_gen = np.clip(new_P_gen, 0, P_gen_max)
        
        if power_output_constraints(new_P_solar, new_P_gen, P_solar_max, P_gen_max, N, T) and \
           energy_balance_constraints(new_P_solar, new_P_gen, D, N, T):
            initial_solutions.append((new_P_solar, new_P_gen))
    return initial_solutions


########################### prompt engineering and output process #####################

def is_number_isdigit(s): # function for parsing str response from LLM
    s_ = ''
    for i in range(len(s)):
        s_ *= s[i].replace('.','',1).replace('-','',1).strip().isdigit()
    return s_


def generate_2d_array_string(N, T):
    array_str = "["
    counter = 1
    for i in range(N * T):
        pair = f"[p{counter}, p{counter + 1}]"
        if i < N * T - 1:
            pair += ", "
        array_str += pair
        counter += 2
    array_str += "]"
    return array_str

def create_prompt(num_var, num_sol, num_illegal_solutions, df, df_illegal, is_decimal, N, T, C_gen, P_solar_max, P_gen_max, D): # create prompt
    num_var = 2*N*T
    
    var = ''
    var_solar = ''
    var_gen = ''
    value = ''
    for j in range(num_var):
        if (j+1) % 2 == 0:
            var_solar += "p{},".format(j+1)
        else:
            var_gen += "p{},".format(j+1)
        var += "p{},".format(j+1)
        value += "<value{}>,".format(j+1)
    var = var[:-1]
    var_gen = var_gen[:-1]
    var_solar = var_solar[:-1]
    value = value[:-1]
    
    var_ = generate_2d_array_string(N, T)


    D_ = D[:N]
    cons1 = ''
    for i in range(len(D_[0])):
        D_inst = D_[0][i]
        cons1 += var_gen[(i+1)*3-3:(i+1)*3-1] + '+' + var_solar[(i+1)*3-3:(i+1)*3-1] + '>=' + str(D_inst) + ';'
    
    P_solar_max_ = P_solar_max[:N]
    P_gen_max_ = P_gen_max[:N]
    C_gen_ = C_gen[:N][0]
    

    meta_prompt_start = f'''You need to assist in solving an economic dispatch optimization problem. This problem involves {num_var} optimization variables. \
    Variables are {var_}, lists divided by time slots, e.g., p1, p3, p5 is the power output of generator at time slot 1, 2, 3, and p2, p4, p6 are for PV panels. \
    
    Objective:
    The objective is to minimize the total generation cost, calculated by multiplying generator cost {C_gen_} with total generator power, i.e., the sum of {var_gen}.\
    
    Constraints:
    These variables are subject to constraints defined by their minimum and maximum values: For {var_solar}, minimum=[0.8,0.8,0.8] and maximum=[1,1,1]. For {var_gen}, minimum=[0,0,0] and maximum=[2,2,2]. \
    Besides, the sum of generator output and PV output has to be larger than the load demand at each time slot: 
    For instance, {cons1}. \
    
    You need to provide values for {var_} that satisfy the above constraints and minimize the objective. \
    Below are some previous solution and their objective value pairs. The pairs are arranged in descending order based on their function values, where lower values are better.\n\n'''

    # print(meta_prompt_start)


    solutions = ''
    if num_sol > len(df.loss):
        num_sol = len(df.loss)

    for i in range(num_sol):
        solution_str = 'input:\n'
        for j in range(df.shape[1]-1):
            solution_str += 'p{}={},'.format(j+1, round(df.iloc[-num_sol + i, j], 1))
        solution_str += '\n function value:\n{}\n\n'.format(round(df.loss.iloc[-num_sol + i],3))
        solutions += solution_str
        # print(solutions)
         
    
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



################################ initialization #########################
client = OpenAI()

P_gen = [[0.2,0,0]]
P_solar = [[1,0.8,1]]
output = []
for i in range(len(P_gen)):
    for j in range(len(P_gen[i])):
        output.append(P_gen[i][j])
        output.append(P_solar[i][j])

loss = objective_function(P_gen, C_gen, N, T)

num_var = 2 * N * T

d = {}
d['loss'] = [loss]
for i in range(num_var):
    d['p{}'.format(i+1)] = [output[i]]
loss_list = [loss] # collect all losses for plotting at the end
sol_list = []
for i in range(num_var):
    sol_list.append([output[i]])
df = pd.DataFrame(data=d) # dataset to store all the proposed weights (w, b) and calculated loss
df.sort_values(by=['loss'], ascending=False, inplace=True)
df_illegal = pd.DataFrame()

loads = [D]

num_sol = 5
num_solutions = 5 # number of observations to feed into the prompt
num_illegal_solutions = 10 # number of illegal solutions, like a rolling buffer in RL, only show the latest N illegal solutions
epochs = 50
temp = 1
num_samples = 10

model = 'gpt-4o'


################################ main simulation ####################

import random

def generate_initial_solutions(num_solutions):
    solutions_solar = []
    solutions_gen = []
    constraints = [1.2, 0.8, 1]

    for _ in range(num_solutions):
        while True:
            solar = [[random.uniform(0.2, 1), random.uniform(0, 0.8), random.uniform(0.2, 1)]]
            gen = [[random.uniform(0, 1), random.uniform(0, 0.8), random.uniform(0, 1)]]
            
            total = [solar[0][i] + gen[0][i] for i in range(3)]
            
            if all(total[i] >= constraints[i] for i in range(3)):
                solutions_solar.append(solar)
                solutions_gen.append(gen)
                break

    return solutions_solar, solutions_gen


def runner(model, df, df_illegal, loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, 
           N, T, C_gen, P_solar_max, P_gen_max, D,
           is_decimal=False, initialization=True, relax=False):
    
    var = ''
    value = ''
    for j in range(num_var):
        var += "p{},".format(j+1)
        value += "<value{}>,".format(j+1)
    var = var[:-1]
    value = value[:-1]
    
    
    for load_demand in loads:
        print(load_demand)
        if initialization:
            P_gen = [[0.2,0,0]]
            P_solar = [[1,0.8,1]]
            d = {}
            loss_list = []
            sol_list = []
            
            num_solutions = 100
            initial_solar, initial_gen = generate_initial_solutions(num_solutions)
            
            # initial_solar = [[[1,0.8,1]],
            #                  [[0.9,0.7,0.9]],
            #                  [[0.8,0.6,0.8]],
            #                  [[0.7,0.5,0.7]],
            #                  [[0.6,0.4,0.6]],
            #                  [[0.5,0.3,0.5]],
            #                  [[0.4,0.2,0.4]],
            #                  [[0.3,0.1,0.3]],
            #                  [[0.2,0,0.2]],
            #                      [[0.5,0.5,0.5]], 
            #                      [[0.8,0.6,0.8]], 
            #                      [[0.3,0.3,0.3]]]
            
            # initial_gen = [[[0.2,0,0]],
            #                 [[0.3,0.1,0.1]],
            #                 [[0.4,0.2,0.2]],
            #                 [[0.5,0.3,0.3]],
            #                 [[0.6,0.4,0.4]],
            #                 [[0.7,0.5,0.5]],
            #                 [[0.8,0.6,0.6]],
            #                 [[0.9,0.7,0.7]],
            #                 [[1,0.8,0.8]],
            #                [[1,0.5,0.5]],
            #                [[0.4,0.2,0.2]],
            #                [[1,0.5,0.7]]]
            
            for P_solar, P_gen in zip(initial_solar, initial_gen):
                # print(f"Solution {i+1}:")
                # print("P_solar:", P_solar)
                # print("P_gen:", P_gen)
            
                output = []
                for i in range(len(P_gen)):
                    for j in range(len(P_gen[i])):
                        output.append(P_gen[i][j])
                        output.append(P_solar[i][j])
                loss = objective_function(P_gen, C_gen, N, T)
                d['loss'] = [loss]
                
                for i in range(num_var):
                    d['p{}'.format(i+1)] = [output[i]]
                loss_list.append(loss) # collect all losses for plotting at the end
               
                for i in range(num_var):
                    sol_list.append([output[i]])
            
            df = pd.DataFrame(data=d) # dataset to store all the proposed weights (w, b) and calculated loss
            df.sort_values(by=['loss'], ascending=False, inplace=True)
            df_illegal = pd.DataFrame()
        else:
            loss_list = []
            sol_list = []
            for i in range(num_var):
                sol_list.append([])
        
        solution_record = {}
        
        # for i in range(epochs):
        #     text = create_prompt(num_var, num_sol, num_illegal_solutions, df, df_illegal, is_decimal, N, T, C_gen, P_solar_max, P_gen_max, D)

        #     chat_completion = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": text,
        #         }
        #     ],
        #     temperature=temp,
        #     n=num_samples,
        #     # logprobs=True
        #     )

        #     solution_record[i] = {}

        #     for j in range(num_samples):
        #         output = chat_completion.choices[j].message.content
        #         # print(output)

        #         try:
        #             response = output.split(var+' =')[1].strip()
        #         except Exception as e:
        #             continue
        #         # print('response:',response)
                
        #         if "\n" in response:
        #             response = response.split("\n")[0].strip()
                    
        #         if "," in response:
        #             numbers = response.split(',')
        #         # print('numbers',numbers)
                

        #         tmp_loss_list = []
                
        #         if len(numbers)==num_var:
        #             # print(numbers)
        #             # print(num_var)
        #             # print(is_number_isdigit(numbers))
        #             # if is_number_isdigit(numbers):
        #             if True:
        #                 output = [float(numbers[i].strip()) for i in range(num_var)]
        #                 output_ = np.array(output)
        #                 P_gen = np.reshape(output_[::2], (N, T))
        #                 P_solar = np.reshape(output_[1::2], (N, T))
                        
        #                 print(output_)
        #                 print(P_gen)
        #                 print(P_solar)
                        
        #                 # print(P_solar)
        #                 # print(P_gen)
        #                 if energy_balance_constraints(P_solar, P_gen, load_demand, N, T) and power_output_constraints(P_solar, P_gen, P_solar_max, P_gen_max, N, T):
        #                     print('legal, then record')
        #                     loss = objective_function(P_gen, C_gen, N, T)
        #                     tmp_loss_list.append(loss)
        #                     for k in range(num_var):
        #                         sol_list.append(output_[k])
        #                     new_row = {'loss': loss}
        #                     for k in range(num_var):
        #                         new_row['p{}'.format(k+1)] = output_[k]
        #                     new_row_df = pd.DataFrame(new_row, index=[0])
        #                     df = pd.concat([df, new_row_df], ignore_index=True)
        #                     df.sort_values(by='loss', ascending=False, inplace=True)
        #                     print(f'loss={loss:.3f}')
        #                     solution_record[i][j] = np.append(output_, 0).tolist()
        #                 else:
        #                     print('illegal')
        #                     new_row = {}
        #                     for k in range(num_var):
        #                         new_row['p{}'.format(k+1)] = output_[k]
        #                     new_row_df = pd.DataFrame(new_row, index=[0])
        #                     df_illegal = pd.concat([df_illegal, new_row_df], ignore_index=True)
        #                     solution_record[i][j] = np.append(output_, 1).tolist()
        #         # print('illegal', illegal_counter)
        #         if len(tmp_loss_list) != 0:
        #             loss_list.append(np.min(tmp_loss_list))
        #         else:
        #             loss_list.append(0)

        iterations = range(1, len(loss_list) + 1)
        torch.save(np.array(loss_list),'.\edmg_results\loss_{}.pt'.format(load_demand[0]))
        for i in range(num_var):
            torch.save(np.array(sol_list[i]),'.\edmg_results\p{}_{}.pt'.format(i+1, load_demand[0]))

        df.to_csv('.\edmg_results\edmggpt_{}.csv'.format(load_demand[0]), index=False)
        df_illegal.to_csv('.\edmg_results\edmggpt_{}_illegal.csv'.format(load_demand[0]), index=False)
        
        with open('edmg_results/solution_{}.json'.format(load_demand[0]), 'w') as json_file:
            json.dump(solution_record, json_file)

        end_time = time.time()
        execution_time = end_time - start_time

        print("Program execution time:", execution_time, "seconds")

    return df, df_illegal




is_decimal = True
initialization = True
df = None
df_illegal = None
relax = False
df, df_illegal = runner(model, df, df_illegal, loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, 
                        N, T, C_gen, P_solar_max, P_gen_max, D,
                        is_decimal=is_decimal, initialization=initialization)


################################ tight simulation ####################

tight_loads = [[[1.5, 1.1, 1.3]],
                [[1.8, 1.4, 1.6]],
                [[2.1, 1.7, 1.9]]
    ]
initialization = False
relax = False
df, df_illegal = runner(model, df, df_illegal, tight_loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, 
                        N, T, C_gen, P_solar_max, P_gen_max, D,
                        is_decimal=is_decimal, initialization=initialization)