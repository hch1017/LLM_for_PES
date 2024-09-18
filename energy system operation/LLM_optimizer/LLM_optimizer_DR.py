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

#### Parameters ####

# time_period = 24
time_period = 12
one_day_power = 100
T = range(time_period)  # Time periods, for example, 1 through 24 hours
c = {t: (25 + 10 * abs(t - time_period/2)) for t in T}  # Simulated electricity prices
P = one_day_power/(24/time_period)  # Minimum total power consumption
p_lower = {t: 2 for t in T}  # Minimum power consumption at time t
p_upper = {t: 10 for t in T}  # Maximum power consumption at time t
r = 5  # Maximum rate of change in power consumption between periods

num_var = time_period

# obj
def calculate(p, c, T):
    total_cost = sum(c[t] * p[t] for t in T)
    return total_cost

# const1 True is for satisfaction
def min_total_power_constraint(p, T, P):
    total_power = sum(p[t] for t in T)
    return total_power >= P

# const2
def rate_of_change_constraint(p, T, r):
    for t in T:
        if t > 1:
            if abs(p[t] - p[t-1]) > r:
                return False
    return True

# const3
def power_bounds_constraint(p, T, p_lower, p_upper):
    for t in T:
        if not (p_lower[t] <= p[t] <= p_upper[t]):
            return False
    return True

########################### prompt engineering and output process #####################

def is_number_isdigit(s):
    try:
        s_ = [float(ss) for ss in s]  # try to convert str into float
        return True
    except ValueError:
        return False



# Include variables in the prompt
def create_prompt(num_var, num_sol, num_illegal_solutions, df, df_illegal, is_decimal, p_lower, p_upper, P, r): # create prompt
    var = ''
    value = ''
    for j in range(df.shape[1]-1):
        var += "p{},".format(j+1)
        value += "<value{}>,".format(j+1)
    var = var[:-1]
    value = value[:-1]

# The increase or decrease of power consumption in any two consecutive time periods in  should not exceed {r}
    meta_prompt_start = f'''You need assistance in solving an demand response optimization problem. This problem involves {num_var} optimization variables, \
     namely {var}. These variables are subject to constraints defined by their minimum and maximum values: p_min={str(list(p_lower.values()))} \
     and p_max={str(list(p_upper.values()))}. Your objective is to provide values for {var} that satisfy the constraints and minimize the optimization objective. \
     Constraints include: the sum of {var} must be greater than or equal to {P}.\
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



################################ initialization #########################
client = OpenAI()

# 12 h
output = np.array([2.00, 2.00, 2.00, 2.00, 3.00, 8.00, 10.00, 10.00, 5.00, 2.00, 2.00,
                     2.00])

loss = calculate(output, c, T)

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


num_solutions = 50 # number of observations to feed into the prompt
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




def runner(model, df, df_illegal, loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, is_decimal=False, initialization=True, relax=False):
    for load_demand in loads:
        # Should initialize if this is the first time and the corresponding lists are empty.
        if initialization:
            output = np.array([2.00, 2.00, 2.00, 2.00, 3.00, 8.00, 10.00, 10.00, 5.00, 2.00, 2.00,
                     2.00])
            loss = calculate(output, c, T)

            d = {}
            d['loss'] = [loss]
            for i in range(num_var):
                d['p{}'.format(i+1)] = [output[i]]
            loss_list = [loss]
            sol_list = []
            for i in range(num_var):
                sol_list.append([output[i]])
            df = pd.DataFrame(data=d)
            df.sort_values(by=['loss'], ascending=False, inplace=True)
            df_illegal = pd.DataFrame()
        # If not, inherit the two lists
        else:
            loss_list = []
            sol_list = []
            for i in range(num_var):
                sol_list.append([])

        
        # if constraints become slacker，then delete legal ones in the df_illegal list
        if relax:
            indexes_to_drop = []
            for i in range(len(df_illegal)):
                thermal_output = df_illegal.iloc[i,:].tolist()
                if satisfy_load_constraints(thermal_output, load_demand) and satisfy_range_constraints(thermal_output):
                    indexes_to_drop.append(i)
            df_illegal.drop(index=indexes_to_drop, inplace=True)
        
        solution_record = {}
        
        for i in range(epochs):
            text = create_prompt(num_var, num_solutions, num_illegal_solutions, df, df_illegal, is_decimal, p_lower, p_upper, P, r)

            chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            temperature=temp,
            n=num_samples,
            # logprobs=True
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
                    # print(is_number_isdigit(numbers))
                    if is_number_isdigit(numbers):
                        output = [float(numbers[i].strip()) for i in range(num_var)]
                        output_ = np.array(output)
                        # rate_of_change_constraint(output_, T, r)
                        if min_total_power_constraint(output_, T, load_demand) and power_bounds_constraint(output_, T, p_lower, p_upper):
                            print('legal, then record')
                            #print('thermal_output',thermal_)
                            loss = calculate(output_, c, T)
                            tmp_loss_list.append(loss)
                            for k in range(num_var):
                                sol_list.append(output[k])
                            new_row = {'loss': loss}
                            for k in range(num_var):
                                new_row['p{}'.format(k+1)] = output[k]
                            new_row_df = pd.DataFrame(new_row, index=[0])
                            df = pd.concat([df, new_row_df], ignore_index=True)
                            df.sort_values(by='loss', ascending=False, inplace=True)
                            print(f'loss={loss:.3f}')
                            solution_record[i][j] = np.append(output_, 0).tolist()
                        else:
                            # print('illegal')
                            if min_total_power_constraint(output_, T, load_demand):
                                sum_counter += 1
                            if power_bounds_constraint(output_, T, p_lower, p_upper):
                                minmax_counter += 1
                            illegal_counter += 1
                            new_row = {}
                            for k in range(num_var):
                                new_row['p{}'.format(k+1)] = output[k]
                            new_row_df = pd.DataFrame(new_row, index=[0])
                            df_illegal = pd.concat([df_illegal, new_row_df], ignore_index=True)
                            solution_record[i][j] = np.append(output_, 1).tolist()
                print('illegal', illegal_counter)
                print('sum:', sum_counter)
                print('minmax:', minmax_counter)
                if len(tmp_loss_list) != 0:
                    loss_list.append(np.min(tmp_loss_list))
                else:
                    loss_list.append(0)


        iterations = range(1, len(loss_list) + 1)
        torch.save(np.array(loss_list),'.\dr_results\loss_{}.pt'.format(load_demand))
        for i in range(num_var):
            torch.save(np.array(sol_list[i]),'.\p{}_{}.pt'.format(i+1, load_demand))

        # save the two lists for the future usage
        df.to_csv('.\dr_results\drgpt_{}.csv'.format(load_demand), index=False)
        df_illegal.to_csv('.\dr_results\drgpt_{}_illegal.csv'.format(load_demand), index=False)
        
        with open('dr_results/solution_{}.json'.format(load_demand), 'w') as json_file:
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

df, df_illegal = runner(model, df, df_illegal, loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, is_decimal=is_decimal, initialization=initialization, relax=relax)


################################ tight simulation ####################
# since the constraint becomes tighter, the illegal should be kept.
# The previous solutions should be also kept until any of them violate the new constraint, and then will be put in the illegal list.

initialization = False
relax = False
tight_loads = [P+5, P+10, P+15]

df, df_illegal = runner(model, df, df_illegal, tight_loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, is_decimal=is_decimal, initialization=initialization, relax=relax)