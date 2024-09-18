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

load_demand = 400 ##(Mw)
units = 5 # the total number of thermal units

output_min = [28,90,68,76,19] ## P_i^th,min (MW)
output_max = [206,284,189,266,53] ## P_i^th,max (MW)

coef_A = [3,4.05,4.05,3.99,3.88] ## a_i^th ($/MW^2)
coef_B = [20,18.07,15.55,19.21,26.18] ## b_i^th ($/MW)
coef_C = [100,98.87,104.26,107.21,95.31] ## c_i^th ($)

def calculate(thermal_output):
    # cost function
    cost = 0
    for i in range(units):
        cost += coef_A[i]*thermal_output[i]*thermal_output[i]+coef_B[i]*thermal_output[i]+coef_C[i]
    
    # Constranits violation
    max_operating_penalty = 0
    min_operating_penalty = 0
    balance_penalty = max(0, load_demand - sum(thermal_output))
    for i in range(units):
        max_operating_penalty += max(0, thermal_output[i]-output_max[i])
        min_operating_penalty += max(0, output_min[i]-thermal_output[i])
    return cost

def satisfy_load_constraints(thermal_output, load_demand):
    if np.sum(thermal_output)>=load_demand: return True
    else: return False
    
def satisfy_range_constraints(thermal_output):
    satisfies_constraints = all(output_min[i] <= thermal_output[i] <= output_max[i] for i in range(len(thermal_output)))
    return satisfies_constraints






########################### prompt engineering and output process #####################

def is_number_isdigit(s): # function for parsing str response from LLM
    n1 = s[0].replace('.','',1).replace('-','',1).strip().isdigit()
    n2 = s[1].replace('.','',1).replace('-','',1).strip().isdigit()
    n3 = s[2].replace('.','',1).replace('-','',1).strip().isdigit()
    n4 = s[3].replace('.','',1).replace('-','',1).strip().isdigit()
    n5 = s[4].replace('.','',1).replace('-','',1).strip().isdigit()
    return n1 * n2 * n3 * n4 * n5

def check_last_solutions(loss_list, last_nums): # function that stops optimization when the last 4 values of the loss function < 1
    if len(loss_list) >= last_nums:
        last = loss_list[-last_nums:]
        return all(num < 1 for num in last)


def create_prompt(num_sol, num_illegal_solutions, df, df_illegal, load_demand, is_decimal): # create prompt
    meta_prompt_start = f'''You need assistance in solving an optimization problem. This problem involves 5 optimization variables, \
     namely p1, p2, p3, p4, and p5. These variables are subject to constraints defined by their minimum and maximum values: p_min=[28, 90, 68, 76, 19] \
     and p_max=[206, 284, 189, 266, 53]. Additionally, the sum of p1, p2, p3, p4, and p5 must be greater than or equal to {load_demand}. \
     Your objective is to provide values for p1, p2, p3, p4, and p5 that satisfy the constraints and minimize the optimization objective. \
     Below are some previous solution and their objective value pairs. The pairs are arranged in descending order based on their function values, where lower values are better.\n\n'''

    solutions = ''
    if num_sol > len(df.loss):
        num_sol = len(df.loss)

    for i in range(num_sol):
        solutions += f'''input:\np1={df.p1.iloc[-num_sol + i]:.3f}, p2={df.p2.iloc[-num_sol + i]:.3f}, p3={df.p3.iloc[-num_sol + i]:.3f}, p4={df.p4.iloc[-num_sol + i]:.3f}, p5={df.p5.iloc[-num_sol + i]:.3f} \n function value:\n{df.loss.iloc[-num_sol + i]:.3f}\n\n''' 
    
    assist_prompt = f'''The following solutions are illegal, which violate constraints. Thus, please do not give solutions same as them:'''
    df_illegal = df_illegal.drop_duplicates()
    if num_illegal_solutions > len(df_illegal):
        num_illegal_solutions = len(df_illegal)
    for i in range(num_illegal_solutions):
        col = df_illegal.iloc[-i,:].tolist()
        prompt = "p1, p2, p3, p4, p5: " + str(col)[1:-1] + '\n'
        assist_prompt += prompt

    if is_decimal:
        meta_prompt_end = f'''Now, without producing any additional text, please give me a new (p1, p2, p3, p4, p5) pair that is different from all pairs above, and has a function value lower than
any of the above. The form of response must stritly follow the example: p1, p2, p3, p4, p5 = <value1>, <value2>, <value3>, <value4>, <value5>, where all values must be floating-point number with one decimal place.'''
    else:
        meta_prompt_end = f'''Now, without producing any additional text, please give me a new (p1, p2, p3, p4, p5) pair that is different from all pairs above, and has a function value lower than
any of the above. The form of response must stritly follow the example: p1, p2, p3, p4, p5 = <value1>, <value2>, <value3>, <value4>, <value5>, where all values must be integer.'''
    return meta_prompt_start + solutions + assist_prompt + meta_prompt_end



################################ initialization #########################
client = OpenAI()

thermal_output = np.array([135, 93, 70, 87, 30])
loss = calculate(thermal_output)

d = {'loss': [loss], 'p1': [thermal_output[0]], 'p2': [thermal_output[1]], 'p3': [thermal_output[2]], 'p4': [thermal_output[3]], 'p5': [thermal_output[4]]}
loss_list = [loss] # collect all losses for plotting at the end
p1_list = [thermal_output[0]]
p2_list = [thermal_output[1]]
p3_list = [thermal_output[2]]
p4_list = [thermal_output[3]]
p5_list = [thermal_output[4]]
df = pd.DataFrame(data=d) # dataset to store all the proposed weights (w, b) and calculated loss
df.sort_values(by=['loss'], ascending=False, inplace=True)
df_illegal = pd.DataFrame()

loads = [400]
tight_loads = [410, 430, 450]
relax_loads = [390, 370, 350]
num_solutions = 50 # number of observations to feed into the prompt
num_illegal_solutions = 100 # number of illegal solutions, like a rolling buffer in RL, only show the latest N illegal solutions
epochs = 200
temp = 1
num_samples = 5
model = 'gpt-3.5-turbo'



################################ main simulation ####################

## simulation runner
## return df, df_illegal

def runner(model, df, df_illegal, loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, is_decimal=False, initialization=True, relax=False):
    for load_demand in loads:
        if initialization:
            thermal_output = np.array([135, 93, 70, 87, 30])
            loss = calculate(thermal_output)

            d = {'loss': [loss], 'p1': [thermal_output[0]], 'p2': [thermal_output[1]], 'p3': [thermal_output[2]], 'p4': [thermal_output[3]], 'p5': [thermal_output[4]]}
            loss_list = [loss] # collect all losses for plotting at the end
            p1_list = [thermal_output[0]]
            p2_list = [thermal_output[1]]
            p3_list = [thermal_output[2]]
            p4_list = [thermal_output[3]]
            p5_list = [thermal_output[4]]
            df = pd.DataFrame(data=d) # dataset to store all the proposed weights (w, b) and calculated loss
            df.sort_values(by=['loss'], ascending=False, inplace=True)
            df_illegal = pd.DataFrame()
        else:
            loss_list = []
            p1_list = []
            p2_list = []
            p3_list = []
            p4_list = []
            p5_list = []

        
        if relax:
            indexes_to_drop = []
            for i in range(len(df_illegal)):
                thermal_output = df_illegal.iloc[i,:].tolist()
                if satisfy_load_constraints(thermal_output, load_demand) and satisfy_range_constraints(thermal_output):
                    indexes_to_drop.append(i)
            df_illegal.drop(index=indexes_to_drop, inplace=True)
        
        solution_record = {}
        
        for i in range(epochs):
            text = create_prompt(num_solutions, num_illegal_solutions, df, df_illegal, load_demand, is_decimal)

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
            )


            solution_record[i] = {}

            for j in range(num_samples):
                output = chat_completion.choices[j].message.content

                try:
                    response = output.split("p1, p2, p3, p4, p5 =")[1].strip()
                except Exception as e:
                    continue
                #print('response:',response)
                
                if "\n" in response:
                    response = response.split("\n")[0].strip()
                    
                if "," in response:
                    numbers = response.split(',')
                print('numbers',i,numbers)
                

                tmp_loss_list = []
                if len(numbers)==5:
                    if is_number_isdigit(numbers):
                        p1, p2, p3, p4, p5 = float(numbers[0].strip()), float(numbers[1].strip()), float(numbers[2].strip()), float(numbers[3].strip()), float(numbers[4].strip())
                        thermal_ = np.array([p1, p2, p3, p4, p5])
                        if satisfy_load_constraints(thermal_, load_demand) and satisfy_range_constraints(thermal_):
                            print('legal, then record')
                            #print('thermal_output',thermal_)
                            loss = calculate(thermal_)
                            tmp_loss_list.append(loss)
                            p1_list.append(p1)
                            p2_list.append(p2)
                            p3_list.append(p3)
                            p4_list.append(p4)
                            p5_list.append(p5)
                            new_row = {'loss': loss, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}
                            new_row_df = pd.DataFrame(new_row, index=[0])
                            df = pd.concat([df, new_row_df], ignore_index=True)
                            df.sort_values(by='loss', ascending=False, inplace=True)
                            print(f'loss={loss:.3f}')
                            solution_record[i][j] = np.append(thermal_, 0).tolist()
                        else:
                            print('illegal')
                            new_row = {'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5}
                            new_row_df = pd.DataFrame(new_row, index=[0])
                            df_illegal = pd.concat([df_illegal, new_row_df], ignore_index=True)
                            solution_record[i][j] = np.append(thermal_, 1).tolist()
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
        torch.save(np.array(loss_list),'.\opf_results\loss_{}.pt'.format(load_demand))
        torch.save(np.array(p1_list),'.\opf_results\p1_{}.pt'.format(load_demand))
        torch.save(np.array(p2_list),'.\opf_results\p2_{}.pt'.format(load_demand))
        torch.save(np.array(p3_list),'.\opf_results\p3_{}.pt'.format(load_demand))
        torch.save(np.array(p4_list),'.\opf_results\p4_{}.pt'.format(load_demand))
        torch.save(np.array(p5_list),'.\opf_results\p5_{}.pt'.format(load_demand))


        df.to_csv('.\opf_results\opfgpt_{}.csv'.format(load_demand), index=False)
        df_illegal.to_csv('.\opf_results\opfgpt_{}_illegal.csv'.format(load_demand), index=False)
        
        with open('opf_results/solution_{}.json'.format(load_demand), 'w') as json_file:
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
df, df_illegal = runner(model, df, df_illegal, loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, is_decimal=is_decimal, initialization=initialization, relax=relax)


################################ tight simulation ####################

initialization = False
relax = False
df, df_illegal = runner(model, df, df_illegal, tight_loads, temp, num_samples, epochs, num_solutions, num_illegal_solutions, is_decimal=is_decimal, initialization=initialization, relax=relax)