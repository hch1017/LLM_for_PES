import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
import sys

# 4
openai.api_key = ''

from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from models.gaussian_process import get_gp_predictions_data
from models.darts import get_TCN_predictions_data, get_NHITS_predictions_data, get_NBEATS_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data
import time
import pickle

from jax import vmap
import jax.numpy as jnp

plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 15})

# task_names = ['power_quality_assessment', 'line_fault', 'etd', 'turbine_fault']
# task_full_names = ['power quality assessment', 'transmission line fault detection', 'energy theft detection', 'wind turbine fault detection']
# task_names = ['etd', 'turbine_fault']
# task_full_names = ['energy theft detection', 'wind turbine fault detection']
task_names = ['voltage']
task_full_names = ['voltage_anomaly_detection']


gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-', half_bin_correction=True)
)


model_hypers = {
    'LLMTime_GPT-4-turbo': {'model': 'gpt-4-turbo', **gpt4_hypers},
}

model_predict_fns = {
    'LLMTime_GPT-4-turbo': get_llmtime_predictions_data,
}

model_names = list(model_predict_fns.keys())
model = 'LLMTime_GPT-4-turbo'
for task_name, task_full_name in zip(task_names, task_full_names):
    if task_name == 'voltage':
        import numpy as np
        from collections import defaultdict

        df = pd.read_csv('D:\Monash\LLM_time_detection_voltage/datasets/vol_anomaly_llm.csv')
        df.rename(columns={'label': 'output'}, inplace=True)


        
    # 查看所有的unique label，这里是0,1,2,3
    values_to_extract = df.iloc[:,-1].unique().tolist()
    
    # 每个label的测试样本数 %20
    samples_per_values = [1, 1, 1, 1]
    
    # few-shot的个数
    instances_per_value = 10
    
    extracted_samples = pd.DataFrame()
    instances = pd.DataFrame()

    # 提取用于测试的样本
    for value, sample_per_value in zip(values_to_extract, samples_per_values):
        filtered_df = df[df.iloc[:,-1] == value]

        n_samples = min(len(filtered_df), sample_per_value)
        
        if n_samples > 0:
            samples = filtered_df.sample(n=n_samples, random_state=42)
            extracted_samples = pd.concat([extracted_samples, samples])


    # 提取few-shot样本
    for value in values_to_extract:
        filtered_df_ = extracted_samples[extracted_samples.iloc[:,-1] == value]

        n_samples = min(len(filtered_df_), instances_per_value)
        
        if n_samples > 0:
            samples = filtered_df_.sample(n=n_samples, random_state=42)
            instances = pd.concat([instances, samples])
            
    labels = extracted_samples.iloc[:,-1].tolist()
    inputs = np.array(extracted_samples.iloc[:, :-1])
    # print(labels)
    # print(inputs.shape)
    print(instances)
    
    
    # sys.exit()

    # 存储每个任务的输出和label
    output_list = []
    label_list = []

    for i in range(len(inputs)):
        train = inputs[i]
        test = [labels[i]]
        start_time = time.time()
        hypers = list(grid_iter(model_hypers[model]))
        num_samples = 3
        # 添加instances作为few-shot
        pred_dict = get_autotuned_predictions_data(train, test, instances, task_full_name, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
        # print(pred_dict)
        out = pred_dict
        end_time = time.time() - start_time
        out['time'] = end_time
        output_list.append(out['completions_list'][0])
        label_list.append(test[0])
        time.sleep(10)
        # if i == 3:
        # break

    path = 'D:\Monash\LLM_time_detection_voltage/{}_results/'.format(task_name)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save('D:\Monash\LLM_time_detection_voltage/{}_results/output_{}_{}_{}.npy'.format(task_name, task_name, instances_per_value, samples_per_values[0]), np.array(output_list))
    np.save('D:\Monash\LLM_time_detection_voltage/{}_results/label_{}_{}_{}.npy'.format(task_name, task_name, instances_per_value, samples_per_values[0]), np.array(label_list))

    final_output = []
    for i in range(len(output_list)):
        numbers = [int(num) for num in output_list[i]]
        from collections import Counter
        counter = Counter(numbers)
        most_common_number = counter.most_common(1)[0][0]
        final_output.append(most_common_number)

    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(label_list, final_output)
    print(f'{task_name}: Accuracy: {accuracy}')
    f1 = f1_score(label_list, final_output, average='macro')
    print(f'{task_name}: F1 Score (Macro): {f1}')



