import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai

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

# WPF, AggLoad, price, ResiLoad
# task_names = ['WPF', 'AggLoad', 'price', 'ResiLoad']
# task_full_names = ['wind_power_forecast', 'aggregated_load_forecast', 'wholesale_electricity_price_forecast', 'residential_load_forecast']
# task_names = ['aggregated_load_forecast', 'wholesale_electricity_price_forecast', 'residential_load_forecast']
task_names = ['WPF', 'ResiLoad']
task_full_names = ['wind_power_forecast', 'residential_load_forecast']


def plot_preds(train, test, pred_dict, model_name, task_name, var, end_time, flag, show_samples=False):
    if model_name == 'gp':
        model_name = 'Gaussian Process'

    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(6, 4))
    ind = range(len(train)+len(test))
    plt.plot(ind[:len(train)], train.values)
    plt.plot(ind[len(train):], test.values, label='Truth', color='black')
    plt.plot(ind[len(train):], pred.values, label=model_name, color='purple')
    diff = test - pred
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    # crps = calculate_crps(test, pred)
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(ind[len(train):], lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(ind[len(train):], samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.xticks([0, len(train), len(train)+len(test)])
    plt.ylim(None, max(train.values)*1.3)
    plt.legend(loc='upper left')
    plt.title(var)

    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.75, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        else:
            plt.text(0.03, 0.75, f'NLL/D: Null', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    else:
        plt.text(0.03, 0.75, f'NLL/D: Null', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.03, 0.65, f'MSE: {mse:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.03, 0.55, f'MAE: {mae:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.03, 0.45, f'Runtime: {end_time:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
        # plt.text(0.03, 0.55, f'CRPS: {crps:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    if not os.path.exists('.\{}_results/'.format(task_name)+var):
        os.makedirs('.\{}_results/'.format(task_name)+var)
    
    plt.savefig('.\{}_results\{}\{}_{}_{}_{}.pdf'.format(task_name, var, model_name, len(train), len(test), flag), bbox_inches='tight')    # plt.show()


gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-', half_bin_correction=True)
)

promptcast_hypers = dict(
    temp=1,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)



model_hypers = {
    'LLMTime_GPT-4': {'model': 'gpt-4', **gpt4_hypers},
    # 'PromptCast GPT-4': {'model': 'gpt-4', **promptcast_hypers},
}

model_predict_fns = {
    'LLMTime_GPT-4': get_llmtime_predictions_data,
    # 'PromptCast GPT-4': get_promptcast_predictions_data,
}

model_names = list(model_predict_fns.keys())
model = 'LLMTime_GPT-4'
for task_name, task_full_name in zip(task_names, task_full_names):
    if task_name == 'WPF':
        # df = pd.read_csv('D:\Monash\LLM_time\datasets\Turbine_Data.csv')
        # df.fillna(0,inplace=True)
        # df.rename(columns={'Unnamed: 0':'Time'}, inplace=True)
        # print(df.columns)
        # df = df.iloc[:, 1:]
        df = pd.read_excel('.\datasets\hourly_wind_power.xlsx')
    elif task_name == 'AggLoad':
        df = pd.read_csv('.\datasets\Aggregated.csv')
        df = df.groupby(df.index // 6).mean()/2
    elif task_name == 'price':
        df = pd.read_csv('.\datasets\Aggregated.csv')
        df = df.groupby(df.index // 6).mean()
    elif task_name == 'ResiLoad':
        df = pd.read_csv('.\datasets\cern_test.csv')
        l = []
        l_test = []
        for i in range(1000, 2000):
            avg = df[df['metre_id']==i]['kwh'].mean()
            if avg > 0.2 and avg < 0.5:
                l.append(i)
            else:
                if len(l_test) < 20:
                    l_test.append(i)
            if len(l) > 100:
                break

        df = df[df['metre_id'].isin(l)]

    import time

    if task_name == 'WPF':
        # cols = ['ActivePower']
        cols = ['WINDGEN']
        input_start = -192
        train_start = -10000
        
        input_len = 168
        test_start = -168
        test_loop = 24
    # interval: 5min
    # input: 1 day
    # output: 3 hours
    elif task_name == 'AggLoad':
        cols = ['TOTALDEMAND']
        input_start = -1500
        train_start = -80000
    
        input_len = 336
        test_start = -336 
        test_end = -1
        test_loop = 48
    elif task_name == 'price':
        cols = ['RRP']
        input_start = -288
        train_start = -80000
        
        input_len = 336
        test_start = -336
        test_end = -1
        test_loop = 48
    # interval: 30min
    # input: one week
    # output: 12 hours
    elif task_name == 'ResiLoad':
        cols = ['kwh']
        input_start = -384
        train_start = -100000
    
        input_len = 336
        test_start = -336
        test_loop = 48

    for c in cols:
        for i in range(int(-test_start/test_loop-1)):
            flag = test_start + i * test_loop
            train = df.iloc[-input_len+test_start+i*test_loop:test_start+i*test_loop, :][c]
            if test_start+(i+1)*test_loop == 0:
                test = df.iloc[test_start+i*test_loop:, :][c]
            else:
                test = df.iloc[test_start+i*test_loop:test_start+(i+1)*test_loop, :][c]
            
            start_time = time.time()
            model_hypers[model].update() # for promptcast
            hypers = list(grid_iter(model_hypers[model]))
            num_samples = 5
            pred_dict = get_autotuned_predictions_data(train, test, task_full_name, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
            out = pred_dict
            end_time = time.time() - start_time
            out['time'] = end_time
            plot_preds(train, test, pred_dict, model, task_name, c, end_time, flag, show_samples=True)
            time.sleep(10)
            with open('.\{}_results\{}_output_{}_{}.pkl'.format(task_name, task_name, model, flag), 'wb') as f:
                pickle.dump(out, f)