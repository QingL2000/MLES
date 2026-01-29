import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation
from llm4ad.task.machine_learning.moon_lander import template_program
# from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_https_mmeoh import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.mmeoh import MMEoH
from llm4ad.method.mmeoh import EoHProfiler

import matplotlib.pyplot as plt
import base64
from io import BytesIO

import glob
import re

import json
from llm4ad.base import TextFunctionProgramConverter as tfpc
import os


def evaluate_init(test_results_root, seeds=None, gravity=-10, enable_wind=False):
    task = MoonLanderEvaluation(whocall='mmeoh',
                                gravity=gravity,
                                enable_wind=enable_wind,)

    full_path = test_results_root + r'\samples\samples_best.json'
    print('Full path is', full_path)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as file:
            heurstics = json.load(file)
    else:
        print(f"File {test_results_root} can't found")
        return None

    heurstic = heurstics[0]['function']
    seed = tfpc.function_to_program(heurstic, template_program)
    functionname = tfpc.text_to_function(heurstic).name
    str_function = str(seed)

    all_globals_namespace = {}
    exec(str_function, all_globals_namespace)
    program_callable = all_globals_namespace[functionname]

    if seeds is None:
        raise ValueError("Must provied seed parameter")
    else:
        print(f"Current seed is {seeds}")
        env_seeds = seeds
    score_images_dict = task.evaluate(action_select=program_callable, env_seeds=env_seeds)
    return score_images_dict['Test result']


def evaluate_path(test_results_root, policy_num=None, seeds=None, gravity=-10, enable_wind=False):
    task = MoonLanderEvaluation(whocall='mmeoh',
                                gravity=gravity,
                                enable_wind=enable_wind)

    # 构建population目录路径
    population_dir = os.path.join(test_results_root, 'population')

    # 检查population目录是否存在
    if not os.path.exists(population_dir):
        print(f"File {population_dir} can't found")
        return None


    pop_files = glob.glob(os.path.join(population_dir, 'pop_*.json'))

    if not pop_files:
        print(f" {population_dir} has no pop_*.json file")
        return None

    def extract_number(filename):
        match = re.search(r'pop_(\d+)\.json$', os.path.basename(filename))
        return int(match.group(1)) if match else -1

    pop_files.sort(key=extract_number)
    last_pop_file = pop_files[-1]

    print('Newest pop file:', last_pop_file)

    try:
        with open(last_pop_file, 'r', encoding='utf-8') as file:
            heuristics = json.load(file)
    except Exception as e:
        print(f"Load {last_pop_file} wrong: {str(e)}")
        return None

    if policy_num is None or policy_num > len(heuristics):
        policy_num = len(heuristics)

    policies = []
    for i in range(policy_num):
        policy = heuristics[i]['function']

        seed = tfpc.function_to_program(policy, template_program)
        functionname = tfpc.text_to_function(policy).name
        str_function = str(seed)

        all_globals_namespace = {}
        exec(str_function, all_globals_namespace)
        program_callable = all_globals_namespace[functionname]
        policies.append(program_callable)
    if seeds is None:
        raise ValueError("Must provied seed parameter")
    else:
        print(f"Current seed is {seeds}")
        env_seeds = seeds

    score_images_dict = task.merge_evaluate(action_selects=policies, env_seeds=env_seeds)
    return score_images_dict['Test result']


def main(test_roots, policy_num=None, seeds=None, gravity=-10, enable_wind=False):
    all_results = []

    for path in test_roots:
        print(f"\nEvaluating path: {path}")
        result = evaluate_path(path, policy_num=policy_num, seeds=seeds, gravity=gravity, enable_wind=enable_wind)
        if result is not None:
            all_results.append(result)
            print("Test result:")
            for key, value in result.items():
                print(f"  {key}: {value}")

    init_result = evaluate_init(path, seeds=seeds, gravity=gravity, enable_wind=enable_wind)
    if init_result is not None:
        print("Init Test result:")
        for key, value in init_result.items():
            print(f"  {key}: {value}")

    if not all_results:
        print("No valid results obtained from any path.")
        return

    # Calculate average and best results for each metric
    metrics = ['Mean Reward', 'NWS']
    summary = {}

    for metric in metrics:
        values = [res[metric] for res in all_results]
        summary[metric] = {
            'average': sum(values) / len(values),
            'best': max(values) if metric in ['Mean Reward', 'Success Rate', 'NWS'] else min(values),
            # Always use max for best (higher is better)
            'all_values': values,
            'init_policy': init_result[metric]
        }

    print("\n=== Final Summary ===")
    print(f"Number of paths evaluated: {len(all_results)}")

    for metric in metrics:
        print(f"\n{metric}:")
        print(f"  Average: {summary[metric]['average']:.4f}")
        print(f"  Best: {summary[metric]['best']:.4f}")
        print(f"Init {metric}: {summary[metric]['init_policy']:.4f}")
        print("  All values:")
        for i, value in enumerate(summary[metric]['all_values'], 1):
            print(f"    Path {i}: {value:.4f}")

if __name__ == '__main__':
    roots = {
        'MLES': [
            r'batch\mmEoh\v0526_0\20250526_213816_Problem_EoH',
            r'batch\mmEoh\v0526_2\20250526_233234_Problem_EoH',
            r'batch\mmEoh\v0526_4\20250528_011828_Problem_EoH',
            r'batch\mmEoh\v0526_6\20250604_210831_Problem_EoH',
            r'batch\mmEoh\v0526_8\20250605_010332_Problem_EoH',
        ],
        'eoh': [
            r'batch\Eoh\v0526_1\20250527_104314_Problem_EoH',
            r'batch\Eoh\v0526_1\20250527_175911_Problem_EoH',
            r'batch\Eoh\v0526_2\20250528_131828_Problem_EoH',
            r'batch\Eoh\v0526_3\20250528_214635_Problem_EoH',
            r'batch\Eoh\v0526_3\20250528_214637_Problem_EoH'
        ]
    }

    # seeds = [i for i in range(10)]  # Testing
    seeds = [42, 520, 1231, 114, 886]    # Training
    # seeds = None
    methods = ['MLES', 'eoh']


    index = 1
    policy_num = None
    root_list = roots[methods[index]]
    main(root_list, policy_num=policy_num, seeds=seeds, gravity=-5, enable_wind=False)
