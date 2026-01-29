import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.car_raceing_continue import RacingCarEvaluation
from llm4ad.task.machine_learning.car_raceing_continue import template_program
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


def evaluate_init(test_results_root):
    task = RacingCarEvaluation(whocall='mmeoh')

    llm = HttpsApi(host='api.bltcy.ai',
                   key='sk-XXXX',
                   model='gpt-4o-mini',
                   timeout=120)

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

    env_seeds = [i for i in range(10)]
    score_images_dict = task.evaluate(action_select=program_callable, env_seeds=env_seeds)
    return score_images_dict['Test result for test']


def evaluate_path(test_results_root, policy_num=None, seeds=None):
    task = RacingCarEvaluation(whocall='mmeoh')  # , env_mode='human'

    # 构建population目录路径
    population_dir = os.path.join(test_results_root, 'population')

    # 检查population目录是否存在
    if not os.path.exists(population_dir):
        print(f"File {test_results_root} can't found")
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
    return score_images_dict['Test result for test']


def main(test_roots, policy_num=None, seeds=None):
    all_results = []

    for path in test_roots:
        print(f"\nEvaluating path: {path}")
        result = evaluate_path(path, policy_num=policy_num, seeds=seeds)
        if result is not None:
            all_results.append(result)
            print("Test result:")
            for key, value in result.items():
                print(f"  {key}: {value}")

    init_result = evaluate_init(path)
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
        print(f"  Average: {summary[metric]['average']:.3f}")
        print(f"  Best: {summary[metric]['best']:.3f}")
        print(f"Init {metric}: {summary[metric]['init_policy']:.3f}")
        print("  All values:")
        for i, value in enumerate(summary[metric]['all_values'], 1):
            print(f"    Path {i}: {value:.3f}")


if __name__ == '__main__':
    roots = {
        'MLES': [
            r'batch\mmEoh\v0526_1\20250527_010943_Problem_EoH',
            r'batch\mmEoh\v0526_8\20250529_035806_Problem_EoH',
            r'batch\mmEoh\v0526_2\20250527_010943_Problem_EoH',
            r'batch\mmEoh\v0526_22\20250607_170916_Problem_EoH',
            r'batch\mmEoh\v0526_10\20250605_112400_Problem_EoH'

        ],
        'eoh': [
            r'All\Eoh\v0526_0\20250526_174026_Problem_EoH',
            r'batch\Eoh\v0526_0\20250529_092724_Problem_EoH',
            r'batch\Eoh\v0526_1\20250529_175413_Problem_EoH',
            r'batch\Eoh\v0526_1\20250529_175544_Problem_EoH',
            r'batch\Eoh\v0526_2\20250529_232855_Problem_EoH'
        ]
    }

    # seeds = [i for i in range(10)]  # Testing
    seeds = (40, 1231, 516, 413)  # Training
    # seeds = None
    methods = ['meoh', 'eoh']

    index = 0
    policy_num = 8
    root_list = roots[methods[index]]
    main(root_list, policy_num=policy_num, seeds=seeds)
