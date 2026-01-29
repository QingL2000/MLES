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

import json
from llm4ad.base import TextFunctionProgramConverter as tfpc
import os


def evaluate_init(test_results_root, seeds=None):
    task = RacingCarEvaluation(whocall='mmeoh')

    llm = HttpsApi(host='api.bltcy.ai',
                   key='sk-XXX',
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

    if seeds is None:
        raise ValueError("Must provide seed parameter")
    else:
        print(f"Current seed is {seeds}")
        env_seeds = seeds
    score_images_dict = task.evaluate(action_select=program_callable, env_seeds=env_seeds)
    return score_images_dict['Test result for test']


def evaluate_path(test_results_root, seeds=None):
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

    heurstic = heurstics[-1]['function']
    seed = tfpc.function_to_program(heurstic, template_program)
    functionname = tfpc.text_to_function(heurstic).name
    str_function = str(seed)

    all_globals_namespace = {}
    exec(str_function, all_globals_namespace)
    program_callable = all_globals_namespace[functionname]

    if seeds is None:
        raise ValueError("Must provide seed parameter")
    else:
        print(f"Current seed is {seeds}")
        env_seeds = seeds
    score_images_dict = task.evaluate(action_select=program_callable, env_seeds=env_seeds)
    return score_images_dict['Test result for test']


def main(test_roots, seeds=None):
    all_results = []

    for path in test_roots:
        print(f"\nEvaluating path: {path}")
        result = evaluate_path(path, seeds=seeds)
        if result is not None:
            all_results.append(result)
            print("Test result:")
            for key, value in result.items():
                print(f"  {key}: {value}")

    init_result = evaluate_init(path, seeds=seeds)
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

    seeds = [i for i in range(10)]  # Testing
    # seeds = (40, 1231, 516, 413)  # Training

    methods = ['eoh', 'MLES']
    index = 1
    root_list = roots[methods[index]]
    main(root_list, seeds=seeds)
