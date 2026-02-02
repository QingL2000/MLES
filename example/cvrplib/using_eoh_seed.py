import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.cvrp_construct_set import CVRPSEvaluation
from llm4ad.task.optimization.cvrp_construct_set import template_program
# from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_https_mmeoh import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.mmeoh import MMEoH
from llm4ad.method.mmeoh import EoHProfiler


import json
from llm4ad.base import TextFunctionProgramConverter as tfpc
import os

def main(run_id, using_algo_designed_path=None):
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-GYO9mUQ6DelPSAHv6APqNEq4muMWyf12c3R41rRwaEm4VdXs',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-5-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)
    log_dir = f'log/Eoh/{run_id}'  # Use run_id to avoid overwriting logs

    run_mode='Using'

    balanced_training_set = [
        './balanced_trainingset/cvrp_clustered_cap30-60_n30-60_num20_train.pkl',
        # './balanced_trainingset/cvrp_clustered_cap30-60_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_clustered_cap30-60_n180-200_num20_train.pkl',
        # './balanced_trainingset/cvrp_clustered_cap90-110_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_heavy_cap30-60_n30-60_num20_train.pkl',
        # './balanced_trainingset/cvrp_heavy_cap30-60_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_heavy_cap30-60_n180-200_num20_train.pkl',
        # './balanced_trainingset/cvrp_heavy_cap90-110_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_uniform_cap30-60_n30-60_num20_train.pkl',
        # './balanced_trainingset/cvrp_uniform_cap30-60_n80-120_num20_train.pkl',
        # './balanced_trainingset/cvrp_uniform_cap30-60_n180-200_num20_train.pkl',
        # './balanced_trainingset/cvrp_uniform_cap90-110_n80-120_num20_train.pkl',
    ]

    testing_set = [
        './testing_set/cvrp_clustered_cap30-60_n30-60_num10_test.pkl',
        # './testing_set/cvrp_clustered_cap30-60_n80-120_num10_test.pkl',
        # './testing_set/cvrp_clustered_cap30-60_n180-200_num10_test.pkl',
        # './testing_set/cvrp_clustered_cap90-110_n80-120_num10_test.pkl',
        # './testing_set/cvrp_heavy_cap30-60_n30-60_num10_test.pkl',
        # './testing_set/cvrp_heavy_cap30-60_n80-120_num10_test.pkl',
        # './testing_set/cvrp_heavy_cap30-60_n180-200_num10_test.pkl',
        # './testing_set/cvrp_heavy_cap90-110_n80-120_num10_test.pkl',
        # './testing_set/cvrp_uniform_cap30-60_n30-60_num10_test.pkl',
        # './testing_set/cvrp_uniform_cap30-60_n80-120_num10_test.pkl',
        # './testing_set/cvrp_uniform_cap30-60_n180-200_num10_test.pkl',
        # './testing_set/cvrp_uniform_cap90-110_n80-120_num10_test.pkl',
    ]

    task = CVRPSEvaluation(
        timeout_seconds=120,
        run_mode=run_mode,
        training_datasets=balanced_training_set,
        testing_datasets=balanced_training_set,
        whocall='mmeoh',
        objective_value=0
    )

    seedpath = r''

    operators_setting = ('e1', 'e2', 'm1', 'm2')

    method = MMEoH(llm=llm,
                   profiler=EoHProfiler(log_dir='logs/eoh',
                                        log_style='complex',
                                        run_mode=run_mode,
                                        using_algo_designed_path=using_algo_designed_path),

                   evaluation=task,
                   max_sample_nums=500,
                   max_generations=None,
                   pop_size=16,
                   num_samplers=8,
                   num_evaluators=8,
                   debug_mode=False,
                   operators=operators_setting, # ('e1', 'e2', 'm1_M', 'm2_M')
                   seed_path=seedpath,
                   multi_thread_or_process_eval='process'
                   )

    method.run(mode='Using')

if __name__ == '__main__':
    Eohs = [
            r"C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log_test\Eoh\4\20260202_152723_Problem_EoH",
        ]

    MEohs = [
        r"C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log_test\MEoh\4\20260202_161717_Problem_EoH",
    ]

    for test_path in Eohs:
        main(0, using_algo_designed_path=test_path)

