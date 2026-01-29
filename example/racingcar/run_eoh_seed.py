import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.car_raceing_continue import RacingCarEvaluation
from llm4ad.task.machine_learning.car_raceing_continue import template_program
# from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_https_mmeoh import HttpsApi
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.method.mmeoh import MMEoH
from llm4ad.method.mmeoh import EoHProfiler


import json
from llm4ad.base import TextFunctionProgramConverter as tfpc
import os

def main(run_id):
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-XXXX',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)
    log_dir = f'batch/Eoh/{run_id}'  # Use run_id to avoid overwriting logs
    task = RacingCarEvaluation(whocall='mmeoh')

    seedpath = r'init_pop_size16.json'

    method = MMEoH(llm=llm,
                   profiler=EoHProfiler(log_dir=log_dir, log_style='complex'),
                   evaluation=task,
                   max_sample_nums=2000,
                   max_generations=None,
                   pop_size=8,
                   num_samplers=8,
                   num_evaluators=8,
                   debug_mode=False,
                   operators=('e1', 'e2', 'm1', 'm2'), # ('e1', 'e2', 'm1_M', 'm2_M')
                   seed_path=seedpath
                   )

    if os.path.exists(seedpath):
        with open(seedpath, 'r', encoding='utf-8') as file:
            seeds = json.load(file)
    else:
        print(f"File {seedpath} can't found")

    prog_db = method._population
    profiler = method._profiler

    for seed_individual in seeds:
        seed_str = seed_individual['function']
        seed = tfpc.function_to_program(seed_str, template_program)
        score_images_dict, eval_time = method._evaluator.evaluate_program_record_time(program=seed)

        seed = tfpc.text_to_function(seed_str)

        if score_images_dict is not None:
            # register to profiler
            seed.score = score_images_dict['score']
            seed.image64 = score_images_dict['image']
            seed.observation = score_images_dict['observation']
        else:
            seed.score = None
        seed.evaluate_time = eval_time
        seed.algorithm = seed_individual['algorithm']
        prog_db.register_function(seed)
        profiler.register_function(seed)

    method.run()



if __name__ == '__main__':
    for i in range(1):
        main(i)
