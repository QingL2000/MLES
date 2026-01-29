import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation
from llm4ad.task.machine_learning.moon_lander import template_program
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
                   key='sk-0hCjhh3wBUP7H2TQF9B6D290Ee604cAc88633dDc5f68B0Ed',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)
    log_dir = f'Withinitpolicy/MLES/withinit_{run_id}'  # Use run_id to avoid overwriting logs
    # batch 表示不是跑全的，而是从某一代就开始跑的
    task = MoonLanderEvaluation(whocall='mmeoh')


    method = MMEoH(llm=llm,
                   profiler=EoHProfiler(log_dir=log_dir, log_style='complex'),
                   evaluation=task,
                   max_sample_nums=2000,
                   max_generations=None,
                   pop_size=16,
                   num_samplers=1,
                   num_evaluators=1,
                   debug_mode=False,
                   operators=('e1', 'e2', 'm1_M', 'm2_M'), # ('e1', 'e2', 'm1_M', 'm2_M')
                   seed_path=seedpath,
                   multi_thread_or_process_eval='process'
                   )

    # 检查文件是否存在
    if os.path.exists(seedpath):
        # 打开并读取JSON文件
        with open(seedpath, 'r', encoding='utf-8') as file:
            seeds = json.load(file)
    else:
        print(f"文件 {seedpath} 不存在")

    # 取出 programs database
    prog_db = method._population
    profiler = method._profiler

    # 对每个 seed function: 1) 加入种群 2) 记录到 profiler
    for seed_individual in seeds:
        seed_str = seed_individual['function']
        seed = tfpc.function_to_program(seed_str, template_program)

        # 用 funsearch 中的 evaluator 评估 seed
        score_images_dict, eval_time = method._evaluator.evaluate_program_record_time(program=seed)

        # 讲 seed 转化成 function 实例
        seed = tfpc.text_to_function(seed_str)
        # tfpc.function_to_program()
        # tfpc.text_to_program()

        if score_images_dict is not None:
            # register to profiler
            seed.score = score_images_dict['score']
            seed.image64 = score_images_dict['image']
            seed.observation = score_images_dict['observation']
        else:
            seed.score = None
        # 讲 seed 转化成 function 实例
        seed.evaluate_time = eval_time
        seed.algorithm = seed_individual['algorithm']
        # 向 prog_db 中加入 seed
        prog_db.register_function(seed)
        # 向 profiler 中加入 seed
        profiler.register_function(seed)

    method.run()



if __name__ == '__main__':
    # for i in range(1):
    #     main(i)
    main(10)
