import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.Cartpole import CartPole
from llm4ad.task.machine_learning.Cartpole import template_program
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
                   key='sk-GYO9mUQ6DelPSAHv6APqNEq4muMWyf12c3R41rRwaEm4VdXs',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)

    # log_dir = f'All/MMeoh/4omini_{run_id}'  # Use run_id to avoid overwriting logs
    log_dir = f'fromone/MELS/5mini_{run_id}'  # Use run_id to avoid overwriting logs
    # batch 表示不是跑全的，而是从某一代就开始跑的, All 表示是从单个算法开始，运行1000个采样获取的。
    task = CartPole(whocall='mmeoh')

    method = MMEoH(llm=llm,
                   profiler=EoHProfiler(log_dir=log_dir, log_style='complex'),
                   evaluation=task,
                   max_sample_nums=10,
                   max_generations=None,
                   pop_size=16,
                   num_samplers=8,
                   num_evaluators=8,
                   debug_mode=False,
                   # operators=('e1', 'e2', 'm1_M', 'm2_M')
                   operators=('e1', 'e2', 'm1_M', 'm2_M') # ('e1', 'e2', 'm1_M', 'm2_M')
                   )

    # 取出 programs database
    prog_db = method._population
    profiler = method._profiler

    seed = tfpc.function_to_program(template_program, template_program)

    # 用 funsearch 中的 evaluator 评估 seed
    score_images_dict, eval_time = method._evaluator.evaluate_program_record_time(program=seed)

    # 讲 seed 转化成 function 实例
    seed = tfpc.text_to_function(template_program)
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
    seed.algorithm = "This is a classic PD (Proportional-Derivative) control algorithm applied to the Gym CartPole environment. The algorithm calculates a weighted control signal based on the pole's current angle (the P-term) and angular velocity (the D-term), and selects the appropriate action (pushing the cart left or right) to correct the pole's posture, thereby achieving the goal of keeping the pole balanced upright."
    # 向 prog_db 中加入 seed
    prog_db.register_function(seed)
    # 向 profiler 中加入 seed
    profiler.register_function(seed)

    method.run()



if __name__ == '__main__':
    for i in range(1):
        main(i)
