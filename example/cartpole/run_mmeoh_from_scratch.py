import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.Cartpole import CartPole
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
                   model='gpt-5-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)
    log_dir = f'fromscrate/MELS/4omini_{run_id}'  # Use run_id to avoid overwriting logs
    # batch 表示不是跑全的，而是从某一代就开始跑的, All 表示是从单个算法开始，运行1000个采样获取的。
    task = CartPole(whocall='mmeoh')

    method = MMEoH(llm=llm,
                   profiler=EoHProfiler(log_dir=log_dir, log_style='complex'),
                   evaluation=task,
                   max_sample_nums=500,
                   max_generations=None,
                   pop_size=16,
                   num_samplers=4,
                   num_evaluators=4,
                   debug_mode=False,
                   # operators=('e1', 'e2', 'm1_M', 'm2_M')
                   operators=('e1', 'e2', 'm1_M', 'm2_M'), # ('e1', 'e2', 'm1_M', 'm2_M'),
                   multi_thread_or_process_eval='process'
                   )

    method.run()



if __name__ == '__main__':
    # for i in range(1):
    #     main(i)
    main(0)
