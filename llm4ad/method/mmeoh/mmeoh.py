from __future__ import annotations

import concurrent.futures
import time
import traceback
from threading import Thread
from typing import Optional, Literal, List

from .population import Population
from .profiler import EoHProfiler
from .prompt import MMEoHPrompt
from .sampler import MMEoHSampler
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase
import itertools
import os
import re
import json
from collections import Counter # 引入计数器方便统计

class MMEoH:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: Optional[int] = 5,
                 selection_num=2,
                 # use_e2_operator: bool = True,
                 # use_m1_operator: bool = False,
                 # use_m2_operator: bool = False,
                 # use_m1_Multimodel_operator: bool = True,
                 # use_m2_Multimodel_operator: bool = True,
                 # use_e2_Multimodal_operator: bool = False,
                 operators: tuple = ('e1', 'e2', 'm1', 'm2'),
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 resume_mode: bool = False,
                 initial_sample_nums_max: int = 50,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 seed_path="",
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. 
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            initial_sample_nums_max     : maximum samples restriction during initialization.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self.evaluation_object = evaluation
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        if hasattr(evaluation, 'non_image_representation_explanation'):
            self._information_discription = evaluation.non_image_representation_explanation
        else:
            self._information_discription = ""  

        if 'm1_text_info' in operators and not self._information_discription:  
            raise ValueError(
                "When 'text' is in operators, non image information description of this task cannot be empty")

        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        # self._use_e2_operator = use_e2_operator
        # self._use_m1_operator = use_m1_operator
        # self._use_m2_operator = use_m2_operator
        # self._use_m1_Multimodel_operator = use_m1_Multimodel_operator
        # self._use_m2_Multimodel_operator = use_m2_Multimodel_operator
        # self._use_e2_Multimodal_operator = use_e2_Multimodal_operator
        self.operators = operators

        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._initial_sample_nums_max = initial_sample_nums_max
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # adjust population size
        self._adjust_pop_size()

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._pop_size)
        self._sampler = MMEoHSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = max(
            self._initial_sample_nums_max,
            2 * self._pop_size
        )

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

    def _adjust_pop_size(self):
        # adjust population size
        if self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
            elif abs(self._pop_size - 40) > 20:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
            elif abs(self._pop_size - 20) > 10:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
            elif abs(self._pop_size - 10) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._pop_size is None:
                self._pop_size = 5
            elif abs(self._pop_size - 5) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _sample_evaluate_register(self, prompt, image_prompt=None, massage=None, operator_name="", parent_number=None):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        thought, func, response = self._sampler.get_thought_and_function(prompt, image_prompt, massage)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return
        # evaluate
        # score_images_dict
        score_images_dict, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()

        if score_images_dict is not None:
            # register to profiler
            func.score = score_images_dict['score']
            func.image64 = score_images_dict['image']
            func.observation = score_images_dict['observation']
        else:
            func.score = None
        if parent_number is not None:
            func.parents = parent_number
        func.operator = operator_name
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        func.response = response
        func.prompt = prompt

        # register to the population
        self._population.register_function(func)

        if self._profiler is not None:
            self._profiler.register_function(func, operation_name=operator_name)
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)
            self._tot_sample_nums += 1

    def _continue_loop(self) -> bool:
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._population.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (self._population.generation < self._max_generations
                    and self._tot_sample_nums < self._max_sample_nums)

    def _iteratively_use_mmeoh_operator(self, tid=0):
        operator_cycle = itertools.cycle(self.operators)
        for _ in range(tid):
            operator = next(operator_cycle)

        while self._continue_loop():
            try:
                operator = next(operator_cycle)

                if operator == 'e1_advanced':
                    # get a new func using e1
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    massage = MMEoHPrompt.get_prompt_e1_advanced(self._task_description_str, indivs,
                                                                 self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E1 Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", massage=massage, operator_name='e1_advanced',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'e1':
                    # get a new func using e1
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    prompt = MMEoHPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E1 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt=prompt, operator_name='e1',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using e2
                elif operator == 'e2':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    prompt = MMEoHPrompt.get_prompt_e2(self._task_description_str, indivs,
                                                       self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2 Prompt: {prompt}')
                    self._sample_evaluate_register(prompt=prompt, operator_name='e2',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using e2
                elif operator == 'e2_advanced':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    massage = MMEoHPrompt.get_prompt_e2_advanced(self._task_description_str, indivs,
                                                                 self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2_advanced Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", massage=massage, operator_name='e2',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using e2 Multimodal
                elif operator == 'e2_M':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    massage = MMEoHPrompt.get_prompt_e2_M(self._task_description_str, indivs,
                                                          self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2 Multimodal Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage, operator_name='e2_M',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using m1
                elif operator == 'm1':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_m1(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1 Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", operator_name='m1', massage=massage,
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using m2
                elif operator == 'm2':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_m2(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M2 Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", operator_name='m2', massage=massage,
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm1_M':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_m1_M(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1_Multimodel Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage, operator_name='m1_M',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm1_text':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_m1_M_text_info(self._task_description_str, indiv,
                                                                    self._function_to_evolve,
                                                                    self._information_discription)
                    if self._debug_mode:
                        print(f'm1_text Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage,
                                                   operator_name='m1_text_info',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm2_M':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_m2_M(self._task_description_str, indiv, self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M2_Multimodel Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage, operator_name='m2_M',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm1_only_imagedescribtion':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_image_description(self._task_description_str, indiv,
                                                                       self._function_to_evolve)
                    description, response = self._sampler.get_image_description(prompt="", image64s=None,
                                                                                massage=massage)
                    massage = MMEoHPrompt.get_prompt_m1_M_image_description(self._task_description_str, indiv,
                                                                            self._function_to_evolve, description)
                    if self._debug_mode:
                        print('Description:', description)
                        print('Description response:', response)
                        print(f'm1_image_describtion Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage,
                                                   operator_name='m1_image_describtion',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm2_only_imagedescribtion':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_image_description(self._task_description_str, indiv,
                                                                       self._function_to_evolve)
                    description, response = self._sampler.get_image_description(prompt="", image64s=None,
                                                                                massage=massage)
                    massage = MMEoHPrompt.get_prompt_m2_M_image_description(self._task_description_str, indiv,
                                                                            self._function_to_evolve, description)
                    if self._debug_mode:
                        print('Description:', description)
                        print('Description response:', response)
                        print(f'm2_image_describtion Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage,
                                                   operator_name='m2_image_describtion',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm1_only_image':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_m1_M_only_image(self._task_description_str, indiv,
                                                                     self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1_only_image_Multimodel Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage,
                                                   operator_name='m1_only_image',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'e1_nothought':
                    # get a new func using e1
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    prompt = MMEoHPrompt.get_prompt_e1_nothought(self._task_description_str, indivs,
                                                                 self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E1_nothought Prompt: {prompt}')
                    self._sample_evaluate_register(prompt=prompt, operator_name='e1_nothought',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                # get a new func using e2
                elif operator == 'e2_nothought':
                    indivs = self._population.selection(number=self._selection_num)
                    parents_pop_register_number = [ind.pop_register_number for ind in indivs]
                    prompt = MMEoHPrompt.get_prompt_e2_nothought(self._task_description_str, indivs,
                                                                 self._function_to_evolve)
                    if self._debug_mode:
                        print(f'E2_nothought Prompt: {prompt}')
                    self._sample_evaluate_register(prompt=prompt, operator_name='e2_nothought',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm1_M_nothought':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_m1_M_nothought(self._task_description_str, indiv,
                                                                    self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M1_Multimodel_nothought Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage, operator_name='m1_M_nothought',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                elif operator == 'm2_M_nothought':
                    indivs = self._population.selection()
                    indiv = indivs[0]
                    parents_pop_register_number = [indiv.pop_register_number]
                    massage = MMEoHPrompt.get_prompt_m2_M_nothought(self._task_description_str, indiv,
                                                                    self._function_to_evolve)
                    if self._debug_mode:
                        print(f'M2_Multimodel_nothought Prompt: {self.messages_to_string(massage)}')
                    self._sample_evaluate_register(prompt="", image_prompt=None, massage=massage, operator_name='m2_M_nothought',
                                                   parent_number=parents_pop_register_number)
                    if not self._continue_loop():
                        break

                else:
                    raise Exception("ERROR: The input operators are not supported at the moment. Please check !!!!!")

            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    # exit()
                continue

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def _iteratively_init_population(self, tid=0):
        """Let a thread repeat {sample -> evaluate -> register to population}
        to initialize a population.
        """
        while self._population.generation == 0:
            try:
                # get a new func using i1
                prompt = MMEoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                if self._debug_mode:
                    print('Init Prompt: ', prompt)
                self._sample_evaluate_register(prompt, operator_name="Initialization")
                if self._tot_sample_nums > self._initial_sample_nums_max:
                    print(f'Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading.
        In EoH, `fn` can be `self._iteratively_init_population` or `self._iteratively_use_eoh_operator`.
        """
        # threads for sampling
        sampler_threads = [
            Thread(target=fn, args=(tid, *args), kwargs=kwargs)
            for tid in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self, mode: Literal['Training', 'Using', 'Combined'] = 'Training', **kwargs):
        if mode == 'Training':
            print("🚀 Starting Training Mode...")
            self._training_flow()
        elif mode == 'Using':
            print("💡 Starting Using Mode...")
            # generalist_protection = kwargs.get('generalist_protection', True)
            # matching_feature = kwargs.get('matching_feature', 'anchor')
            leader_top_k = kwargs.get('leader_top_k', 1)
            worst_case_percent = kwargs.get('worst_case_percent', 10)
            print(f'Using algorithms from 🚀 {self._profiler._log_dir} 🚀 to solve new instances.')

            self._using_flow(top_k=leader_top_k,
                             worst_case_percent=worst_case_percent)

    def _training_flow(self):
        if not self._resume_mode:
            # do initialization
            self._multi_threaded_sampling(self._iteratively_init_population)
            # terminate searching if
            if len(self._population) < self._selection_num:
                print(
                    f'The search is terminated since EoH unable to obtain {self._selection_num} feasible algorithms during initialization. '
                    f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                    f'Please also check your evaluation implementation and LLM implementation.')
                return

        # evolutionary search
        self._multi_threaded_sampling(self._iteratively_use_mmeoh_operator)
        # finish
        if self._profiler is not None:
            self._profiler.finish()

    # 修改函数签名，增加 top_k 参数 (默认为 None，表示全部运行)
    def _using_flow(self, worst_case_percent=10, top_k=None, max_gen=None):
        """
        Args:
            worst_case_percent: 统计最后百分之几的情况
            top_k: 从读取的文件中筛选前 K 个算法
            max_gen: 限制最高代数。如果为 None，则读取目录下最大的代数。
        """
        print(f"🔍 Loading model from {self._profiler._log_dir}...")
        designed_results_path = os.path.join(self._profiler._log_dir, 'population')

        pattern = re.compile(r'^pop_(\d+)\.json$')

        max_x = -1
        latest_file = None

        # 2. Check if the directory exists
        if not os.path.isdir(designed_results_path):
            print(f"Error: Directory not found: {designed_results_path}")
            return  # Or raise an Exception

        # --- 修改后的文件搜索逻辑 ---
        for filename in os.listdir(designed_results_path):
            match = pattern.match(filename)
            if match:
                current_x = int(match.group(1))

                # 如果指定了 max_gen，则忽略超过该代数的文件
                if max_gen is not None and current_x > max_gen:
                    continue

                # 在符合条件的范围内寻找最大的代数
                if current_x > max_x:
                    max_x = current_x
                    latest_file = filename
        # --------------------------

        # 6. Check if any matching file was found
        if latest_file is None:
            limit_msg = f" within max_gen={max_gen}" if max_gen else ""
            print(f"Error: No valid 'pop_x.json' files found{limit_msg} in {designed_results_path}")
            return

        # 7. Construct the full path to the correct file
        full_path_to_file = os.path.join(designed_results_path, latest_file)
        print(f"Found latest file: {full_path_to_file}")

        # --- End of file finding logic ---

        with open(full_path_to_file, 'r') as f:
            trained_data = json.load(f)

        # ===================================================================
        # [新增逻辑] Top-K 筛选
        # ===================================================================
        if top_k is not None and isinstance(top_k, int) and top_k > 0:
            print(f"✂️  Filtering Population: Selecting top {top_k} algorithms...")
            original_size = len(trained_data)

            # 假设 json 中的每个 item 都有 'fitness' 字段。
            # 如果是最大化问题 (score越高越好)，使用 reverse=True。
            # 这里使用了 get('fitness', -inf) 防止字段缺失报错。
            try:
                trained_data.sort(key=lambda x: x.get('score', float('-inf')), reverse=True)

                # 截取前 k 个
                trained_data = trained_data[:top_k]
                print(f"   -> Reduced population from {original_size} to {len(trained_data)}.")
            except Exception as e:
                print(f"   -> ⚠️ Warning: Could not sort by 'score'. Using original order. Error: {e}")
        # ===================================================================

        using_time_start = time.time()
        print("💪 [Brute Force Mode] Evaluating selected algorithms on each instance...")

        print(f"   -> Found {len(trained_data)} unique algorithms to test.")
        ins_to_be_solve_set = self.evaluation_object.ins_to_be_solve_set
        ins_to_be_solve_id_set = [id for id in ins_to_be_solve_set.keys()]

        final_results = {}
        all_scores = []

        # --- b. 遍历每个实例，并用所有算法进行测试 ---
        for instance_id in ins_to_be_solve_id_set:
            print(f"\n[Brute Force] Solving new instance: {instance_id}")
            best_algo_for_instance = None
            best_score_for_instance = float('-inf')
            best_perf_for_instance = None

            for i, algo_json in enumerate(trained_data):
                # 显示进度
                print(f"  -> Testing algorithm {i + 1}/{len(trained_data)}...", end='\r')
                try:
                    program = TextFunctionProgramConverter.function_to_program(algo_json['function'],
                                                                               self._template_program)
                    func = TextFunctionProgramConverter.text_to_function(str(program))
                    eval_result = self._evaluator._evaluate(str(program), func.name,
                                                                              ins_to_be_evaluated_id=(instance_id,),
                                                                              training_mode=False)

                    score = eval_result.get('all_ins_performance').get(instance_id, {}).get('score', float('-inf'))
                    # print(f'{i} score is ', score) # 可选：为了日志干净可以注释掉详细打印

                    if score is not None and score > best_score_for_instance:
                        print(f'   Update! New Best: {score:.4f} (Algo index: {i})')
                        best_score_for_instance = score
                        best_algo_for_instance = algo_json
                        best_perf_for_instance = eval_result.get('all_ins_performance').get(instance_id, {})
                except Exception as e:
                    print(f"\n      -> ❌ Error evaluating algorithm on instance {instance_id}: {e}")
            print()  # 换行

            if best_algo_for_instance:
                print(
                    f"   -> ✅ Best score found: {best_score_for_instance:.4f}")
                final_results[instance_id] = {
                    'algorithm': best_algo_for_instance['algorithm'],
                    'function': best_algo_for_instance['function'],
                    'score': best_perf_for_instance.get('score'),
                }
                if best_perf_for_instance.get('score') is not None:
                    all_scores.append(best_perf_for_instance['score'])
            else:
                final_results[instance_id] = {'score': None, 'evaluate_time': None}
                print(f"   -> ⚠️ Warning: No algorithm produced a valid score for instance {instance_id}.")

        # ===================================================================
        # FINALIZE: 计算最终统计数据并保存
        # ===================================================================
        valid_scores = [s for s in all_scores if s is not None]

        if valid_scores:
            final_results['sum_score_of_all_instances'] = sum(valid_scores)
            final_results['average_score_of_all_instances'] = sum(valid_scores) / len(valid_scores)
        else:
            final_results['sum_score_of_all_instances'] = None
            final_results['average_score_of_all_instances'] = None

        final_results['each_result'] = all_scores

        # ... (此处省略 Worst-Case 统计代码，保持你原有的逻辑不变) ...
        # 为了完整性，你可以直接把你的 Worst-Case 代码块放在这里
        # ===================================================================
        # [统计 Worst-Case (Bottom K%)]
        # ===================================================================
        id_score_pairs = []
        for k, v in final_results.items():
            if isinstance(k, int) and isinstance(v, dict) and v.get('score') is not None:
                id_score_pairs.append((k, v['score']))
        id_score_pairs.sort(key=lambda x: x[1])
        total_valid_count = len(id_score_pairs)
        cutoff_count = int(total_valid_count * (worst_case_percent / 100.0))
        if cutoff_count == 0 and total_valid_count > 0:
            cutoff_count = 1
        worst_cases = id_score_pairs[:cutoff_count]
        worst_instance_ids = [pair[0] for pair in worst_cases]
        worst_scores_values = [pair[1] for pair in worst_cases]
        worst_avg_score = sum(worst_scores_values) / len(worst_scores_values) if worst_scores_values else None

        if worst_avg_score is not None:  # 加个判断防止打印 None
            print(f"\n📉 [Worst-Case Stats] Bottom {worst_case_percent}% (Count: {len(worst_cases)}):")
            print(f"   -> Average Score: {worst_avg_score}")

        final_results['worst_case_stats'] = {
            'percent_threshold': worst_case_percent,
            'count': len(worst_cases),
            'average_score': worst_avg_score,
            'instance_ids': worst_instance_ids,
            'scores': worst_scores_values
        }
        # ===================================================================

        using_time_end = time.time()
        final_results['running_time'] = using_time_end - using_time_start
        print(f"Running time: {final_results['running_time']} seconds")

        if self._profiler:
            self._profiler.using_final(final_results=final_results)
        print(f"\n💡 Using Mode finished.")

        print(
            f'There are {len(ins_to_be_solve_set)} instances to solve. \nSuccessfully solved {len(valid_scores)} instances, with an average score of {final_results["average_score_of_all_instances"]}.')

    # 修改函数签名，增加 top_k 参数 (默认为 None，表示全部运行)
    def _Multi_using_flow(self, designed_results_paths: List[str], top_k=1, worst_case_percent=10,
                          cvrplib_which='A', who='DontKnow', max_gen=None):  # <--- 1. 增加参数
        """
        Args:
            designed_results_paths (List[str]): 包含多个实验日志根目录的列表。
            top_k (int, optional): 从每个路径中筛选 Top K 个算法。
            worst_case_percent (int): 统计最差百分比。
        """

        all_candidates = []  # 用于汇总所有路径下筛选出的 candidates

        print(f"🚀 Starting Multi-Path Evaluation. Paths to check: {len(designed_results_paths)}")

        # ===================================================================
        # PART 1: 遍历所有路径，收集 Candidates 并打上来源标签
        # ===================================================================
        for path_idx, base_path in enumerate(designed_results_paths):
            print(f"\n📂 [{path_idx + 1}/{len(designed_results_paths)}] Processing path: {base_path}")

            population_dir = os.path.join(base_path, 'population')

            if not os.path.isdir(population_dir):
                print(f"   ⚠️  Warning: Directory not found, skipping: {population_dir}")
                continue

            pattern = re.compile(r'^pop_(\d+)\.json$')
            max_x = -1
            latest_file = None

            # --- 2. 这里的循环逻辑加入 max_gen 限制 ---
            for filename in os.listdir(population_dir):
                match = pattern.match(filename)
                if match:
                    current_x = int(match.group(1))

                    # 如果超过了限制的代数，直接跳过
                    if max_gen is not None and current_x > max_gen:
                        continue

                    # 寻找限制范围内的最大代数
                    if current_x > max_x:
                        max_x = current_x
                        latest_file = filename

            if latest_file is None:
                print(f"   ⚠️  Warning: No 'pop_x.json' files found in {population_dir}")
                continue

            full_path_to_file = os.path.join(population_dir, latest_file)
            print(f"   -> Found latest file: {latest_file}")

            try:
                with open(full_path_to_file, 'r') as f:
                    current_data = json.load(f)
            except Exception as e:
                print(f"   ❌ Error loading JSON: {e}")
                continue

            # --- Top-K 筛选 ---
            if top_k is not None and isinstance(top_k, int) and top_k > 0:
                original_size = len(current_data)
                try:
                    current_data.sort(key=lambda x: x.get('score', float('-inf')), reverse=True)
                    selected_data = current_data[:top_k]
                    print(f"   ✂️  Top-{top_k} Filter: Selected {len(selected_data)}/{original_size} algorithms.")
                    current_data = selected_data
                except Exception as e:
                    print(f"   ⚠️  Sort failed, using original order: {e}")
            else:
                print(f"   -> Keeping all {len(current_data)} algorithms.")

            # --- [关键修改] 给每个算法打上来源标签 ---
            for algo in current_data:
                # 记录完整路径，或者你也可以只记录文件夹名 os.path.basename(base_path)
                algo['source_path'] = base_path

            all_candidates.extend(current_data)

        # ===================================================================
        # PART 2: 统一进行评估 (Evaluation)
        # ===================================================================

        candidates_count = len(all_candidates)
        if candidates_count == 0:
            print("\n❌ Error: No valid algorithms found.")
            return

        print(f"\n✅ Collection Finished. Total algorithms to evaluate: {candidates_count}")

        using_time_start = time.time()
        ins_to_be_solve_set = self.evaluation_object.ins_to_be_solve_set
        ins_to_be_solve_id_set = [id for id in ins_to_be_solve_set.keys()]

        final_results = {}
        all_scores = []

        for instance_id in ins_to_be_solve_id_set:
            print(f"\n[Brute Force] Solving new instance: {instance_id}")
            best_algo_for_instance = None
            best_score_for_instance = float('-inf')
            best_perf_for_instance = None

            for i, algo_json in enumerate(all_candidates):
                print(f"  -> Testing algorithm {i + 1}/{candidates_count}...", end='\r')
                try:
                    program = TextFunctionProgramConverter.function_to_program(algo_json['function'],
                                                                               self._template_program)
                    func = TextFunctionProgramConverter.text_to_function(str(program))

                    eval_result = self._evaluator._evaluate(str(program), func.name,
                                                            ins_to_be_evaluated_id=(instance_id,),
                                                            training_mode=False)

                    score = eval_result.get('all_ins_performance').get(instance_id, {}).get('score', float('-inf'))

                    if score is not None and score > best_score_for_instance:
                        # 获取该算法的来源路径，仅用于打印日志
                        src = algo_json.get('source_path', 'unknown')
                        print(f'   Update! New Best: {score:.4f} (Algo index: {i}, Src: {src})')

                        best_score_for_instance = score
                        best_algo_for_instance = algo_json
                        best_perf_for_instance = eval_result.get('all_ins_performance').get(instance_id, {})
                except Exception as e:
                    pass

            print()

            if best_algo_for_instance:
                print(
                    f"   -> ✅ Best: {best_score_for_instance:.4f} | Winner: {best_algo_for_instance.get('source_path')}")

                # --- [关键修改] 将来源路径写入最终结果 ---
                final_results[instance_id] = {
                    'algorithm': best_algo_for_instance['algorithm'],
                    'function': best_algo_for_instance['function'],
                    'score': best_perf_for_instance.get('score'),
                    'source_path': best_algo_for_instance.get('source_path')  # 这里记录胜出的路径
                }
                if best_perf_for_instance.get('score') is not None:
                    all_scores.append(best_perf_for_instance['score'])
            else:
                final_results[instance_id] = {'score': None, 'evaluate_time': None, 'source_path': None}

        # ===================================================================
        # PART 3: 统计结果
        # ===================================================================
        valid_scores = [s for s in all_scores if s is not None]

        if valid_scores:
            final_results['sum_score_of_all_instances'] = sum(valid_scores)
            final_results['average_score_of_all_instances'] = sum(valid_scores) / len(valid_scores)
        else:
            final_results['sum_score_of_all_instances'] = None
            final_results['average_score_of_all_instances'] = None

        final_results['each_result'] = all_scores

        # --- Worst-Case Statistics ---
        id_score_pairs = []
        for k, v in final_results.items():
            if isinstance(k, int) and isinstance(v, dict) and v.get('score') is not None:
                id_score_pairs.append((k, v['score']))

        id_score_pairs.sort(key=lambda x: x[1])
        total_valid_count = len(id_score_pairs)
        cutoff_count = int(total_valid_count * (worst_case_percent / 100.0))
        if cutoff_count == 0 and total_valid_count > 0:
            cutoff_count = 1

        worst_cases = id_score_pairs[:cutoff_count]
        worst_instance_ids = [pair[0] for pair in worst_cases]
        worst_scores_values = [pair[1] for pair in worst_cases]
        worst_avg_score = sum(worst_scores_values) / len(worst_scores_values) if worst_scores_values else None

        final_results['worst_case_stats'] = {
            'percent_threshold': worst_case_percent,
            'count': len(worst_cases),
            'average_score': worst_avg_score,
            'instance_ids': worst_instance_ids,
            'scores': worst_scores_values
        }

        using_time_end = time.time()
        final_results['running_time'] = using_time_end - using_time_start

        # ===================================================================
        # PART 4: [新增] 路径胜率统计 (Source Path Distribution)
        # ===================================================================
        # 统计每个路径赢了多少次
        winner_paths = []
        for k, v in final_results.items():
            # 确保 key 是 instance ID (int) 且有 source_path
            if isinstance(k, int) and isinstance(v, dict) and v.get('source_path'):
                winner_paths.append(v['source_path'])

        path_counts = dict(Counter(winner_paths))

        print("\n🏆 [Dominance Statistics] Who contributed the best solutions?")
        for path, count in path_counts.items():
            print(f"   -> {path}: {count} instances")

        final_results['path_dominance_stats'] = path_counts
        # ===================================================================

        cvrplib_which_path = os.path.join('cvrplib_results', cvrplib_which)
        os.makedirs(cvrplib_which_path, exist_ok=True)
        output_file_path = os.path.join(cvrplib_which_path, f'using_final_output_{who}.json')

        with open(output_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=4)

        print(f"\n💡 Using Mode finished.")
        print(
            f'Solved {len(valid_scores)} instances. Avg Score: {final_results["average_score_of_all_instances"]}. Saved to {output_file_path}')


    def messages_to_string(self, messages, image_placeholder="<<<IMAGE>>>"):
        """
        Convert a structured messages list (OpenAI-style) into a single formatted string.
        Supports both 'text' and 'image_url' content types.

        :param messages: list of dicts with 'role' and 'content'
        :param image_placeholder: str or callable, placeholder inserted for images
        :return: str
        """
        output_lines = []
        for message in messages:
            role = message.get("role", "user")
            contents = message.get("content", [])

            output_lines.append(f"[{role.upper()}]")
            for item in contents:
                if item.get("type") == "text":
                    text = item.get("text", "").strip()
                    if text:
                        output_lines.append(text)
                elif item.get("type") == "image_url":
                    # Optional: handle custom placeholders with description
                    url = item.get("image_url", {}).get("url", "")
                    desc = item.get("image_url", {}).get("detail", "an image")
                    if callable(image_placeholder):
                        placeholder = image_placeholder(url, desc)
                    else:
                        placeholder = f"{image_placeholder}  # {desc}"
                    output_lines.append(placeholder)
            output_lines.append("")  # blank line between messages

        return "\n".join(output_lines)
