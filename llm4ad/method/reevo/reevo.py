# Module Name: ReEvo
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import concurrent.futures
import time
import traceback
from threading import Thread
from typing import Optional, Literal, List

from torch.utils.data import Sampler

from .population import Population
from .profiler import ReEvoProfiler
from .prompt import ReEvoPrompt
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator, SampleTrimmer
)
from ...tools.profiler import ProfilerBase

from threading import Lock
import os
import re
import json
from collections import Counter # 引入计数器方便统计

class ReEvo:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: Optional[int] = 20,
                 mutation_rate: float = 0.5,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Reflective Evolution.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'llm4ad.method.reevo.ReEvoProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations or reach 'max_sample_nums',
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs        : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """

        self.evaluation_object = evaluation

        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._mutation_rate = mutation_rate



        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval
        self._MAX_SHORT_TERM_REFLECTION_PROMPT = 5

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._pop_size)
        self._sampler = SampleTrimmer(llm)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        # statistics
        self._tot_sample_nums = 0

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

        self._lock = Lock()
        self._global_crx_count = 0
        # 将 Prompt 记录提升为全局变量，这样线程 A 生成的思考，线程 B 也能看见
        self._short_term_reflection_prompts = []
        self._long_term_reflection_prompts = []
        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

    def _sample_evaluate_register(self, prompt, operator_name=""):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        func = self._sampler.draw_sample(prompt)
        func = SampleTrimmer.sample_to_function(func, self._template_program)
        sample_time = time.time() - sample_start
        if func is None:
            return
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return
        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()

        if score is None:
            func.score = None
            func.all_ins_performance = None
        else:
            func.score = score.get('score', None)
            func.all_ins_performance = score.get('all_ins_performance', None)

        # register to profiler
        # func.score = score
        func.operator = operator_name
        func.evaluate_time = eval_time
        func.sample_time = sample_time
        with self._lock:
            # 1. 注册到 Profiler (虽然 Profiler 可能线程安全，但放在锁里更保险)
            if self._profiler is not None:
                self._profiler.register_function(func)
                if isinstance(self._profiler, ReEvoProfiler):
                    self._profiler.register_population(self._population)

            # 2. 更新总采样数 (必须加锁！)
            self._tot_sample_nums += 1

            # 3. 注册到 Population (Population 内部有锁，但放在这里也没问题)
            self._population.register_function(func)

    def _iteratively_ga_evolve(self):
        """
        修复版进化循环：
        1. 使用全局计数器 _global_crx_count 来决定何时触发 Mutation。
        2. 关键数据操作加锁。
        3. 共享 Reflection 历史。
        """
        while True:
            try:
                # ==========================
                # 0. 循环开始前的终止检查 (加锁)
                # ==========================
                with self._lock:
                    if self._tot_sample_nums >= self._max_sample_nums:
                        break

                # ==========================
                # 1. Short Term Reflection (短时反思)
                # ==========================
                # Population 内部有锁，select 是安全的
                indivs = [self._population.selection() for _ in range(2)]

                # 生成反思 Prompt
                st_prompt_template = ReEvoPrompt.get_short_term_reflection_prompt(
                    self._task_description_str, indivs
                )

                # 调用 LLM (耗时操作，不加锁，允许并发)
                st_reflection = self._sampler._llm.draw_sample(st_prompt_template)

                # 将反思结果存入全局列表 (加锁)
                with self._lock:
                    self._short_term_reflection_prompts.append(st_reflection)
                    # 保持列表不要太长，只留最近的 N 个
                    if len(self._short_term_reflection_prompts) > self._MAX_SHORT_TERM_REFLECTION_PROMPT:
                        self._short_term_reflection_prompts.pop(0)

                # ==========================
                # 2. Crossover (交叉)
                # ==========================
                crx_prompt = ReEvoPrompt.get_crossover_prompt(
                    self._task_description_str,
                    st_reflection,
                    indivs
                )

                # 采样并注册 (内部包含 Evaluation)
                self._sample_evaluate_register(crx_prompt, operator_name='reevo_cro')

                # ==========================
                # 3. 检查是否触发 Mutation (变异)
                # ==========================
                do_mutation_phase = False

                with self._lock:
                    # 全局计数器 +1
                    self._global_crx_count += 1

                    # 关键修复：只要全局计数器达到了 pop_size 的倍数，就触发 Mutation
                    # 这样无论几个线程在跑，每生成 20 个 Crossover，一定会有一个线程进入这里
                    if self._global_crx_count > 0 and self._global_crx_count % self._pop_size == 0:
                        do_mutation_phase = True

                # ==========================
                # 4. 执行 Mutation 阶段
                # ==========================
                if do_mutation_phase:
                    # 注意：进入这里的只有一个线程 (这一轮的"幸运儿")

                    # 4.1 Long Term Reflection
                    # 获取快照 (加锁读取，防止读取时被其他线程修改)
                    with self._lock:
                        last_lt_prompt = self._long_term_reflection_prompts[
                            -1] if self._long_term_reflection_prompts else ''
                        recent_st_prompts = list(self._short_term_reflection_prompts)  # 浅拷贝副本

                    lt_prompt_template = ReEvoPrompt.get_long_term_reflection_prompt(
                        self._task_description_str,
                        last_lt_prompt,
                        recent_st_prompts,
                    )

                    # LLM 调用
                    lt_reflection = self._sampler._llm.draw_sample(lt_prompt_template)

                    # 存回全局列表
                    with self._lock:
                        self._long_term_reflection_prompts.append(lt_reflection)

                    # 4.2 Elite Mutation Loop
                    # 计算需要变异的数量
                    mutation_count = int(self._mutation_rate * self._pop_size)

                    for _ in range(mutation_count):
                        # 每次变异前检查是否由于其他线程的操作导致名额用完了
                        with self._lock:
                            if self._tot_sample_nums >= self._max_sample_nums:
                                break

                        # 获取精英
                        elite_func = self._population.elite_function

                        mut_prompt = ReEvoPrompt.get_elist_mutation_prompt(
                            self._task_description_str,
                            lt_reflection,
                            elite_func
                        )

                        self._sample_evaluate_register(mut_prompt, operator_name='reevo_mut')

            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                continue

    def _iteratively_init_population(self):
        """Let a thread repeat {sample -> evaluate -> register to population}
        to initialize a population.
        """
        while self._population.generation == 0:
            try:
                # get a new func using i1
                prompt = ReEvoPrompt.get_pop_init_prompt(self._task_description_str, self._function_to_evolve)
                if self._debug_mode:
                    print(f'Init Prompt: {prompt}')
                self._sample_evaluate_register(prompt, operator_name='init')
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
            Thread(target=fn, args=args, kwargs=kwargs)
            for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        if not self._resume_mode:
            # do initialization
            self._multi_threaded_sampling(self._iteratively_init_population)

        # evolutionary search
        self._multi_threaded_sampling(self._iteratively_ga_evolve)

        # finish
        if self._profiler is not None:
            self._profiler.finish()

    def using_flow(self, worst_case_percent=10, top_k=None, max_gen=None):
        print(f"🔍 Loading model from {self._profiler._log_dir}...")
        designed_results_path = os.path.join(self._profiler._log_dir, 'population')

        # 1. Define the pattern to match 'pop_X.json' and capture 'X'
        pattern = re.compile(r'^pop_(\d+)\.json$')

        max_x = -1
        latest_file = None

        # 2. Check if the directory exists
        if not os.path.isdir(designed_results_path):
            print(f"Error: Directory not found: {designed_results_path}")
            return  # Or raise an Exception

        # 3. Iterate over files in the directory
        for filename in os.listdir(designed_results_path):
            match = pattern.match(filename)

            # 4. If the filename matches
            if match:
                # Extract the number (group 1) and convert to int
                current_x = int(match.group(1))

                # --- [新增逻辑] 限制最大代数 ---
                if max_gen is not None and current_x > max_gen:
                    continue
                    # ---------------------------

                # 5. Check if it's the largest number found so far
                if current_x > max_x:
                    max_x = current_x
                    latest_file = filename

        # 6. Check if any matching file was found
        # 6. Check if any matching file was found
        if latest_file is None:
            error_msg = f"No 'pop_x.json' files found"
            if max_gen: error_msg += f" (with max_gen <= {max_gen})"
            print(f"Error: {error_msg} in {designed_results_path}")
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
