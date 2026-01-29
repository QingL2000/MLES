<div align="center">
<h1 align="center">MELS: Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies</h1>
</div>
<br>

---

## Introduction 📖

Transparency and high performance are essential goals in designing control policies, particularly for safety-critical tasks. 
MELS (Multimodal LLM-assisted Evolutionary Search) combines the powerful reasoning and generation capabilities of Multimodal Large Language Models (MLLMs) with the iterative optimization strengths of evolutionary computation, **enabling the automatic design of high-performing and transparent control policies**. 



MELS is designed to mimic how human experts develop policies. It analyzes behavior patterns and then intelligently refines programmatic policies with targeted improvements. Unlike traditional Deep Reinforcement Learning (DRL) methods, MELS offers:

1. **Completely Transparent Control Policies**: Policies are expressed as human-readable programs, making their logic transparent and easily understandable.

2. **Fully Transparent and Traceable Policy Design Process**: Every step of the policy evolution is meticulously recorded and can be thoroughly analyzed, providing full insight into the discovery journey.

We demonstrate that MELS achieves performance comparable to Proximal Policy Optimization (PPO) in terms of both policy search efficiency and the performance of the generated policies across two control tasks.


<p align="center">
<img src="./figs/MLES_0919.png" alt="Car Racing Evolution Process Comparison" style="width:80%;" />
</p>


In this repository, we showcase the application of MELS for automated policy discovery using the Lunar Lander and Car Racing environments as illustrative examples. We provide the discovered policies from our experiments and offer tools to analyze the evolutionary process.

---

## 🎁 Requirements & Installation

You can quickly set up the required Python environment using the provided `environment.yml` file.

1.  **Create the Conda environment**:
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the environment**:
    ```bash
    conda activate mmeoh
    ```

---

## 💻 Example Usage

### Quick Start:

> [!Note]
> Before running the script, you'll need to configure your Large Language Model (LLM) API settings. Here's an example configuration for DeepSeek:
>
> 1.  Set `host`: `'api.deepseek.com'`
> 2.  Set `key`: `'your_api_key'` (Replace with your actual API key)
> 3.  Set `model`: `'deepseek-chat'`

```python
from llm4ad.task.machine_learning.car_raceing_continue import RacingCarEvaluation
from llm4ad.task.machine_learning.car_raceing_continue import template_program
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.mmeoh import MMEoH
from llm4ad.method.mmeoh import EoHProfiler


def main(run_id):
    # Initialize LLM API
    llm = HttpsApi(host="xxx",  # Your host endpoint, e.g., api.openai.com, api.deepseek.com
                   key="sk-xxx",  # Your API key, e.g., sk-xxxxxxxxxx
                   model="xxx",  # Your LLM, e.g., gpt-4o-mini, deepseek-chat
                   timeout=20)
    
    # Define logging directory for the run
    log_dir = f'batch/mmeoh/{run_id}'  # 'mmeoh' is our MELS implementation
    
    # Initialize the task for evaluation (e.g., Car Racing)
    task = RacingCarEvaluation(whocall='mmeoh')   
    
    # Path to the initial population seed
    seedpath = r'init_pop_size16.json'

    # Define the operators for the evolutionary process
    operators_setting = ('e1', 'e2', 'm1_M', 'm2_M')

    # Initialize and run the MMEoH (MELS) method
    method = MMEoH(llm=llm,
                   profiler=EoHProfiler(log_dir=log_dir, log_style='complex'),
                   evaluation=task,
                   max_sample_nums=2000,
                   max_generations=None, # Set to a specific number for a fixed run duration
                   pop_size=16,
                   num_samplers=4,
                   num_evaluators=4,
                   debug_mode=False,
                   operators=operators_setting, # Example: ('e1', 'e2', 'm1_M', 'm2_M')
                   seed_path=seedpath,
                   multi_thread_or_process_eval='process' # Use multiprocessing for evaluation
                   )

    # Start the policy discovery process
    method.run()


if __name__ == '__main__':
    main(1)
```

In just about an hour of automated discovery, MELS can provide you with a near-perfect control policy for Car Racing!

<p align="center">
<img src="./figs/performance on test.png" alt="Car Racing Performance" style="width:90%;" />
</p>

The discovery process is completely traceable and verifiable, offering insights into how policies evolve:

<p align="center">
<img src="./figs/Interpretable evolutionary process_v4.png" alt="Interpretable Evolutionary Process" style="width:90%;" />
</p>

Compared to traditional DRL algorithms like PPO and DQN, MELS demonstrates remarkably efficient algorithm discovery:

<p align="center">
<img src="./figs/car_racing_evolurion_process.png" alt="Car Racing Evolution Process Comparison" style="width:40%;" />
</p>

---

##  Analyzing Your MELS Results

We provide tools in the `analysis_results` directory to help you deeply understand the evolutionary process and comprehensively evaluate the performance of discovered policies.

* **Track Policy Ancestry**: Use `analysis_family_of_one_individual_v2.py` to trace the entire lineage of any specific policy you're interested in. This allows you to explore its "family tree" and understand its evolutionary path.
* **Compare Performance and Efficiency**: The `analysis_results/LES_RL_behavior_v3.py` script lets you compare the performance and efficiency of different methods on policy discovery tasks, giving you clear insights into MELS's advantages.