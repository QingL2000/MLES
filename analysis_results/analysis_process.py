import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# ==============================================================================
# 1. 用户配置区域 (USER CONFIGURATION)
# ==============================================================================

SCENARIOS = {
    "CVRP_Scenario": {
        "EoH": [
            r"C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Eoh\0\20260202_171934_Problem_EoH",
        ],
        "MEoH": [
            r"C:\0_QL_work\014_mmeoh\MLES\example\cvrplib\log\Eoh\0\20260202_171939_Problem_EoH",
        ],
    },
}

SCENARIO_SETTINGS = {
    "CVRP_Scenario": {
        "negate_data": True,
        "minimize": True,
        "max_plot_range": 2000
    },
    "Moonlander_Scenario": {
        "negate_data": False,
        "minimize": False,
        "max_plot_range": 2000
    },
    "OBP_Scenario": {
        "negate_data": True,
        "minimize": True,
        "max_plot_range": 2000
    },
    "TSP_Scenario": {
        "negate_data": True,
        "minimize": True,
        "max_plot_range": 2000
    }
}

# ==============================================================================
# 2. 全局绘图风格配置
# ==============================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['lines.linewidth'] = 2.0


class EvolutionAnalyzer:
    def __init__(self, run_config: Dict[str, List[str]], minimize: bool = False, negate_data: bool = False,
                 max_range: Optional[int] = None):
        self.run_config = run_config
        self.minimize = minimize
        self.negate_data = negate_data
        self.max_range = max_range
        self.df = None

    def load_data(self):
        target_str = 'Minimization' if self.minimize else 'Maximization'
        process_str = ' (Negating values)' if self.negate_data else ''
        range_str = f' [Truncated @ {self.max_range}]' if self.max_range else ' [Full Range]'
        print(f"[{target_str}{process_str}{range_str}] Loading data...")

        all_records = []

        for method_name, paths in self.run_config.items():
            if isinstance(paths, str): paths = [paths]

            for run_idx, dir_path in enumerate(paths):
                search_pattern = os.path.join(dir_path, "samples", "samples_*.json")
                json_files = glob.glob(search_pattern)
                json_files = [f for f in json_files if "best" not in os.path.basename(f)]

                if not json_files: continue

                run_data = []
                for jf in json_files:
                    try:
                        with open(jf, 'r') as f:
                            c = json.load(f)
                            if isinstance(c, list): run_data.extend(c)
                    except:
                        pass

                run_data.sort(key=lambda x: x.get('sample_order', 0))

                current_run_best = float('inf') if self.minimize else float('-inf')

                for entry in run_data:
                    all_ins = entry.get('all_ins_performance', {})
                    if not all_ins: continue

                    scores = []
                    for v in all_ins.values():
                        val = v['score'] if isinstance(v, dict) and 'score' in v else v
                        if self.negate_data: val = -val
                        scores.append(val)

                    if not scores: continue
                    sample_mean = np.mean(scores)

                    if self.minimize:
                        current_run_best = min(current_run_best, sample_mean)
                    else:
                        current_run_best = max(current_run_best, sample_mean)

                    if self.minimize and current_run_best == float('inf'): current_run_best = sample_mean
                    if not self.minimize and current_run_best == float('-inf'): current_run_best = sample_mean

                    record = {
                        'Method': method_name,
                        'Run_ID': run_idx,
                        'Sample Order': entry.get('sample_order'),
                        'Current Best Mean': current_run_best,
                        'Sample Mean Score': sample_mean,
                        'Operator': entry.get('operator', 'init'),
                        'Source Pool': str(entry.get('from_which_pool', 'N/A')),
                        'all_ins': all_ins,
                        'root_path': dir_path
                    }
                    all_records.append(record)

        self.df = pd.DataFrame(all_records)
        if self.df is not None and not self.df.empty:
            print(f"Data loaded. Total records: {len(self.df)}")
        else:
            print("No data loaded.")
        print("-" * 60)

    # =================================================================
    # 对齐 + 截断逻辑
    # =================================================================
    def _align_and_interpolate(self, raw_df, value_col, x_col='Sample Order'):
        if raw_df.empty: return pd.DataFrame()

        min_x = raw_df[x_col].min()
        max_x = raw_df[x_col].max()

        if self.max_range is not None:
            max_x = min(max_x, self.max_range)

        # 这里使用了500个点的插值网格
        grid_x = np.linspace(min_x, max_x, num=500).astype(int)
        grid_x = np.unique(grid_x)

        aligned_records = []
        for (method, run_id), group in raw_df.groupby(['Method', 'Run_ID']):
            group = group.sort_values(x_col).drop_duplicates(subset=[x_col])
            group = group.set_index(x_col)
            reindexed = group.reindex(group.index.union(grid_x)).ffill().reindex(grid_x)

            reindexed['Method'] = method
            reindexed['Run_ID'] = run_id
            reindexed[x_col] = reindexed.index

            aligned_records.append(reindexed[[x_col, 'Method', 'Run_ID', value_col]])

        if not aligned_records: return pd.DataFrame()
        return pd.concat(aligned_records, ignore_index=True)

    # =================================================================
    # 统计表格
    # =================================================================
    def print_statistics(self):
        if self.df is None or self.df.empty: return
        print("\n" + "=" * 100)
        direction = "Min (Lower Better)" if self.minimize else "Max (Higher Better)"
        print(f" 📊 统计汇总 / Statistical Summary [{direction}]")
        print("=" * 100)

        stats_list = []
        for (method, run_id), group in self.df.groupby(['Method', 'Run_ID']):
            group = group.sort_values('Sample Order')
            final_score = group.iloc[-1]['Current Best Mean']

            row_data = {'Method': method, 'Run_ID': run_id, 'Final Score': final_score}

            if self.max_range is not None:
                cut_group = group[group['Sample Order'] <= self.max_range]
                if not cut_group.empty:
                    cut_score = cut_group.iloc[-1]['Current Best Mean']
                else:
                    cut_score = group.iloc[0]['Current Best Mean']
                row_data[f'Score@{self.max_range}'] = cut_score

            stats_list.append(row_data)

        metrics_df = pd.DataFrame(stats_list)

        def get_summary(df, col_name):
            res = df.groupby('Method')[col_name].agg(['mean', 'std', 'min', 'max']).reset_index()
            res[col_name] = res.apply(
                lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}" if not pd.isna(r['std']) else f"{r['mean']:.4f}", axis=1)
            return res[['Method', col_name]]

        final_summary = get_summary(metrics_df, 'Final Score')

        if self.max_range is not None:
            cut_col = f'Score@{self.max_range}'
            cut_summary = get_summary(metrics_df, cut_col)
            final_table = pd.merge(cut_summary, final_summary, on='Method')
            final_table = final_table[['Method', cut_col, 'Final Score']]
        else:
            final_table = final_summary

        counts = metrics_df.groupby('Method').size().reset_index(name='Runs')
        final_table = pd.merge(counts, final_table, on='Method')

        print(final_table.to_string(index=False, justify='center'))
        print("-" * 100 + "\n")

    # =================================================================
    # 辅助功能：绘图样式生成器 (含 Markers)
    # =================================================================
    def _get_run_style_dicts(self, plot_df):
        plot_df['Legend_Label'] = plot_df.apply(lambda x: f"{x['Method']} - Run {x['Run_ID']}", axis=1)
        unique_labels = sorted(plot_df['Legend_Label'].unique())
        unique_methods = sorted(plot_df['Method'].unique())

        base_method_palette = sns.color_palette("tab10", n_colors=len(unique_methods))
        method_color_map = dict(zip(unique_methods, base_method_palette))

        # 线型循环
        style_cycle = ['', (2, 2), (1, 1), (3, 1, 1, 1), (5, 2), (1, 3)]

        # [新增] 标记形状循环：圆圈, 方块, 三角, 菱形, 倒三角, 叉, 星
        marker_cycle = ['o', 's', '^', 'D', 'v', 'X', '*']

        final_palette = {}
        final_dashes = {}
        final_markers = {}

        for label in unique_labels:
            parts = label.rsplit(' - Run ', 1)
            m_name = parts[0]
            r_id = int(parts[1])

            final_palette[label] = method_color_map[m_name]
            final_dashes[label] = style_cycle[r_id % len(style_cycle)]
            # 根据 Run ID 分配 Marker
            final_markers[label] = marker_cycle[r_id % len(marker_cycle)]

        return plot_df, final_palette, final_dashes, final_markers

    # =================================================================
    # 图 1: 全局收敛 (Global Convergence - Mean)
    # =================================================================
    def plot_global_convergence(self, show=True, save_path='1_global_convergence.png'):
        if self.df is None or self.df.empty: return

        plot_df = self._align_and_interpolate(self.df, value_col='Current Best Mean')
        if plot_df.empty: return

        plt.figure()
        sns.lineplot(
            data=plot_df, x='Sample Order', y='Current Best Mean',
            hue='Method', style='Method', palette='tab10', errorbar='sd'
        )

        limit_txt = f" (First {self.max_range})" if self.max_range else " (Full)"
        plt.title(f'Global Convergence (Aggregated){limit_txt}')
        ylabel = 'Cost' if self.minimize else 'Score'
        plt.ylabel(f'Best Mean {ylabel}')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        if show: plt.show()
        plt.close()

    # =================================================================
    # 图 2: 全局收敛 - 单次运行 (含 Markers)
    # =================================================================
    def plot_individual_runs(self, show=True, save_path='2_individual_runs.png'):
        if self.df is None or self.df.empty: return

        plot_df = self._align_and_interpolate(self.df, value_col='Current Best Mean')
        if plot_df.empty: return

        # 获取样式字典，现在包含 markers
        plot_df, final_palette, final_dashes, final_markers = self._get_run_style_dicts(plot_df)

        plt.figure()
        sns.lineplot(
            data=plot_df,
            x='Sample Order',
            y='Current Best Mean',
            hue='Legend_Label',
            style='Legend_Label',
            palette=final_palette,
            dashes=final_dashes,
            markers=final_markers,  # 启用 Markers
            estimator=None,
            linewidth=2,
            alpha=0.8,
            markersize=8,  # 标记大小
            markevery=30  # [关键] 每30个点画一个标记，避免太密
        )

        limit_txt = f" (First {self.max_range})" if self.max_range else " (Full)"
        plt.title(f'Global Convergence (Individual Runs){limit_txt}')
        ylabel = 'Cost' if self.minimize else 'Score'
        plt.ylabel(f'Best Mean {ylabel}')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Run Details')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show: plt.show()
        plt.close()

    # =================================================================
    # 辅助功能：计算鲁棒性数据
    # =================================================================
    def _calculate_robustness_data(self):
        agg_data = []
        for (m_name, r_id), sub_df in self.df.groupby(['Method', 'Run_ID']):
            sub_df = sub_df.sort_values('Sample Order')
            all_ids = set()
            for m in sub_df['all_ins']: all_ids.update(m.keys())

            init_val = float('inf') if self.minimize else float('-inf')
            bsf = {i: init_val for i in all_ids}

            run_history = []
            for _, row in sub_df.iterrows():
                is_within_range = (self.max_range is None) or (row['Sample Order'] <= self.max_range)
                updated = False
                for iid, v_obj in row['all_ins'].items():
                    val = v_obj['score'] if isinstance(v_obj, dict) else v_obj
                    if self.negate_data: val = -val

                    if self.minimize:
                        if val < bsf[iid]: bsf[iid], updated = val, True
                    else:
                        if val > bsf[iid]: bsf[iid], updated = val, True

                if is_within_range:
                    vals = [v for v in bsf.values() if v != init_val]
                    if vals:
                        run_history.append({
                            'Method': m_name, 'Run_ID': r_id,
                            'Sample Order': row['Sample Order'],
                            'Mean Instance Best': np.mean(vals)
                        })
            agg_data.extend(run_history)
        return pd.DataFrame(agg_data)

    # =================================================================
    # 图 3: 鲁棒性 - 均值
    # =================================================================
    def plot_multi_method_robustness(self, show=True, save_path='3_robustness_mean.png'):
        if self.df is None or self.df.empty: return

        raw_df = self._calculate_robustness_data()
        plot_df = self._align_and_interpolate(raw_df, value_col='Mean Instance Best')

        if plot_df.empty: return
        plt.figure()
        sns.lineplot(
            data=plot_df, x='Sample Order', y='Mean Instance Best', hue='Method',
            style='Method', linewidth=2.5, errorbar='sd'
        )
        limit_txt = f" (First {self.max_range})" if self.max_range else ""
        plt.title(f'Robustness: Mean of Instance Best (Aggregated){limit_txt}')
        ylabel = 'Cost' if self.minimize else 'Score'
        plt.ylabel(f'Mean Best {ylabel}')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300)
        if show: plt.show()
        plt.close()

    # =================================================================
    # 图 4: 鲁棒性 - 单次运行 (含 Markers)
    # =================================================================
    def plot_individual_robustness(self, show=True, save_path='4_robustness_individual.png'):
        if self.df is None or self.df.empty: return

        raw_df = self._calculate_robustness_data()
        plot_df = self._align_and_interpolate(raw_df, value_col='Mean Instance Best')

        if plot_df.empty: return

        plot_df, final_palette, final_dashes, final_markers = self._get_run_style_dicts(plot_df)

        plt.figure()
        sns.lineplot(
            data=plot_df,
            x='Sample Order',
            y='Mean Instance Best',
            hue='Legend_Label',
            style='Legend_Label',
            palette=final_palette,
            dashes=final_dashes,
            markers=final_markers,  # 启用 Markers
            estimator=None,
            linewidth=2,
            alpha=0.8,
            markersize=8,
            markevery=30  # 每30个点画一个标记
        )
        limit_txt = f" (First {self.max_range})" if self.max_range else ""
        plt.title(f'Robustness: Mean of Instance Best (Individual Runs){limit_txt}')
        ylabel = 'Cost' if self.minimize else 'Score'
        plt.ylabel(f'Mean Best {ylabel}')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='Run Details')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show: plt.show()
        plt.close()

    # =================================================================
    # 图 5/6: 机制分析
    # =================================================================
    def plot_mechanism_analysis(self, feature_col, title_prefix, save_filename, show=True):
        if self.df is None or self.df.empty: return

        improved_records = []
        for (m, r), sub in self.df.groupby(['Method', 'Run_ID']):
            if self.max_range is not None:
                sub = sub[sub['Sample Order'] <= self.max_range].copy()
            else:
                sub = sub.copy()
            sub = sub.sort_values('Sample Order')
            if sub.empty: continue

            sub['Prev_Global_Best'] = sub['Current Best Mean'].shift(1)
            init_val = float('inf') if self.minimize else float('-inf')
            sub.iloc[0, sub.columns.get_loc('Prev_Global_Best')] = init_val

            if self.minimize:
                wins = sub[sub['Sample Mean Score'] < sub['Prev_Global_Best']]
            else:
                wins = sub[sub['Sample Mean Score'] > sub['Prev_Global_Best']]

            wins = wins[wins['Operator'] != 'init']
            improved_records.append(wins)

        if not improved_records:
            print(f"No improvement events found for {title_prefix}.")
            return

        impact_df = pd.concat(improved_records)

        fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1.2]})

        # Bar Chart
        count_data = impact_df.groupby(['Method', 'Run_ID', feature_col]).size().reset_index(name='Count')
        count_data['Bar_Label'] = count_data.apply(lambda x: f"{x['Method']} - R{x['Run_ID']}", axis=1)

        methods = sorted(count_data['Method'].unique())
        base_colors = sns.color_palette("husl", len(methods))
        final_bar_palette = {}
        for m_idx, m in enumerate(methods):
            m_runs = sorted(count_data[count_data['Method'] == m]['Bar_Label'].unique())
            m_colors = sns.light_palette(base_colors[m_idx], n_colors=len(m_runs) + 2, reverse=True)[0:len(m_runs)]
            for r_idx, label in enumerate(m_runs):
                final_bar_palette[label] = m_colors[r_idx]

        sns.barplot(
            data=count_data, x=feature_col, y='Count', hue='Bar_Label',
            palette=final_bar_palette, edgecolor='k', ax=axes[0]
        )
        axes[0].set_title(f'{title_prefix} Contribution Count (Breakdown by Run)')
        axes[0].set_xlabel('')
        axes[0].legend(loc='upper right', title='Run Details', ncol=2)

        # Timeline
        n_features = impact_df[feature_col].nunique()
        if n_features <= 10:
            timeline_palette = "tab10"
        elif n_features <= 20:
            timeline_palette = "tab20"
        else:
            timeline_palette = "gist_ncar"

        sns.stripplot(
            data=impact_df, x='Sample Order', y='Method', hue=feature_col,
            jitter=0.25, size=7, alpha=0.8, palette=timeline_palette,
            ax=axes[1], linewidth=0.5, edgecolor='white'
        )

        axes[1].grid(True, which='both', axis='x', linestyle='--', alpha=0.5)
        limit_txt = f" (First {self.max_range})" if self.max_range else " (Full)"
        axes[1].set_title(f'{title_prefix} Timeline Dynamics{limit_txt}')
        axes[1].set_xlabel('Sample Order')
        sns.move_legend(axes[1], "upper left", bbox_to_anchor=(1, 1), title=feature_col)

        plt.tight_layout()
        if save_filename: plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        if show: plt.show()
        plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 40)
    print(" >>> Select Application Scenario <<< ")
    print("=" * 40)

    scenario_keys = list(SCENARIOS.keys())
    for idx, key in enumerate(scenario_keys):
        print(f" [{idx + 1}] {key}")

    try:
        choice = int(input("\nEnter ID: ").strip())
        selected_key = scenario_keys[choice - 1]
    except:
        print("Invalid selection.")
        exit()

    paths = SCENARIOS[selected_key]
    settings = SCENARIO_SETTINGS.get(selected_key, {"negate_data": False, "minimize": False, "max_plot_range": None})
    max_rng = settings.get('max_plot_range', None)

    print(f"\nSelected: {selected_key}")
    print(f"Settings: Negate={settings['negate_data']}, Minimize={settings['minimize']}, Max Range={max_rng}")

    analyzer = EvolutionAnalyzer(
        paths,
        minimize=settings['minimize'],
        negate_data=settings['negate_data'],
        max_range=max_rng
    )

    analyzer.load_data()

    if analyzer.df is not None:
        analyzer.print_statistics()

        print("Generating plots (Check your plotting window)...")
        # 1. 全局收敛 - 均值
        analyzer.plot_global_convergence(save_path=f'{selected_key}_1_Conv_Mean.png')

        # 2. 全局收敛 - 单次
        analyzer.plot_individual_runs(save_path=f'{selected_key}_2_Conv_Indiv.png')

        # 3. 鲁棒性 - 均值
        analyzer.plot_multi_method_robustness(save_path=f'{selected_key}_3_Robust_Mean.png')

        # 4. 鲁棒性 - 单次
        analyzer.plot_individual_robustness(save_path=f'{selected_key}_4_Robust_Indiv.png')

        analyzer.plot_mechanism_analysis(
            feature_col='Operator',
            title_prefix='Operator',
            save_filename=f'{selected_key}_5_Operator_Analysis.png',
            show=True
        )

        analyzer.plot_mechanism_analysis(
            feature_col='Source Pool',
            title_prefix='Source Pool',
            save_filename=f'{selected_key}_6_Pool_Analysis.png',
            show=True
        )

        print("Done.")
