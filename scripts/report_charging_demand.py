"""Summarize already sealed, independently checked sequence forecasts (no tuning)."""
import os
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[var] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    root = Path(args.run)
    assert json.loads((root/'VERIFIED.json').read_text())['passed']
    assert not (root/'REPORT.md').exists()
    test = json.loads((root/'test_metrics.json').read_text())
    val = json.loads((root/'validation_metrics.json').read_text())
    seal = json.loads((root/'FINAL_SELECTION.json').read_text())
    config = json.loads((root/'CONFIG.json').read_text())
    targets = ['order_count', 'start_cohort_energy_kwh']
    simple = ['previous_day_asof', 'previous_week_asof', 'training_hour_of_week_mean', 'ridge_ar']
    families = ['mlp', 'gru', 'lstm']
    names = {'previous_day_asof': 'Previous day', 'previous_week_asof': 'Previous week',
             'training_hour_of_week_mean': 'Weekly mean', 'ridge_ar': 'Ridge AR',
             'mlp': 'MLP', 'gru': 'GRU', 'lstm': 'LSTM'}
    def value(result, name, c, metric):
        return result['methods'][name]['metrics'][c]['all_24_steps'][metric]
    aggregate = {}
    rows = []
    for name in simple + families:
        members = [name] if name in simple else ['{}_seed{}'.format(name, s) for s in config['seeds']]
        aggregate[name] = {}
        row = [names[name]]
        for c in targets:
            aggregate[name][c] = {}
            for metric in ['MAE', 'RMSE', 'WAPE', 'positive_target_MAE', 'bias']:
                a = [value(test, m, c, metric) for m in members]
                aggregate[name][c][metric] = {'mean': float(np.mean(a)), 'sample_std': float(np.std(a, ddof=1)) if len(a)>1 else 0.0}
                if metric in ['MAE', 'RMSE']:
                    row.append('{:.4f}'.format(np.mean(a)) + (' ± {:.4f}'.format(np.std(a, ddof=1)) if len(a)>1 else ''))
        rows.append('| '+' | '.join(row)+' |')
    selected = seal['selected_on_validation']
    capped = [name for name, item in seal['checkpoints'].items()
              if item.get('epoch') == config['max_epochs']]
    baseline = 'training_hour_of_week_mean'
    improvements = {c: 100*(1-value(test, selected, c, 'MAE')/value(test, baseline, c, 'MAE')) for c in targets}
    def pair(name, metric):
        return '/'.join('{:.4f}'.format(value(test, name, c, metric)) for c in targets)
    interpretation = (
        'MAE不能单独代表全部精度。验证选定模型的订单/电量RMSE为{}，星期小时均值为{}，Ridge为{}。'
        '验证选定模型在正需求小时的MAE为{}，均值基线为{}；这项指标须与全体小时MAE一起判断。'
        '验证选定模型的平均有符号误差为{}（负数表示低估）；WAPE为{}（比例，不是准确率）。'
        '以上是当前短时间测试与轻量配置下的结果。').format(
            pair(selected, 'RMSE'), pair(baseline, 'RMSE'), pair('ridge_ar', 'RMSE'),
            pair(selected, 'positive_target_MAE'), pair(baseline, 'positive_target_MAE'),
            pair(selected, 'bias'), pair(selected, 'WAPE'))
    y = np.load(root/'test_targets.npz')['y'].astype(np.float64)
    p = np.load(root/(selected+'_test.npz'))['prediction'].astype(np.float64)
    b = np.load(root/(baseline+'_test.npz'))['prediction'].astype(np.float64)
    # Origin-level moving-block bootstrap retains full horizons/regions together.
    # Exploratory uncertainty only: adjacent overlapping forecasts are not IID samples.
    differences = (np.abs(p-y)-np.abs(b-y)).mean(axis=(1,2))
    rng = np.random.default_rng(20261022)
    block, B, n = 48, 2000, len(y)
    bs = np.empty((B, 2))
    for i in range(B):
        starts = rng.integers(0, n-block+1, size=int(np.ceil(n/block)))
        indices = np.concatenate([np.arange(s, s+block) for s in starts])[:n]
        bs[i] = differences[indices].mean(0)
    interval = {c: {'mean_MAE_difference': float(differences[:, i].mean()),
                    'exploratory_95pct_interval': np.quantile(bs[:, i], [.025, .975]).tolist()}
                for i, c in enumerate(targets)}
    summary = {'selected_on_validation': selected, 'test_windows': test['origins'], 'regions': test['regions'],
               'seed_mean_and_std': aggregate, 'selected_model_metrics': test['methods'][selected],
               'selected_MAE_reduction_percent_vs_weekly_mean': improvements,
               'paired_MAE': interval, 'bootstrap': {'block_hours': block, 'draws': B, 'seed': 20261022,
                    'caution': 'Exploratory moving-block interval from 265 overlapping origins in a short test period; not IID confidence or causal evidence'},
               'test_tuning': False, 'prior_exposure': config['prior_exposure']}
    summary['best_checkpoint_at_epoch_cap'] = capped
    summary['interpretation'] = interpretation
    (root/'RESULT_SUMMARY.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    display = simple+families
    colors = ['#afbccb']*3+['#658ba5', '#387db4', '#239a8c', '#8064a2']
    for j, (ax, c) in enumerate(zip(axes, targets)):
        means = [aggregate[m][c]['MAE']['mean'] for m in display]
        stds = [aggregate[m][c]['MAE']['sample_std'] for m in display]
        bars = ax.bar(np.arange(len(display)), means, yerr=stds, color=colors, capsize=3)
        ax.set_xticks(np.arange(len(display))); ax.set_xticklabels([names[m] for m in display], rotation=30, ha='right')
        ax.set_ylabel('MAE (orders / region / hour)' if j == 0 else 'MAE (kWh / region / start-hour)')
        ax.set_title('Order count' if j == 0 else 'Session energy of starting orders')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.set_ylim(0, max(means)*1.17)
        for bar, number in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, number+max(means)*.02, '{:.3f}'.format(number), ha='center', fontsize=9)
    fig.suptitle('Held-out test | 112 regions | hourly, 168 -> 24 | neural: mean +/- SD of 3 seeds', fontsize=12)
    fig.savefig(root/'test_accuracy.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    lead = [1,3,6,12,24]
    for c, ax in zip(targets, axes):
        for name in ['training_hour_of_week_mean', 'ridge_ar', 'mlp_ensemble', 'gru_ensemble', 'lstm_ensemble']:
            metric = test['methods'][name]['metrics'][c]
            ax.plot(lead, [metric['lead_hours'][str(k)]['MAE'] for k in lead], marker='o', label=name.replace('_', ' '))
        ax.set_xlabel('Forecast lead (hours)'); ax.set_ylabel('MAE'); ax.set_title(c); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.savefig(root/'test_horizons.png', dpi=180); plt.close(fig)

    text = ['# 区域充电需求：序列预测基线实验', '', '## Material Passport', '',
            '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
            '- Origin Date: 2026-10-22', '- Verification Status: VERIFIED for saved prediction metrics and checkpoint re-inference; training rerun not performed',
            '- Version Label: charging-sequence-baselines-v1', '',
            '## 任务与评价', '',
            '杭州112个训练期选定的活跃约2km网格，1小时粒度，历史168小时直接预测未来24小时。训练/验证/测试起点为865/337/265；预测窗口每小时滑动一次。训练2025-11-01至12-15，验证12-16至12-31，测试2026-01-01至01-13。预测标签加24小时成熟缓冲不跨阶段。', '',
            '指标按预测起点×24个预测步×区域计分；重叠目标在不同起点下分别计分。以下单位是每区域每目标小时；神经网络行为独立种子的均值±样本标准差，不是最优种子，也不是集成预测的分数。电量是开始于该小时的订单最终整单电量，不是小时电表负荷。', '',
            '| 模型 | 订单 MAE | 订单 RMSE | 电量 MAE (kWh) | 电量 RMSE (kWh) |',
            '|---|---:|---:|---:|---:|'] + rows + ['', '## 验证选定模型与测试', '',
            '在测试评分前根据验证集双目标相对MAE选定 `{}`，相关权重、代码、输入、配置哈希已写入 `FINAL_SELECTION.json`。所有模型均只在训练集拟合；没有在验证集上重新训练，也没有根据测试结果改参数。'.format(selected), '',
            '该模型测试订单MAE {:.4f}，电量MAE {:.4f} kWh；相对训练期星期小时均值基线的MAE下降分别为 {:.2f}% / {:.2f}%。负值表示变差。'.format(value(test,selected,targets[0],'MAE'),value(test,selected,targets[1],'MAE'),improvements[targets[0]],improvements[targets[1]]), '',
            interpretation, '',
            '固定三种子的均值集成也统一评价，并与种子平均指标分开保存。具体每个种子、集成、各预测步、正需求小时误差、系统偏差和分区域误差见JSON及CSV。', '',
            '## 训练与复现', '',
            'Ridge逐区域使用过去168小时的两个目标及小时/星期独热变量，直接输出24步；alpha候选1、10、100、1000，由验证集选择。MLP展平全区域历史，隐藏层128；GRU/LSTM将全区域历史作为多变量序列，单层隐藏状态64，接直接24步预测头。神经网络额外使用过去日历、历史结算成熟标记和未来已知日历；Softplus保证非负输出。它们是轻量基线实现，不是某篇论文的完整复现。', '',
            '模型专用标准化只由训练截止前已完成订单拟合，原数据集log1p标准化文件未修改。损失为按全局训练通道标准差缩放后的MSE。AdamW学习率0.001，weight decay0.0001，batch32，最多30轮，耐心5轮，梯度裁剪1，dropout0.1；随机种子17/29/43。每个种子的训练轮次根据验证双目标相对MAE选定。', '',
            '最佳检查点达到30轮训练上限的条件为：{}。这些条件尚未验证充分收敛；结果描述本轮计算预算和配置，不能据此判定某类网络在充分调参后必然更差。'.format(', '.join(capped) or '无'), '',
            '本地CPU、单线程、单训练进程。环境Python3.9.7、NumPy1.22.4、Pandas1.3.4、PyTorch2.8.0+cpu；训练全量865个时间起点，没有将重叠窗口随机分入不同集合。', '',
            '## 核验与解释边界', '',
            '已独立复算保存的全部验证/测试指标，抽取每模型3个起点从权重重推并比对；事件掩码重建历史与流式实现一致；修改尚未完成订单的电量不影响历史输入；验证选轮、输入及模型哈希一致。未从头重复一次全部训练，因此核验不等同于跨环境训练复现。', '',
            '48小时连续起点区块bootstrap（2000次）提供相对星期小时均值的探索性配对MAE差异区间，保留同一起点全部区域和预测步；测试期间短、重叠窗口相关，不能把几十万个区域小时点当作独立样本。具体值见RESULT_SUMMARY.json，不据此宣称普遍显著优势。', '',
            '本轮只评价112个核心网格，不能外推到全部1054个目录网格。数据中的0表示没有观测到合格订单；此前个人站点实验已看过另一协议下January聚合结果，不能称首次盲测。本结果不是LLM、记忆策略、RL或AgentMove复现，也不衡量真实充电费用或等待时间。', '',
            '## 文件', '', '`validation_metrics.json`、`test_metrics.json`：全部原单位指标；`*_summary.csv`：便于查看的表格；`training_history.json`：每轮训练/验证记录；`FINAL_SELECTION.json`：测试前封存；`DEVELOPMENT_VERIFIED.json`与`VERIFIED.json`：核验；`test_accuracy.png`与`test_horizons.png`：图。', '']
    (root/'REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    output_names = ['RESULT_SUMMARY.json','REPORT.md','test_accuracy.png','test_horizons.png']
    manifest = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in output_names}
    (root/'REPORT_FILES.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'selected_on_validation': selected, 'improvement_percent': improvements, 'test_aggregate': aggregate}, ensure_ascii=False))


if __name__ == '__main__':
    main()
