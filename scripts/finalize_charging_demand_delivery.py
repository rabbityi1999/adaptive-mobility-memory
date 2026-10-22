# -*- coding: utf-8 -*-
"""Publish only reviewed source and aggregate experiment summaries, never customer data."""
from pathlib import Path
import ast
import datetime as dt
import hashlib
import json

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--repository", required=True)
args = ap.parse_args()
RUN = Path(args.run).resolve()
REPO = Path(args.repository).resolve()
assert hashlib.sha256((RUN/"FINAL_SELECTION.json").read_bytes()).hexdigest() == "482925b6bfb71fd211a31ac5b585568e0112ddbe2d8d17fc1dfd6a58e84401df", "One-off delivery helper for the frozen 2026-10-22 run only"
assert (RUN/'VERIFIED.json').exists()
assert not (RUN/'RUNNING.lock').exists()
report_source = REPO/'scripts/report_charging_demand.py'
# Add the reporting interpretation without changing any frozen predictions or scores.
metric = json.loads((RUN/'test_metrics.json').read_text(encoding='utf-8'))
summary = json.loads((RUN/'RESULT_SUMMARY.json').read_text(encoding='utf-8'))
selected = summary['selected_on_validation']
targets = ['order_count', 'start_cohort_energy_kwh']
def pair(name, key):
    return '/'.join('{:.4f}'.format(metric['methods'][name]['metrics'][c]['all_24_steps'][key]) for c in targets)
interpretation = (
    'MAE不能单独代表全部精度。验证选定模型的订单/电量RMSE为{}，星期小时均值为{}，Ridge为{}。'
    '验证选定模型在正需求小时的MAE为{}，均值基线为{}；这项指标须与全体小时MAE一起判断。'
    '验证选定模型的平均有符号误差为{}（负数表示低估）；WAPE为{}（比例，不是准确率）。'
    '以上是当前短时间测试与轻量配置下的结果。').format(
        pair(selected,'RMSE'),pair('training_hour_of_week_mean','RMSE'),pair('ridge_ar','RMSE'),
        pair(selected,'positive_target_MAE'),pair('training_hour_of_week_mean','positive_target_MAE'),
        pair(selected,'bias'),pair(selected,'WAPE'))
summary['interpretation'] = interpretation
(RUN/'RESULT_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),encoding='utf-8')
report = (RUN/'REPORT.md').read_text(encoding='utf-8')
anchor = '固定三种子的均值集成也统一评价'
if interpretation not in report:
    report = report.replace(anchor, interpretation+'\n\n'+anchor)
(RUN/'REPORT.md').write_text(report,encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {n:sha(RUN/n) for n in ['RESULT_SUMMARY.json','REPORT.md','test_accuracy.png','test_horizons.png']}
manifest['report_generator_sha256'] = sha(report_source)
(RUN/'REPORT_FILES.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

# Public artifact: aggregate metrics and audit receipts only, no region-specific errors/IDs.
public = json.loads(json.dumps(summary))
for c in targets:
    public['selected_model_metrics']['metrics'][c].pop('region_MAE',None)
public['verification'] = {
    split: {k:v for k,v in json.loads((RUN/name).read_text()).items() if k != 'artifact_sha256'}
    for split,name in [('validation','DEVELOPMENT_VERIFIED.json'),('test','VERIFIED.json')]}
public['seal_sha256'] = sha(RUN/'FINAL_SELECTION.json')
public['configuration_sha256'] = sha(RUN/'CONFIG.json')
public['run_directory_name'] = RUN.name
public_path = REPO/'reports/charging_demand_sequence_20261022.json'
assert not public_path.exists()
public_path.write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')

addition = '''

## 6. 区域充电需求预测（独立于AgentMove的任务）

2026-10-22完成首轮序列预测比较：杭州112个训练期活跃约2km网格，1小时粒度，168小时预测24小时，训练/验证/测试865/337/265个滚动起点。训练2025-11-01至12-15、验证12-16至12-31、测试2026-01-01至01-13；阶段尾部保留24小时标签成熟缓冲。只使用当时完成的订单重建输入。电量指开始于该小时订单的整单最终电量，不是电力负荷。

| 方法（测试） | 订单MAE | 订单RMSE | 电量MAE kWh | 电量RMSE kWh |
|---|---:|---:|---:|---:|
|训练期星期小时均值|0.6777|1.2472|30.4209|54.8548|
|Ridge自回归|0.6735|1.2001|30.1012|52.6639|
|MLP（三种子指标均值）|0.6878|1.3020|30.3092|57.5032|
|GRU（三种子指标均值）|0.6957|1.2351|31.0358|54.0099|
|LSTM（三种子指标均值）|0.6989|1.2430|31.1681|54.3233|
|MLP三种子预测集成（验证选定）|0.6639|1.2506|29.3442|55.4022|

种子17/29/43，CPU单线程；30轮上限、验证早停。测试前封存全部模型、配置和哈希，没有验证重训或测试调参。MLP集成的两个MAE相对星期小时均值降低2.04%/3.54%，但RMSE略差，正需求小时MAE较高，并存在明显低估（平均偏差−0.3187单/−13.8553 kWh）；不能宣称全面更准。Ridge在本轮RMSE更优。部分循环网络最佳轮次达到预算上限，尚未验证充分收敛。

两个阶段各独立复算1020项指标，并从保存权重重推3个起点/模型、验证时间可用性与未来完成事件隔离。训练约624秒、测试约7.4秒；9次神经网络训练，无LLM调用。核验不是从头重复训练。测试窗口高度重叠，独立车辆数不适用于这个区域任务；不能把区域小时当作独立用户。January曾在另一协议下被查看过聚合结果，保留披露。

结果与标准差：[聚合报告](../reports/charging_demand_sequence_20261022.json)。配置、模型定义和复跑命令：[区域预测说明](CHARGING_DEMAND_BASELINES.md)。私有输出位于工作区 `research_workspace/charging_demand_20261021/experiments/sequence_baselines_v1`，包含封存、预测、指标、权重、图与核验文件；原始数据和权重不上传。该实验不改变上面的AgentMove复现结论，也没有训练记忆策略或RL。后续如调整模型，保留本轮测试结果，另设新留出期验证改进。
'''
worklog=REPO/'docs/WORK_LOG.md'
assert '## 6. 区域充电需求预测' not in worklog.read_text(encoding='utf-8')
with worklog.open('a',encoding='utf-8') as f: f.write(addition)
handoff = {'at':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'completed_and_verified',
           'pid':None,'automation_resumed':False,'validation_selected':selected,'test_scored':True,
           'no_post_test_tuning':True,'data_unchanged':True,'report_sha256':sha(RUN/'REPORT.md'),
           'source_delivery':'commit/push pending','reporting_change_only':True}
(RUN/'HANDOFF.json').write_text(json.dumps(handoff,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'report':str(RUN/'REPORT.md'),'public_summary':str(public_path),'verified':True},ensure_ascii=False))
