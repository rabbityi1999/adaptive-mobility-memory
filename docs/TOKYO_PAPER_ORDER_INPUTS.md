# 论文文字流程：独立输入检查点

## Material Passport

2026-10-23；AgentMove输入协议对照；状态ANALYZED。这是按附录10.7重建筛选顺序的单独条件，不修改已完成的官方代码版基线和消融。模型调用0，尚无新预测分数。

## 本轮完成

沿用已核验的东京72小时锚定session和当前地址，先保留至少5个原始session的用户，再删除少于4次签到的session，最后按保留段的顺序切分70%／10%／20%。原始轨迹ID保留，新的保留段排序单独记录。LLM个人历史仍使用前80%，与现有框架接口一致。

| 项目 | 结果 |
|---|---:|
| 保留用户 |7,360|
| 保留session |88,611|
| 保留签到 |687,542|
| 训练／验证／测试session |58,524／9,055／21,032|
| LLM历史session／记录 |67,579／556,513|
| 已生成查询 |200名用户各1条|
| 逐字段读回验证 |200/200通过|
| 个人时间边界违规 |0|

构建退出0、耗时7.82秒、峰值RSS约1.12GiB，限制2核、16GiB地址空间、600秒CPU时间与900秒外层超时。独立读回脚本也退出0，核验6个产物哈希，以及每条查询的历史、上下文、目标占位、独立真值和截止时间。[构建汇总](../reports/tokyo_paper_order_inputs_20261023.json)；[读回验证](../reports/tokyo_paper_order_validation_20261023.json)。

本轮只生成200条选定查询的预测输入字典；完整21,032条测试索引保存在session清单。过滤后的全城记录、清单、查询输入和真值保留在服务器 `/mnt/xuyi/agentmove_paper_order_inputs_20261023`，不上传原始轨迹。

## 下一步及边界

1. 从全体合格用户的前80%历史构建群体图。**不能直接把仅含200人的查询字典交给原构图器，否则会误构成200人子图。** 本检查点保存了全城过滤记录和历史标记；总历史记录556,513条可作核对。
2. 单列原始静态群体图及按每条查询截止时间过滤的证据，核验边、属性来源和邻居排序。
3. 固定相同Llama3接口、核验实际提示与截断，之后才启动新对照推理；不复用旧查询的WKG／full输出而跳过身份检查。

这200名用户与旧队列重叠178名、查询重叠22条，不是独立开发集。论文表4的112,942条轨迹仍未解释；现代地址、模型接口等差异也仍保留。新旧查询集不同，跨协议分数差不能解释为配对方法收益。

## 复跑与验证

Linux，Python环境含pandas2.1.4。只接受已有哈希固定的可信pickle输入；输出目录必须不存在。验证输出也必须是新文件。

```bash
timeout --signal=TERM --kill-after=10s 900s python scripts/build_paper_order_inputs.py \
  --checkpoint /path/to/verified_session_checkpoint \
  --audit /path/to/protocol_audit/result_v2.json \
  --output /new/path/paper_order_inputs

timeout --signal=TERM --kill-after=10s 900s python scripts/validate_paper_order_inputs.py \
  --checkpoint /path/to/verified_session_checkpoint \
  --inputs /new/path/paper_order_inputs \
  --output /new/path/paper_order_inputs/validation.json
```

发布物仅为源码、汇总、哈希及说明。未构建群体图、未核验渲染提示、未启动模型推理；这些步骤不得从“输入通过”推定为已完成。
