# 记忆证据与条目预算对照 v1

## Material Passport

- 类型：已授权项目的开发诊断协议与CPU准备，沿用ARS实验记录框架；单会话实现与检查，非独立研究审查。
- 协议：`memory-controls-v1-20261026`。冻结时间：2026-10-26，在本批控制提示构建和新预测之前；上一批600人结果已经观察，不称对研究假设的事前预注册或独立确认实验。
- 输入：[上一轮结果](MEMORY_READER_RESULTS.md)、封存的400fit／200development输入与提示。
- 本轮不新增reserve查询，不改变AgentMove复现结果，不训练路由。

## 问题与对照

上一轮个人记忆在development的Acc@1为16%，固定融合14%，群体8.5%。融合同时减少个人条目并加入群体证据，无法区分两个变化。新对照分别检验候选与记忆支持数值的对应关系、个人条目数量。下列选择在看新控制结果前固定，不以正确／错误样本筛选查询。

| 条件 | 变动 | 对照含义 |
|---|---|---|
|personal6|保留原personal的前6条，且须逐字段等于原融合中的个人行|与原personal比较条目预算；与原融合比较加入群体后的整体效果|
|personal_rebound|把原personal每行的候选ID换为确定性轮换ID，其他字符与行序保持|正确的候选—支持证据关联是否比错误关联更有用|
|fusion_population_rebound|只轮换原融合的群体行候选ID，个人行完全不动|在个人行相同时，正确群体关联是否优于错误关联|

两种轮换均按候选ID排序后循环平移，偏移量为`1 + int(sha256(protocol:user:trajectory:condition),16) % (N-1)`，对每行无固定点、映射一一对应。不使用目标、标签、预测是否正确或候选排名选偏移量。保留支持次数、占比、距最近支持天数、行顺序和来源名称。轮换只作用于记忆文本，不改候选、轨迹和标签；这是人工负对照，不是真实的新知识。

必须逐条核验两种轮换与各自原提示的真实聊天模板token数完全相等。不满足即停止，不尝试其他种子、补齐或裁剪。personal6是预算干预，长度可以减少，不称等token比较。因原个人记忆少于等于6条或群体为空而没有发生干预的查询仍保留完整分母，另报告空干预数量。

## 复用与成本

每阶段所有原用户各生成三个控制计划，共1,800条计划；原personal和fixed_fusion的1,200份响应复用。控制计划如果与同一阶段、同一用户和同一查询的旧提示messages逐字相同、token指纹相同，则复用对应旧响应，不重复调用。复用条件及响应文件SHA写入每条计划，不把复用输出计为新增调用。CPU预检封存实际需调用数，上限1,800。不得依据预测内容挑选缓存。

沿用原Llama3、权重元数据、模板、FP16、batch1、greedy、128输出上限和严格五ID解析。新runner须核对旧响应文件与已验收结果一致，处理生成和复用记录并分别计成本；不能直接调用只支持固定四条件的旧runner。旧脚本、输入和结果保持不变。

## 分析与决策规则

固定每阶段400／200分母，主指标Acc@1；Acc@5和严格NDCG@5完整报告。预定四组Acc@1比较：personal−personal_rebound、fixed_fusion−fusion_population_rebound、personal−personal6、fixed_fusion−personal6。各阶段内对四个精确不一致对检验做Holm校正；按用户配对bootstrap 10,000次，seed20261026，区间为逐项区间，不合并阶段或额外挑选显著分组。

格式失败及候选不覆盖仍计错，空干预仍在主分母中。非空干预子集只作描述；报告行数变化、真实token、格式成功率、新调用／复用数量。新结果必须在全部计划完成且退出码正常后独立复算。

正确对应关系优于错误对应关系只能支持模型对本干预敏感；不能证明全面推理或习惯变化适应。轮换可能改变语义合理性及两来源一致性。融合与personal6的比较也改变文本长度，不能单独声称纯群体信息效应。personal−personal6和fusion−personal6共同提供预算诊断，不能用不同显著性结论直接推导两种效应显著不同。

不显著不等于等效：报告区间与效应，不凭小样本零结果否定所有群体知识。即使发现正信号，也先检查是否有足够且可预测的互补空间；本批不解锁复杂路由或reserve。后续修改应另设版本并披露本批结果已观察。

## 执行门槛与当前状态

2026-10-26追加独立核验通过：[验证器](../scripts/validate_memory_controls.py)没有导入准备器、渲染器、runner或复算器。它从旧提示文本独立重建允许的变化，核对全部1,800条新提示、真实token指纹和复用规则，并验证2,400份旧响应的用户／查询／条件、源指纹及生成token解码；旧运行与外层退出码均为0。服务器耗时10.96秒、退出0，无标签读取、无模型权重加载、无新调用；全部输入与缓存统计与准备报告一致。[独立核验报告](../reports/memory_controls_validation_20261026.json)。七项合成测试包含四项验证器接受／拒绝测试，检查正确映射、数值和共享文本篡改、跨用户／伪缓存／长度错误、精确提示复用。独立性指代码实现与检查路径独立，不是另一研究者的盲审。

**当前剩余门槛：** 实现并验证支持这三个控制条件及旧响应复用的runner、完成后独立复算器；两者就绪后重新检查GPU再启动。本轮没有控制预测。以下准备器“待独立验证”是其冻结检查点的历史状态，不修改已封存报告。

2026-10-26 CPU预检退出0，耗时6.83秒、峰值RSS484,552KiB。全部1,800条计划通过准备器自检，最大输入1,635 token；两种轮换逐条等token检查均通过，无裁剪。fit新增1,054调用／复用146，development新增530／复用70，总计**1,584新增＋216复用**。personal6复用109份（fit74、development35），其中多数与群体为空时的旧融合提示相同，并不代表个人条目没有减少；personal6相对原personal的空干预仅fit2条。群体轮换空干预共107条，均复用旧融合。三项合成测试通过，未加载权重、未读取标签或逐人响应、0次新模型调用。[聚合预检报告](../reports/memory_controls_preflight_20261026.json)。这是准备器自检，独立核验仍待完成。

[配置](../configs/memory_controls_v1.json)与[CPU准备器](../scripts/prepare_memory_controls.py)已形成。准备器只读取已验收的无目标输入、旧提示及聚合验收报告，不读取标签文件和逐人响应、不加载权重。它检查共享输入、原提示可重建性、原输入指纹、轮换长度、前6行与融合一致性、缓存匹配。

后续仍需独立核验全部控制提示、控制runner与缓存复用逻辑、完成门控复算器，以及新任务启动前GPU资源检查。CPU自检与合成测试不等同于独立验证或真实生成成功。用户已授权日常实验推进，本次不新增付费服务、外部数据上传或其他任务资源权限；文献PPT的人工作决策继续等待。

准备命令（各环境路径由已有交接记录提供；输出必须不存在）：

```bash
python scripts/prepare_memory_controls.py \
  --input-root "$ACCEPTED_INPUT_ROOT" \
  --accepted-inputs reports/memory_diagnostic_inputs_20261025.json \
  --original-prompts "$SEALED_READER_PROMPTS" \
  --original-preflight reports/memory_reader_preflight_20261025.json \
  --original-results reports/memory_reader_results_20261025.json \
  --reader-config configs/memory_reader_v1.json \
  --config configs/memory_controls_v1.json --model "$LOCAL_LLAMA3" \
  --output "$NEW_CONTROL_PREFLIGHT"
```

独立核验命令（输出必须不存在；验证只读已有提示、原输入及旧响应，不调用模型）：

```bash
python scripts/validate_memory_controls.py \
  --input-root "$ACCEPTED_INPUT_ROOT" \
  --accepted-inputs reports/memory_diagnostic_inputs_20261025.json \
  --original-prompts "$SEALED_READER_PROMPTS" \
  --original-preflight reports/memory_reader_preflight_20261025.json \
  --original-results reports/memory_reader_results_20261025.json \
  --original-run "$COMPLETED_READER_RUN" \
  --reader-config configs/memory_reader_v1.json \
  --config configs/memory_controls_v1.json --model "$LOCAL_LLAMA3" \
  --prompts "$NEW_CONTROL_PREFLIGHT" --output "$NEW_VALIDATION_JSON"
```
