# 个人／群体记忆诊断 v1：冻结用户与输入规则

2026-10-25更新：下文保留冻结时的约定与历史状态。后续600条CPU输入已构建并通过独立验收，详见[输入验收](MEMORY_DIAGNOSTIC_INPUTS.md)。LLM提示／解码尚未冻结，未产生本诊断的新预测。

2026-10-24，在新目标抽取、候选覆盖检查和模型预测前固定。[机器可读协议](../configs/memory_diagnostic_v1.json)、[冻结检查点](../reports/memory_diagnostic_split_20261024.json)、[生成代码](../scripts/freeze_memory_diagnostic.py)。这是工作五的开发诊断，独立标注于已完成的AgentMove流程重建；论文轨迹计数差异仍待解释。

## 本轮已完成

在上一轮审计的1,633名用户中，以协议名称与原始用户ID的SHA256排序，固定**400名fit、200名development、1,033名reserve**。三组两两交集为0，合并完整覆盖库存，均不与已登记实验查询、供体和保留名单重叠。倒序输入仍生成相同分组。

fit/development各选每人第一条验证session，合计600条session标识；只读取已有session元数据，不打开签到行、不抽取目标地点。未来输入构建时，取该session最后一次签到为目标，之前最多5次签到为近期上下文，倒数第二次签到时间为查询截止点。目标时间、类别和地点不进入模型输入。这延续回顾性session预测定义，不能声称完全依赖过去信息的在线入组。

reserve只封存身份，本轮没有为其选择session或目标。身份和session清单保存在工作区`research_workspace/agentmove_reproduction/strict_official/memory_diagnostic_v1_20261024/private_split.json`，不上传仓库；其字节指纹与各组指纹写入公开检查点。上轮已核验全时段资格，因此本组也不是全新、未经处理的原始数据。

## 时间边界与隔离

- 个人长期记忆只使用本人的train session，且事件严格早于所选验证session起点。近期上下文另列，各条件相同。
- 群体源只使用其他用户的train session，排除本人及全部reserve用户。贡献session的结束时间和每条事件时间均须严格早于当前查询截止点；不接入任何人的validation/test事件。转移统计只连接同一个train session内相邻签到，不跨session拼接。
- 城市地址／类别仍沿用已有资源，保留现代地图语义的局限；动态频次、最近时间和转移证据必须从允许历史重建，不能直接复用原静态图。
- fit只用于管线开发，development用于冻结实现后的诊断。各用户的验证session日期可能交错；此划分不自动允许将较晚fit标签训练出的模型应用于较早development查询。本版不训练路由；若加入监督模型，必须另核验统一训练标签截止点。
- 在输入构建和核验通过前，不运行LLM。保留用户不得用于构建群体记忆、调参或挑选结论；后续启用需要单独冻结评估规则。

## 候选与记忆预算

候选目录只来自允许的个人与群体历史，不强行添加真实答案。四个有序队列为个人访问频次、从当前地点出发的群体转移频次、距当前地点最近的历史POI、群体访问频次；按这个顺序轮转取一个未重复地点，跳过耗尽队列，最多96个。频次并列按最新支持事件时间、原始POI ID排序；距离采用经纬度球面距离并以POI ID破同分。

预先固定报告前24／48／96个候选的覆盖率，**正式读取诊断只用前48个**，不按覆盖结果更换K。不足48个时保留实际数量，不按目标补齐。候选展示顺序按协议、用户、session和POI ID的哈希排序，各条件完全一致。目录外／候选外目标仍保留在总分母。

比较context、personal、population、fixed_fusion四条件。近期轨迹、候选类别和距离字段一致；没有WKG调用。单源记忆最多12行、1024个Llama3 tokenizer token；融合分别最多6行，不用另一源补足空缺。行字段固定为候选ID、支持次数、源内占比、距最新支持事件的天数；个人支持为访问数，群体支持为从当前地点出发的转移数，占比分母分别为该源允许历史中的全部访问／当前出发转移数。只显示固定候选内的证据。

超预算按预定次序删除末尾整行；融合先交替排列个人与群体行。不做整份提示的尾部字符截断。实际行数和token可能不同，因此比较的是同一预算上限，不称实际token完全匹配。共享候选本身已经使用两类历史；条件差异研究的是**候选固定后额外证据的读取**，不是纯粹隔离两种信息的整个预测系统。

## 下一步验收与分析

下一轮先构建CPU输入和候选，先检查fit管线，development按冻结实现处理。独立核验：目标遮蔽、本人／reserve排除、所有支持时间、候选目录、行预算及同一查询四条件候选一致性。此阶段不启动LLM；后续须在生成前固定提示、输出解析和解码设置，保持模型一致。

开发诊断报告全分母Acc@1/5、NDCG@5、目录覆盖、候选覆盖，以及仅在候选命中子集内的排序结果；后者只能作为条件性诊断。配对单位为用户，bootstrap种子20261024、10,000次。正式读取条件预设Acc@1比较为personal与population、fusion与personal、fusion与population，三项Holm校正；context作为参考。先记录互有独占命中的数量和事后理想选择上限；上限不能充当可部署方法成绩。是否值得研究路由，由开发证据及预测前特征可识别性决定，不预设一定有效。

这600条开发查询预算不是统计效能证明，也没有证明习惯变化存在。单次来源读取诊断通过后，才考虑按时间连续的习惯变化与更新实验；已观察开发结果不能再称独立验证。

## 复跑冻结步骤

```powershell
python scripts/freeze_memory_diagnostic.py --config configs/memory_diagnostic_v1.json --inventory-report reports/development_user_inventory_20261024.json --private-inventory /private/path/private_users.json --session-manifest /private/path/paper_session_manifest.csv --output /new/path/split_report.json --private-output /private/new/path/private_split.json
```

输入库存须为修正后1,633名版本；脚本检查排除集合、session指纹、分组及输出清单指纹，拒绝覆盖已存在输出。后续如改变规则，新增协议版本并记录原因，不覆盖当前冻结文件。
