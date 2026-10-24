# 四条件记忆读取 v1：提示、解析与解码冻结

2026-10-25（北京时间）。延续已验收的600条[诊断输入](MEMORY_DIAGNOSTIC_INPUTS.md)，本轮只固定读取接口并预检，未生成模型预测。配置：[memory_reader_v1.json](../configs/memory_reader_v1.json)；实现与严格解析器：[prepare_memory_reader.py](../scripts/prepare_memory_reader.py)；[预检聚合报告](../reports/memory_reader_preflight_20261025.json)。

## 固定的读取接口

模型为本地Meta-Llama-3-8B-Instruct，FP16、batch1、greedy、num_beams=1，种子20261024。使用原tokenizer的聊天模板与assistant生成起点，终止ID为128001／128009，pad=128001；最多生成128 token，输入与最大输出总长度不得超过8192，超长直接拒绝，不截断，不换候选。

四条件为context、personal、population、fixed_fusion，共用系统提示、近期轨迹和48候选。轨迹时间明确表示UTC，附UTC星期，避免将其误写成东京本地时间。候选字段为Cxx编号、类别、距最后地点的公里数；历史中能匹配候选的地点映射到相同编号，其余按出现次序标为不可选Hxx。原始用户、session及POI标识不进入模型文本，但保存在服务器私有索引中用于配对和审计。没有目标时间／类别／地点；本轮不打开标签文件。

附加记忆使用前轮冻结的文本，各条件只改这个区块；无记忆使用`(none)`。系统说明访问数、群体转移数、源内占比和最近支持时间的含义，但不要求优先选择某个来源。共享候选本来已包含两源检索，因此context不是无历史信息系统，条件比较针对额外证据读取。

输出要求为恰好5个不重复有效候选ID构成的JSON数组，按预测排名排列。解析对整个去首尾空白的响应做`json.loads`；Markdown、解释文字、非数组、长度错误、重复、非字符串、非候选ID一律记录失败和空预测，不提取局部片段，不重试。失败保留在全分母中并单列格式成功率，避免误将格式失败视为已证明缺乏记忆推理能力。后续若改解析器或提示，应新增明确标注的开发版本，不能覆盖本协议。

## 实际token预检

400名fit×4=1,600份提示，200名development×4=800份提示，共2,400份。使用服务器上的真实tokenizer和聊天模板；四条件共享文本一致性、源指纹和记忆token聚合核对通过；超长／裁剪0。耗时6.77秒，峰值RSS485,672 KiB，未加载权重，模型调用0，标签文件打开0，reserve查询0。

| 阶段 | 条件 | 总输入token | 平均输入token | 最大输入token |
|---|---|---:|---:|---:|
|fit|context|520,691|1,301.73|1,377|
|fit|personal|621,291|1,553.23|1,630|
|fit|population|585,835|1,464.59|1,630|
|fit|fixed_fusion|607,331|1,518.33|1,630|
|development|context|260,397|1,301.99|1,382|
|development|personal|310,957|1,554.79|1,635|
|development|population|292,922|1,464.61|1,607|
|development|fixed_fusion|303,697|1,518.49|1,607|

这些是正式待输入文本的token数，不含尚未生成的输出token；不能写成已经消耗的推理成本。最大输入1,635加128输出预算仍在8192上下文内。实际token不同，不能称四条件精确成本匹配。

四项单元测试通过：合法JSON保留排序；七类非法输出拒绝且不回退；用户/session/raw POI不进入文本且C/H映射正确；共享字段一致、目标附加字段与未来上下文拒绝。测试采用合成数据，不是新准确率结果。

## 复跑及下一步

Linux环境执行，需已验收两组输入、本地tokenizer、可信模型配置，输出目录必须不存在：

```bash
python scripts/prepare_memory_reader.py \
  --input-root "$DIAGNOSTIC_INPUT_ROOT" \
  --accepted-report reports/memory_diagnostic_inputs_20261025.json \
  --config configs/memory_reader_v1.json --model "$LOCAL_LLAMA3" \
  --output "$NEW_PREFLIGHT_OUTPUT"
python -m unittest discover -s tests -p test_memory_reader.py -v
```

预检最多2个CPU亲和核、nice19、16GiB进程地址空间、600秒CPU时间；实际外层timeout900秒，真实退出码0另保留。私有`fit_prompts.jsonl`和`development_prompts.jsonl`不得上传；报告保存文件字节指纹，私有提示文件内保存每条token序列指纹用于后续runner核对。

本轮GPU快照：0／1／2／3空闲显存约11,409／13,155／655／5,161 MiB，利用率83／27／0／99%，均不满足原门槛（空闲至少19,456MiB、利用率<10%）。这只是检查时状态，不是未来可用性的保证。本轮没有启动GPU任务或中断其他任务。

下一步实现并验证消费这些封存提示的runner与结果复算器，核验模型权重指纹、32GiB可用主机内存、项目GPU锁及即时GPU状态。优先GPU0，否则检查其他已授权卡；不因排队降低资源门槛。运行时必须保留原始响应、严格解析状态、调用成本与每个固定用户的失败记录。CPU预检完成不表示runner、推理或科学假设已验证。
