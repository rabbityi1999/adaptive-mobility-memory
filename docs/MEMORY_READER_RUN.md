# 记忆读取 v1：运行器、复算器与启动条件

2026-10-25（北京时间）。本轮实现运行代码并验证CPU路径，未启动GPU预测。[冻结接口](MEMORY_READER_V1.md)及输入保持不变；[runner](../scripts/run_memory_reader.py)、[独立复算器](../scripts/evaluate_memory_reader.py)、[执行前报告](../reports/memory_runner_preflight_20261025.json)。

## 已完成的验证

本地8项合成测试通过，覆盖严格解析、身份隐藏、时间／共享输入约束、未完成运行拒绝、固定分母、排名与NDCG、事后二选一上界、配对bootstrap与确切检验／Holm。服务器实际runner的`--preflight-only`路径退出0，外层退出0，耗时8.42秒，核验了全部2,400份提示、真实聊天模板token指纹、模型元数据和权重清单；未加载权重、0次模型调用、未读取标签。

权重核验沿用此前完整SHA256清单，本轮核对全部分片的大小与mtime及模型元数据SHA256，没有重新读取全部权重计算SHA256。清单指纹保存于报告。GPU生成分支尚待实际执行，CPU预检不等同于生成分支已经调通。

## 固定执行行为

- 顺序运行fit的400人四条件、development的200人四条件，共2,400次生成。每个人按context、personal、population、fixed_fusion顺序，使用已封存聊天提示；不根据fit响应修改development设置。
- tokenizer再次核对长度、每条token序列SHA256和提示文件SHA256。读取器不打开标签文件，不输出中间准确率。
- 模型为本地Llama3 FP16，batch1，greedy，128输出token上限，双EOS及严格JSON解析与冻结配置一致。使用独立GenerationConfig，避免继承checkpoint的采样参数；记录实际Torch／Transformers版本及attention实现。
- 获取既有项目GPU锁，并即时检查空闲显存至少19,456MiB、利用率<10%、主机可用内存至少32GiB。优先GPU0，其他获授权卡可在满足条件时使用。不会抢占、终止其他用户进程。
- 每次调用保存原始响应、生成token ID、解析结果／失败原因、耗时、输入输出token和来源指纹；格式失败无预测、不重试。执行异常停止整轮并保留失败证据，未完成结果不能作为固定分母的完整实验。输出目录必须不存在，不自动覆盖、重跑或断点续跑。
- GPU运行不设置RLIMIT_AS（CUDA虚拟映射与实际RAM不同），CPU亲和限定2核、nice19；外层最长6小时timeout21600秒。超时或中断没有成功完成凭证时，不进行分数复算。

## 独立复算

复算器不导入runner或其解析函数。先检查正常退出、completed状态、2400次调用及600用户记录、文件指纹，再打开标签。对每个固定查询和四条件核对身份／顺序、候选、源提示、token长度，独立严格解析原始响应并比对已存解析结果。

fit和development分别报告固定400／200分母的Acc@1、Acc@5、严格NDCG@5；格式失败和候选未覆盖均计错。另报告格式成功数及仅在候选命中子集上的条件性指标，后者不是主分数。三组预设Acc@1配对比较为personal-population、fusion-personal、fusion-population；每阶段内部Holm校正，按用户bootstrap10,000次，种子20261024。三个指标的差值区间均保存；不合并fit/development作确认性检验。个人／群体事后理想选择上界只用于诊断，不是可部署路由成绩。

## 启动与复跑

先使用新的输出目录运行CPU预检；`PROMPT_ROOT`为已封存的提示目录，冻结的prepare_memory_reader.py须与runner同目录，字节指纹必须匹配预检记录。

```bash
python scripts/run_memory_reader.py \
  --prompts "$PROMPT_ROOT" --preflight-report reports/memory_reader_preflight_20261025.json \
  --config configs/memory_reader_v1.json --model "$LOCAL_LLAMA3" \
  --weight-manifest "$WEIGHT_MANIFEST" --output "$NEW_CPU_CHECK" --preflight-only
```

资源满足后使用另一新目录，移除`--preflight-only`并指定实际GPU：

```bash
timeout 21600s python scripts/run_memory_reader.py \
  --prompts "$PROMPT_ROOT" --preflight-report reports/memory_reader_preflight_20261025.json \
  --config configs/memory_reader_v1.json --model "$LOCAL_LLAMA3" \
  --weight-manifest "$WEIGHT_MANIFEST" --output "$NEW_RUN" --gpu 0
```

保留外层退出码和日志。成功结束后才运行：

```bash
python scripts/evaluate_memory_reader.py \
  --run "$NEW_RUN" --input-root "$DIAGNOSTIC_INPUT_ROOT" --prompts "$PROMPT_ROOT" \
  --preflight-report reports/memory_reader_preflight_20261025.json \
  --accepted-inputs reports/memory_diagnostic_inputs_20261025.json \
  --config configs/memory_reader_v1.json --output "$NEW_VALIDATED_REPORT"
python -m unittest discover -s tests -p 'test_memory*.py' -v
```

只上传源代码与无身份聚合报告，私有提示、响应、生成token ID、标签及模型文件留在服务器。

## 本轮状态

最终资源快照：GPU0／1／2／3空闲显存分别12,021／12,021／655／8,730MiB，利用率78／75／0／0%，均不达启动门槛；主机可用内存147,150,336KiB满足要求。没有启动推理，无活跃本项目模型任务，暂无新四条件准确率。下轮先重查进程和即时资源；满足条件时执行已冻结协议，不重做输入或提示。PPT的人工作决策继续等待，不受本实验心跳影响。
