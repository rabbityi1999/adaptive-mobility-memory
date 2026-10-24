# 论文文字流程：东京200查询推理协议

状态更新（2026-10-24）：本轮已完成且退出0，600份预测独立复算通过；见[结果报告](TOKYO_PAPER_ORDER_RESULTS.md)。以下保留生成前协议与历史启动记录，不应据此重复启动。

2026-10-24，模型生成前固定。使用已验收的新输入、全城群体图和时间过滤证据，检验按论文文字筛选顺序重建后的框架表现。它仍保留72小时锚定分段、现代地址、本地Llama3接口和LLM前80%历史，不称论文全部配置已对齐。

## 条件与预算

同一200名用户各1条查询；清单SHA256 `fc9baee539c4456555dd537ab74623d9448e2385bcd9129a1aeaa89357a0a71e`。三个条件为原静态CKE框架、事件时间过滤CKE框架、LLM-ZS。每条查询两次WKG生成由两个框架条件共享，共400次WKG与600次预测，上限1000次调用。不复用旧队列输出，不按正确性重试。

模型固定Meta-Llama-3-8B-Instruct、FP16、greedy、batch1、seed100、PYTHONHASHSEED100、最多1000生成token。沿用GPT分词计数超过2000时保留最后6000字符的原裁剪；记录实际Llama输入长度和完整框架各模块保留程度。新队列与旧队列重叠178名用户、22条查询；不能视为独立验证集或将跨队列分差当作配对收益。

## 显式接口与预检

`run_llama3_baseline.py --input-protocol paper-order`调用独立adapter，接受新检查点真实状态，核验全部输入、图、bundle、查询截止点、真值隔离及9个固定官方文件哈希。默认`official-code`保留原入口；纸面流程条件要求同时启用静态CKE对照，禁止与旧消融缓存混用。

预检捕获400个真实WKG输入模板，以及600个实际Agent调用点的预测模板，不执行生成、不写预测。预测模板使用明确标记的WKG占位文本，不能把占位模板的截断或长度统计当作正式推理成本。实际WKG输出进入模型时，runner仍逐次检查上下文容量并记录实际提示及截断。

纸面流程的`--preflight-only`只检查CPU资源，不要求空闲GPU、不占项目GPU锁。正式运行仍要求至少19GiB空闲显存且利用率低于10%，优先GPU0，随后检查1/2/3；不降低门槛、不打断其他工作。正式加载前再次检查资源。CPU最多2核、nice19、6小时超时、CUDA分配器上限80%，沿用32GiB可用主存门槛。

## 分析

完成且退出0后运行既有 `validate_llama3_results.py`，对600份预测固定每条件200条分母独立复算Acc@1、Acc@5、严格NDCG@5及官方未截断NDCG。三个条件的三组配对比较均报告：用户bootstrap种子20261023、10,000次、描述性95%百分位区间；Acc@1精确双侧不一致对检验及三比较Holm校正。失败或空预测计错；共享WKG与各预测条件成本分列。不读取中间准确率用于调整协议。

## 启动模板

以下路径由操作者指向既有外部产物；源码目录中同时放置`paper_order_adapter.py`和`ablation_adapter.py`（后者仅提供模块保留诊断函数）。预检和正式运行使用不同新输出目录。

```bash
python scripts/run_llama3_baseline.py --launch --preflight-only \
  --input-protocol paper-order --include-official-cke \
  --upstream /path/to/pinned/upstream --inputs /path/to/paper_order_inputs \
  --causal-bundle /path/to/paper_order_graph/causal_bundle.json \
  --model /path/to/Meta-Llama-3-8B-Instruct --weight-manifest /path/to/weight_manifest.json \
  --token-cache /path/to/token_cache --gpu 0 --output /new/path/preflight
```

正式运行去掉`--preflight-only`，指定已通过资源门槛的GPU及新的输出目录。若资源不足，仅保留已通过的预检，不创建伪运行状态。

## 执行记录：2026-10-24

CPU预检已完成且退出0，600个实际预测调用点一致，无模型加载或模型调用。聚合证据见[预检报告](../reports/tokyo_paper_order_preflight_20261024.json)，原始提示清单保留在服务器，不上传。四项既有合成验收测试通过，覆盖固定分母、配对统计、未完成拒绝分析及内容篡改拒绝。

03:25正式启动前GPU0空闲24,121MiB、利用率0%，GPU1/3也空闲，GPU2有其他作业。按优先顺序使用GPU0，runner加载前再次检查资源。输出 `/mnt/xuyi/agentmove_paper_order_200_20261024`，supervisor1186569／worker1186571；源码在 `/mnt/xuyi/agentmove_paper_order_code_20261024`。runner SHA256 `21457a9949f291e5cf616a9adeb93ae6da2473793a92f3396d18bfb35bf185ab`，adapter SHA256 `63cb8d96750367063b496e0925aac177f2b3a6705cf81f4516499ba2a4278909`。启动后进程健康、正在加载权重，新预测指标待运行完成后核验。

完成且`exit.json`为0后执行：

```bash
python /mnt/xuyi/agentmove_paper_order_code_20261024/validate_llama3_results.py \
  --run /mnt/xuyi/agentmove_paper_order_200_20261024 \
  --bundle /mnt/xuyi/agentmove_paper_order_graph_20261023/causal_bundle.json \
  --output /mnt/xuyi/agentmove_paper_order_200_20261024/validated_summary.json
```

本轮不重新执行已完成的旧代码条件、消融、输入处理或群体图构建。后续先查该运行状态与退出码，避免重复启动。

03:27健康快照：权重已加载，首条查询完成4次模型调用，尚未完成该查询全部三条件；未见错误或fatal文件。该快照只确认已开始生成，不是最终分数。
