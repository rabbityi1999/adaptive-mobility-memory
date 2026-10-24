# 记忆诊断 v1：CPU 输入构建与独立验收

日期：2026-10-25（北京时间）。本阶段延续[冻结协议](MEMORY_DIAGNOSTIC_V1.md)，不训练模型，不运行LLM，不修改400／200／1033分组、候选K或记忆预算。

## 实现与证据

- [构建器](../scripts/build_memory_diagnostic_inputs.py)的检索函数只接收观测前缀及允许历史，目标事件在检索完成后单独提取；输入、标签、来源审计分别保存。
- 个人记忆来自本人train；群体记忆来自其他人的train，严格过滤查询时间，排除全部reserve用户；转移不跨session。
- 共用候选来自个人频次、群体转移、地理距离、群体频次四个队列，按冻结顺序去重轮转；读取固定48个，24／48／96仅用于覆盖诊断，不按结果选择K。
- [独立验证器](../scripts/validate_memory_diagnostic_inputs.py)不导入构建器。全部查询复算源行指纹、时间与身份排除、目录、标签与上下文、记忆内容和token预算、覆盖计数；每组首／中／末三个查询独立重建完整候选队列及属性。不能将抽查写成所有候选队列全量独立重建。
- 每次运行最多使用两个CPU亲和核，nice19，16GiB进程虚拟地址空间上限、1500秒CPU时间上限，外部timeout1800秒；这些是本进程限制，不是管理员部署的全服务cgroup总资源配额。CUDA不可见，仅加载本地tokenizer。
- development必须读取fit通过凭证，并验证fit结果指纹、构建器与配置指纹相同；未通过不得向后推进。

## 复跑

使用已有AgentMove环境的Python（pandas、numpy、transformers及本地Llama3 tokenizer）；Linux进程资源接口是必需依赖。不要使用Python `-O`。先准备冻结协议、清单与可信pickle。以下变量均为操作者设置的路径，不包含凭据；`RUN_ROOT`应是新建输出父目录，单阶段输出必须不存在。

```bash
python scripts/build_memory_diagnostic_inputs.py \
  --inputs "$INPUT_ROOT" --config configs/memory_diagnostic_v1.json \
  --split "$PRIVATE_SPLIT" --split-report reports/memory_diagnostic_split_20261024.json \
  --model "$LOCAL_LLAMA3" --output "$RUN_ROOT/fit" --phase fit
```

实际运行外层须使用`timeout 1800s`，将真实进程退出码写入`$RUN_ROOT/fit.exit.json`，结构为`{"exit_code":0}`（0仅可在成功退出后记录），保留标准输出与错误日志。然后独立验证：

```bash
python scripts/validate_memory_diagnostic_inputs.py \
  --run "$RUN_ROOT/fit" --inputs "$INPUT_ROOT" \
  --config configs/memory_diagnostic_v1.json --split "$PRIVATE_SPLIT" --model "$LOCAL_LLAMA3"
```

确认验证器退出0且生成`fit/validation.json`后，执行相同构建器：

```bash
python scripts/build_memory_diagnostic_inputs.py \
  --inputs "$INPUT_ROOT" --config configs/memory_diagnostic_v1.json \
  --split "$PRIVATE_SPLIT" --split-report reports/memory_diagnostic_split_20261024.json \
  --model "$LOCAL_LLAMA3" --output "$RUN_ROOT/development" --phase development \
  --accepted-fit "$RUN_ROOT/fit"
```

按同样方式记录development真实退出码，验证器的`--run`改为`$RUN_ROOT/development`。失败日志不覆盖；不要伪造退出或验收凭证。私有`inputs.jsonl`、`labels.jsonl`、`audit.jsonl`和身份清单不上传；公开聚合结果和来源指纹。

## 后续边界

候选覆盖不是预测准确率；这两个开发组的覆盖差异也不是方法改进的配对证据。共享候选已使用两种历史，后续四条件只比较额外证据读取。实际记忆长度不同，本轮只是统一上限，不是精确token成本匹配。reserve仍未选择预测目标。生成之前还需固定提示、候选输出解析与解码设置、核对实际输入长度并检查GPU状态。

## 已验收结果

两组构建及独立验收均退出0，源代码／配置指纹一致。[聚合结果与核验凭证](../reports/memory_diagnostic_inputs_20261025.json)保留构建时的pending状态及后续通过凭证，不覆盖原始构建记录。全部600条输入验收完成，0次模型调用；1,033名reserve仍未选查询。

| 阶段 | 用户／查询 | 目录覆盖 | Recall@24 | Recall@48 | Recall@96 | 群体记忆为空 |
|---|---:|---:|---:|---:|---:|---:|
|fit|400|364（91.00%）|198（49.50%）|231（57.75%）|258（64.50%）|72|
|development|200|170（85.00%）|93（46.50%）|102（51.00%）|116（58.00%）|35|

development构建耗时131.91秒，峰值RSS 1,219,780 KiB；独立验收46.94秒。额外记忆token合计context=0、personal=50,960、population=32,855、fixed_fusion=43,700；有记忆条件单查询最大均255 token。两组个人和融合记忆均无空查询。development未按fit覆盖结果调整检索参数。

fit的400条输入及独立验证均退出0。目录覆盖364/400（91.00%），24／48／96候选覆盖198／231／258条（49.50%／57.75%／64.50%）。群体转移记忆为空72条，个人记忆为空0条；不把空源补齐为其他来源。

fit构建耗时264.51秒，峰值RSS 1,225,168 KiB；独立验收89.19秒。四条件额外记忆token合计context=0、personal=101,400、population=65,800、fixed_fusion=87,440；各有记忆条件单查询最大255 token。这不包含共用上下文、候选、聊天模板或后续生成token。

这些是冻结协议下的新开发输入诊断，与旧400人pilot的用户、session和检索定义不同，不将57.75%与旧27.5%相减宣称检索方法提升。尚无四条件的新LLM准确率。
