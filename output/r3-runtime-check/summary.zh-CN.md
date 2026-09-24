# R3 调用次数与耗时检查

统计时间：2026-09-23T20:08:48+0100。仅检查已有运行和日志，未重跑或修改实验。

| Host | Benchmark | 已结算调用数 | 总耗时（分钟） | 模型请求耗时（分钟） | 代码执行耗时（分钟） | 状态 |
|---|---|---:|---:|---:|---:|---|
| EvoScientist | discoverybench | 446 | 124.3 | 115.4 | 8.6 | 预算保护停止；结果已评分 |
| EvoScientist | corebench | 148 | 76.9 | 42.5 | 32.9 | 达到 hops 上限；结果已评分 |
| Agent Laboratory | corebench | 33 | 10.5 | 4.8 | 0.0 | 达到 hops 上限；无有效提交 |
| Agent Laboratory | discoverybench | 121 | 63.7 | 47.1 | 11.0 | 运行中；尚未评分 |

ARK 两个 benchmark 的初始 R3 启动即失败，调用数均为 0；补跑批次的 R3 在启动前由用户取消，不计作真正开展的实验。

## 当前 Agent Laboratory / DiscoveryBench / R3

已结算 121 次模型调用，统计时另有 1 次请求等待返回；已结算请求未见网关错误文件。模型请求平均 22.0 秒，中位数 5.8 秒，95% 分位约 129.1 秒，最长已结束请求约 208.4 秒。

按请求/响应文件时间近似分解：模型请求约 47.1 分钟（包含统计时仍在进行的请求），代码执行约 11.0 分钟，其余启动、协调等未覆盖间隔约 5.6 分钟。时间窗口有并发时取并集，属于日志时间近似值，不是 CPU profiling。

代码执行 16 次，其中 11 次失败：

- 000001：0.44 秒；ModuleNotFoundError: No module named 'pandas'
- 000003：0.63 秒；FileNotFoundError: [Errno 2] No such file or directory: 'data/processed/nls_bmi_cleaned.csv'
- 000004：0.79 秒；NameError: name '__file__' is not defined. Did you mean: '__name__'?
- 000006：0.87 秒；ModuleNotFoundError: No module named 'numpy'
- 000007：1.22 秒；ValueError: Matrix is singular
- 000008：0.80 秒；ModuleNotFoundError: No module named 'numpy'
- 000009：591.53 秒；执行超时
- 000010：0.82 秒；ModuleNotFoundError: No module named 'numpy'
- 000011：0.98 秒；ModuleNotFoundError: No module named 'pandas'
- 000015：0.71 秒；SyntaxError: f-string: unmatched '('
- 000016：0.90 秒；ModuleNotFoundError: No module named 'numpy'

当前 coordination.jsonl 最后一个调度决定是 hop 3 的 running_experiments：已通过文献、方案和数据准备阶段，正在执行实验；尚未完成最终提交或评分。这里的 hop 是外层调度步骤，单个步骤内可发起很多模型请求，调用次数不是完成百分比。

同 host、同 DiscoveryBench 的历史基线：

- N0：182 次模型调用，180.0 分钟。
- R1：179 次模型调用，74.4 分钟。

定位结论：当前耗时主要来自模型请求与代码反复修正。任务执行日志多次显示缺少 pandas/numpy、缺少中间文件、语法错误等，另有一次约 591.5 秒的执行超时。统计时仍在持续推进，不能据此给出可靠剩余时长。

## 数据来源

- `runs/smoke-20260923/status.json`
- `runs/smoke-20260923/logs/<episode_id>/model/usage.json` 与逐次 request/response/error 文件
- `runs/smoke-20260923/logs/<episode_id>/task-runtime/` 的 request/result 文件
- `runs/smoke-20260923/episodes/<episode_id>/coordination.jsonl`

本目录 statistics.json 保留完整统计。模型调用数包含失败的模型请求；代码执行失败次数是另一种计数。费用字段只是本地保守估算，不是 Azure 账单，且不含 judge 费用。
