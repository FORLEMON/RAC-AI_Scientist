# DiscoveryBench smoke test 结果

批次：smoke-20260923。六项已结束，评分处理已完成。任务 nls_bmi/metadata_0/0/0，train / real-no-domain-knowledge，seed=0。

| Host | 条件 | 分数（0–1） | 运行 / 评分状态 | 耗时（分钟） | 模型调用 | 保守费用（USD） |
|---|---|---:|---|---:|---:|---:|
| evo_scientist | N0 | 0.000000 | completed / scored | 7.8 | 41 | 2.5950 |
| evo_scientist | R1 | —（无效提交） | failed / invalid_submission | 52.5 | 134 | 19.3577 |
| evo_scientist | R3 | 0.218182 | failed / scored | 124.3 | 446 | 39.0427 |
| agent_laboratory | N0 | 0.000000 | completed / scored | 180.0 | 182 | 10.5116 |
| agent_laboratory | R1 | 0.107143 | completed / scored | 74.4 | 179 | 7.0015 |
| agent_laboratory | R3 | 0.000000 | completed / scored | 111.2 | 177 | 11.5084 |

模型：DeepSeek-V4-Pro。已评分五项的 judge：gpt-5.4；Evo R1 在提交验证阶段结束，未调用 judge。

六项模型调用合计 1,159 次；保守费用合计 $90.0169（不含 judge）。

| Host / 条件 | 预算计账输入 token | 预算计账输出 token | 成功响应输入 token | 成功响应输出 token |
|---|---:|---:|---:|---:|
| evo_scientist / N0 | 1,265,358 | 16,065 | 1,265,358 | 16,065 |
| evo_scientist / R1 | 9,482,268 | 98,302 | 8,002,086 | 73,726 |
| evo_scientist / R3 | 19,139,854 | 190,737 | 17,826,223 | 166,161 |
| agent_laboratory / N0 | 3,916,908 | 669,438 | 3,788,873 | 472,830 |
| agent_laboratory / R1 | 2,894,915 | 302,924 | 2,743,789 | 237,388 |
| agent_laboratory / R3 | 4,836,427 | 458,897 | 4,390,749 | 327,825 |

## 结束原因与解释

- **evo_scientist / N0**：host 正常退出，原生工作流报告完成；提交有效并评分。 
- **evo_scientist / R1**：模型响应被 Azure content_filter（Jailbreak 标签）阻断；规定路径缺少有效 discovery_result.json，未评分。 outputs/discovery_result.json 是中间文件，并非评分器要求的根目录提交；保留调试，不视为有效提交。
- **evo_scientist / R3**：控制器以退出码 125 停止 host；依据冻结控制器代码和剩余预算，判断为模型网关保守预算预留保护触发。已有提交获得评分。 该原因由退出码与冻结代码推断，未记录独立的超限错误文件；episode.json 仍为 initializing/pending，最终运行状态取控制器 failed，提交有效性取 score.json。费用未达 $40，也可能因下一次请求预留额超过余额而停止。
- **agent_laboratory / N0**：host 正常退出，原生工作流报告完成；提交有效并评分。 
- **agent_laboratory / R1**：host 正常退出，原生工作流报告完成；提交有效并评分。 
- **agent_laboratory / R3**：host 报告终止状态 stop（host reports a terminal state），退出码 0；控制器标记 completed，提交有效并评分。 保留原始 stop 语义，不等同于原生工作流声明 completed。

## 数据口径

- 每个条件只有一个任务、seed=0、一次重复的 smoke test，不能据此判断框架总体优劣。
- elapsed_seconds 是控制器记录的单项总耗时，包含启动、host、清理及评分；不是纯模型或纯分析耗时。
- budget_accounted token 包含失败请求的保守预留；successfully_reported 仅是成功响应中服务端 usage 的合计，不能代表所有请求的真实账单。
- 费用按输入 $2/百万、输出 $4/百万估算，缓存按普通输入计算；非 Azure 账单，不含 judge 费用。
- 失败请求（明确的 429 除外）按完整预留计账；model_calls 含失败尝试。
- 运行状态、原始 episode 状态、提交有效性和评分状态分别保存。原始分析代码/输出未经重新运行或科学正确性验证。

每项预算：USD 40、输入 60,000,000 token、输出 1,300,000 token、模型调用 900 次、最长 21,600 秒（6 小时）、max_hops=8。

## 文件与版本

- `results.json`：完整记录、模型与 judge 用量、预算、时间、结束原因及代码版本。
- `results.csv`：一行一个实验；空分数代表无效提交，不代表 0 分。
- `discoverybench-debug.zip` / `debug/`：六项原始评分、episode 元数据、提交、分析代码/输出、模型/host/scorer/任务运行日志及冻结控制器源码。
- `file-manifest.json`：原始与导出文件 SHA-256、是否脱敏，以及排除项。`package-manifest.json` 覆盖完整交付文件。
- `validation.json`：六项完整性、原始分数/提交哈希、用量对账和敏感信息检查。

仅导出允许的文本数据与图像；.env、凭据存储、依赖/缓存、pickle 状态快照及一个 Arrow 中间数据文件未导出，原文件保留在运行目录。
Git HEAD（导出时）：`05a9184f15ef65a1086709e2118b96a16d5f7cf7`，工作区有未提交修改；不能仅凭 HEAD 复现运行代码。
控制器源码树 SHA-256：`2b901260bedef74f44cb6c209b152de10138d50db52c3a54a008e73f1c212015`。实际运行 integration 哈希、上游 host/benchmark revision 和镜像 ID 见 JSON。
