# 全部 DiscoveryBench 实验数据

范围：仓库内四个 smoke 批次的全部 DiscoveryBench 运行与计划记录，包含历史启动失败、中断和取消。不是 DiscoveryBench 官方全量题库。

共 30 条计划记录：17 次实际启动，2 项启动前取消，11 项随批次中止而未启动。共有 5 项有效评分。

已落盘模型调用合计 1,355 次，保守费用估算 $105.0752，不含 judge；ARK 取消运行可能存在未计入的在途用量。

| 批次 | Host | 条件 | 状态 | 分数（0–1） | 调用次数 | 耗时（分钟） |
|---|---|---|---|---:|---:|---:|
| smoke-20260922 | ark | N0 | failed | — | 0 | 0.2 |
| smoke-20260922 | ark | R1 | failed | — | 0 | 0.2 |
| smoke-20260922 | ark | R3 | failed | — | 0 | 0.2 |
| smoke-20260922 | agent_laboratory | N0 | interrupted | — | 0 | 3.9 |
| smoke-20260922 | agent_laboratory | R1 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922 | agent_laboratory | R3 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922 | evo_scientist | N0 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922 | evo_scientist | R1 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922 | evo_scientist | R3 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922-live | ark | N0 | failed | — | 0 | 0.4 |
| smoke-20260922-live | ark | R1 | failed | — | 0 | 0.4 |
| smoke-20260922-live | ark | R3 | interrupted | — | 0 | 5.2 |
| smoke-20260922-live | agent_laboratory | N0 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922-live | agent_laboratory | R1 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922-live | agent_laboratory | R3 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922-live | evo_scientist | N0 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922-live | evo_scientist | R1 | not_started_batch_aborted | — | 0 | — |
| smoke-20260922-live | evo_scientist | R3 | not_started_batch_aborted | — | 0 | — |
| smoke-20260923 | evo_scientist | N0 | completed | 0.000000 | 41 | 7.8 |
| smoke-20260923 | evo_scientist | R1 | failed | — | 134 | 52.5 |
| smoke-20260923 | evo_scientist | R3 | failed | 0.218182 | 446 | 124.3 |
| smoke-20260923 | ark | N0 | failed | — | 0 | 0.2 |
| smoke-20260923 | ark | R1 | failed | — | 0 | 0.2 |
| smoke-20260923 | ark | R3 | failed | — | 0 | 0.3 |
| smoke-20260923 | agent_laboratory | N0 | completed | 0.000000 | 182 | 180.0 |
| smoke-20260923 | agent_laboratory | R1 | completed | 0.107143 | 179 | 74.4 |
| smoke-20260923 | agent_laboratory | R3 | completed | 0.000000 | 177 | 111.2 |
| smoke-20260923-ark-recovery | ark | N0 | cancelled | — | 196 | 95.2 |
| smoke-20260923-ark-recovery | ark | R1 | cancelled | — | 0 | — |
| smoke-20260923-ark-recovery | ark | R3 | cancelled | — | 0 | — |

说明：
- 未评分留空，不改为 0。计划记录不等于实际运行；批次名与 episode_id 共同构成唯一键。
- 计账 token 包含失败请求的保守预留，费用非 Azure 账单，不包含 judge。
- ARK 恢复 N0 为取消时最后落盘用量，不能保证覆盖所有正在进行的供应商调用。已有 discovery_result.json 也不等于经过验证或评分。
- 历史失败/恢复尝试不是新的独立 seed；所有有效结果仍来自一个任务、seed=0 的 smoke test。

- EvoScientist R3 有评分但 host 异常退出；Agent Laboratory R3 的 episode 原始终态为 stop，退出码 0。
- `results.csv` 为一行一条计划/运行记录，`results.json` 含详细状态、预算、时间、调用、token、费用、结束原因及版本。
- `batches/` 按批次保留评分、episode、工作区分析代码与输出、模型/host/scorer/任务运行日志、OpenHands 会话日志和冻结控制器源码。
- `file-manifest.json` 记录每份原始/导出文件哈希与排除项。凭据、环境文件、依赖缓存及不适合安全导出的二进制状态快照不在数据包中。
- 未启动项目没有 episode 目录或日志。没有补造 score.json，没有重跑或补评分。
