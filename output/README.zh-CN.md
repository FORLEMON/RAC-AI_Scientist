# 实验数据与日志

## DiscoveryBench 全部记录

- [完整数据包 ZIP，约 65 MB](discoverybench-all-results/discoverybench-all-data.zip)
- [中文说明与全部记录](discoverybench-all-results/summary.zh-CN.md)
- [CSV 汇总表](discoverybench-all-results/results.csv)
- [完整 JSON](discoverybench-all-results/results.json)
- [文件清单与原始/导出 SHA-256](discoverybench-all-results/file-manifest.json)
- [压缩包 SHA-256](discoverybench-all-results/archive-sha256.txt)
- [校验记录](discoverybench-all-results/validation.json)

覆盖四个批次、30 条计划记录：17 次实际启动、2 项启动前取消、11 项随批次中止而未启动。CSV 中筛选 `was_started=true` 可只查看实际启动的尝试。五项有有效数值评分；其他评分为空，不代表 0 分。

解压 ZIP 后，日志在 `batches/<批次>/logs/<host>-<条件>/`，包括模型请求与响应、host、scorer、任务运行日志，以及 ARK 的 OpenHands 会话日志。`batches/<批次>/episodes/<host>-<条件>/` 包含原始评分、episode 元数据、提交、分析代码与输出及协作记录。未启动的项目没有 episode 或日志目录。

仓库直接保存汇总表和完整 ZIP；展开的数据目录和重复的六项 ZIP 保留在本地，避免重复存储。下载 ZIP 并解压即可取得完整日志。数据包排除了 `.env`、API key、SharedNet invite/member token、凭据存储、依赖缓存和不适合安全导出的二进制状态快照，详见文件清单中的排除记录。

## 本轮保留的六项结果

- [EvoScientist 与 Agent Laboratory 各 N0/R1/R3 的报告](discoverybench-results/summary.zh-CN.md)
- [六项 CSV](discoverybench-results/results.csv) / [六项 JSON](discoverybench-results/results.json)
- [评分与异常结果审计](benchmark-score-analysis/analysis.zh-CN.md) / [审计数据](benchmark-score-analysis/audit.json)

六项报告中的原始本地路径描述了运行时目录；完整产物统一由上方的全量 ZIP 提供。审计脚本保留了本机原始运行目录的读取路径，原始审计结果已随仓库提交。

实验模型为 DeepSeek-V4-Pro，五项有效评分的 judge 为 gpt-5.4。费用是本地保守估算，不是 Azure 账单，且不包含 judge 费用。ARK 被取消的恢复运行仅记录最后落盘的用量，可能未包含在途请求。

所有有效结果来自 `nls_bmi/metadata_0/0/0`、seed=0 的单任务 smoke test；历史失败和恢复尝试不能当作独立重复样本或用于直接比较总体性能。
