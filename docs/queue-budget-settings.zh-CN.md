# N0 → R1 → R3 的配置与预算来源

已为 DiscoveryBench 和 CORE-Bench 各保存三个宿主的配置，文件位于 `configs/<benchmark>.<host>.json`。每份只包含一个 host，条件顺序为 N0、R1、R3，默认 seed 为 0、repeat 为 1。2026-09-22 已分别选定 `nls_bmi/metadata_0/0/0` 和 `capsule-4180912`，两项 prepared 输入均已验证。三个 host 的队列清单位于 `configs/queues/<host>.smoke.json`。用户提高 Azure 配额并通过完整模型预检后，2026-09-23 已启动 `smoke-20260923`：EvoScientist → ARK → Agent Laboratory，每个 host 的两个 benchmark 并行，各自 N0 → R1 → R3。每项实验预算保持下表不变。详见 [运行说明](running-smoke-queue.zh-CN.md)。

| Host | 输出 token 总上限 | 模型调用总上限 | 最长时间 | max_hops | 费用上限 | 输入 token 总上限 | 来源 |
|---|---:|---:|---|---:|---:|---:|---|
| ARK | 1,800,000 | 1,500 | 43,200 秒（12 小时） | 23 | USD 40 | 100,000,000 | 除费用与 hops 外，沿用交接手册 §5.2 的旧 ARK Math_000 N0 Pro 队列记录 |
| Agent Laboratory | 1,300,000 | 900 | 21,600 秒（6 小时） | 8 | USD 40 | 60,000,000 | 除费用与 hops 外，沿用交接手册 §4.2 / README 示例，尚未核实为历史实际运行值 |
| EvoScientist | 1,300,000 | 900 | 21,600 秒（6 小时） | 8 | USD 40 | 60,000,000 | 除费用与 hops 外，暂借用上述示例，未找到该宿主的历史实际预算 |

这是配置中的上限，不是预计消耗。输出 token 数是一次完整 episode 的累计上限，不是单次模型请求的生成长度。相同 host/task/seed 下的 N0、R1、R3 使用同一组生命周期预算。

旧 ARK 记录的 hops 为 33，通用示例为 14；本次均以用户提供的最新截图为准，改为 ARK 23、另外两个 host 8。2026-09-22 用户指定 DiscoveryBench 和 CORE-Bench 均改为每个 episode USD 40，适用于全部三个 host 和 N0、R1、R3，覆盖此前 USD 70 / USD 25 的费用值。其余预算来自上述记录或示例，尚未针对 DiscoveryBench / CORE-Bench 校准。每份 JSON 的 `budget_provenance` 字段记录了证据类别和适用范围。

两个 benchmark 各选一个 task，三个 host 各跑 N0、R1、R3，seed 0、repeat 1，共 18 个 episode，配置的 agent 费用预算合计 USD 720；这不是预计花费，也不包含 DiscoveryBench 独立 judge 的费用。实际消耗和完成情况见批次状态及各项模型日志。

CORE 的选定 capsule 和私有参考数据已准备；每项的独立任务执行服务由执行器创建并清理。每次 R1/R3 使用独立 SharedNet 房间，12 个本地配置均已填写、格式合法且无重复。启动前已实际验证 `DeepSeek-V4-Pro` 和 Discovery judge `gpt-5.4` 接口，并验证 SharedNet HTTPS；房间在对应实验启动时加入。

队列模型代理按每百万输入 token USD 2、输出 token USD 4 做保守预算核算，忽略缓存折扣；此费率高于启动前查询到的 Azure 商业版 Global/DataZone Deepseek V4 Pro 零售费率。每个请求先预留费用和 token 额度，再按响应中的 usage 结算本地计数；费用、调用次数、token 或时间耗尽即结束 host、尝试评分并进入下一项。它是本地预算保护，并非 Azure 账单。定价证据和说明位于 `configs/queues/execution.local.json` 与 `runs/preflight/azure-v4-prices.json`。

密钥仍由根目录 `.env` 提供，不写入这些 JSON。CORE 配置采用已有本地环境验收报告中固定的任务镜像 ID。
