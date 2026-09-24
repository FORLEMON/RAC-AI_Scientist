# 18 项首测队列的运行与日志

2026-09-23 17:10 更新：用户取消全部 CORE-Bench 和全部 ARK，只保留 DiscoveryBench 的 EvoScientist / Agent Laboratory。ARK 补跑控制器及两个正在运行的 ARK 已停止，另 4 项待运行 ARK 已取消，取消项不再触发评分；日志、已有提交与历史评分保留。主队列中的 CORE-Bench 已全部结束，没有待启动项。Evo 的 DiscoveryBench 三项已结束，不自动重跑；Agent Laboratory 的 DiscoveryBench N0 继续，之后仍按 R1 → R3 执行、评分并记录错误。有效范围记录在 `runs/smoke-20260923/selection.json`，看板将取消项单独计数，下面的补跑与并发安排保留为历史记录。

2026-09-23 15:32 更新：用户要求在现有 Agent Laboratory 队列之外补跑 ARK。新增批次 `smoke-20260923-ark-recovery`，仅重做原批次 6 个模型调用为 0 的 ARK 启动失败项；历史状态和日志保留。每个 benchmark 仍按 N0 → R1 → R3、每项 USD 40、结束后独立评分、出错后继续。先增加 1 个 ARK；收到其真实模型响应后，Windows/WSL 整机 CPU 连续 60 秒低于 50%、WSL 可用内存至少 6 GiB，才准许第 2 个 ARK 并行。因此现在总任务并发最多为原队列 2 项 + ARK 2 项。若未达到准入条件，ARK 按单项继续执行。

ARK 补跑的 host / task 容器分别限制 1 CPU / 3 GiB、2 CPU / 2 GiB，原队列资源限制不变。CPU 持续达到 90% 的暂停保护在两个控制器中各自管理本批次容器；50% 是增加第 2 个 ARK 的准入门槛。补跑注册记录为原批次的 `ark-recovery.json`；每个原批次只允许注册一个补跑控制器。先前失败是 Docker 预建的空 `.conda_env` 挂载点被误判成已有实验，已在 CLI 和输入复制阶段修复；仍拒绝复用非空目录、普通未挂载目录及已有实验输出。两个真实无网络 Docker 检查通过，Windows 定向 36 项测试中 34 项通过、2 项跳过，Linux 关键测试 19 项全部通过。

原始批次：`smoke-20260923`，已于 2026-09-23 11:45（Europe/London）进入实验。该批次内部最多同时运行 2 项，按 EvoScientist → ARK → Agent Laboratory 顺序。每个 host 的 DiscoveryBench 和 CORE-Bench 两条队列并行，各自保持 N0 → R1 → R3；一项执行和评分结束后，该条队列才进入下一项。当前 host 的 6 项全部结束后才换下一个 host，因此某条队列先结束时可能只剩 1 项运行。保持原定任务、seed、房间和每项 USD 40 的 agent 预算。

每项使用独立 host 容器及任务容器，分别限制 2 CPU / 4 GiB 和 3 CPU / 3 GiB。两项合计最多使用 10 个 CPU 配额及 14 GiB 容器内存；这些是上限，不代表持续占用。任务容器从干净镜像启动，由 agent 自行安装任务依赖。host 完成、异常、超时或预算耗尽后，执行器关闭该项运行环境，调用独立评分容器并保存结果，然后运行下一项。评分报错也记录并继续；无有效提交会记录相应评分状态，不冒充成功。不会自动重试。

CPU 监控约每 5 秒读取一次 Windows 整机和 WSL CPU，取较高值；Windows 探测失败时记录异常并使用 WSL 数值。两项运行期间 CPU 达到 90% 且持续 20 秒时，暂停较后启动项的 host 与任务容器。降至 70% 以下持续 10 秒后恢复；每次暂停最多 20 秒，随后冷却 20 秒，持续过载时可再次暂停，以免原生模型网络连接或命令超时。暂停时间仍计入最长运行时间。采样、暂停和恢复事件均写入日志。

运行使用隐藏的 Windows `wsl.exe` 前台进程维持 WSL 生命周期，脱离交互终端。单独的 WSL systemd 服务未能维持 WSL 存活，最初启动在进入实验前被停止，随后改用此方式。电脑需要保持开机并避免睡眠；该队列没有配置开机自动恢复。

## 进度与输出

可在仓库目录的 PowerShell 中打开只读终端看板，每 5 秒刷新：

```powershell
.\scripts\watch_smoke_queue.ps1
```

只看一次使用 `.\scripts\watch_smoke_queue.ps1 -Once`。若 Windows 的脚本策略阻止执行，可仅对这次查看使用 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\watch_smoke_queue.ps1`，无需修改系统策略。按 Ctrl+C 只关闭看板，实验队列继续运行。

看板发现 `ark-recovery.json` 后，自动以补跑批次替换表格中的 6 行 ARK，合并显示两个控制器的活动项，并提示补跑批次、已开放的 ARK 并发数及心跳；原始失败记录仍完整保存在原批次。只看 ARK 补跑可用 `.\scripts\watch_smoke_queue.ps1 -Batch smoke-20260923-ark-recovery`。

看板列出计划中的全部 18 项，包括未开始项；汇总结束、运行、等待、正常完成、异常结束以及已评分数。费用/token/调用数直接读取各项 `model/usage.json`，因为运行中的 `status.json` 尚未写入最终 usage。`LastLog` 表示最新模型或控制台日志写入时间；不是任务完成百分比。`failed` 与分数分开显示：达到 hops/预算上限等异常结束后仍可能有有效提交和官方分数。`invalid_submission` 表示缺少有效提交，不是 0 分。CPU 与可用内存来自控制器采样；运行状态的心跳超过 30 秒未更新时，看板会提示检查控制器。

查看队列切换、评分结果和报错事件的实时流：

```powershell
Get-Content .\runs\smoke-20260923\controller.jsonl -Tail 20 -Wait
```

其中 `episode_started` 表示开始，`episode_finished` 表示本项执行与评分处理结束；同时检查 `host_exit`、`score_status` 与 `score`。深入排错时，根据 `episode_id` 打开下面对应目录。`host.stdout.log` 可能受原生 host 缓冲影响，暂时为空时可同时查看 `model/usage.json`、最新 `*.request.json` / `*.response.json` 以及 `task-runtime/`，不要只凭单个输出文件判断卡住。

- `runs/smoke-20260923/status.json`：当前阶段、每项开始/结束时间、退出码、评分状态、费用/token 计数、实际日志目录。
- `runs/smoke-20260923/controller.jsonl`：逐项开始、完成、异常、CPU 暂停/恢复及队列结束事件。
- `runs/smoke-20260923/resources.jsonl`：Windows / WSL CPU、WSL 可用内存及暂停任务的持续采样。
- `runs/smoke-20260923/launcher.stdout.log`、`launcher.stderr.log`：后台控制器启动输出。
- `runs/smoke-20260923/logs/<episode_id>/host.stdout.log`、`host.stderr.log`：host 完整控制台输出。
- 同目录的 `scorer.stdout.log`、`scorer.stderr.log`：独立评分器输出。
- 同目录的 `model/`：逐次请求、响应、错误和累计 usage，用于排查模型调用。
- 同目录的 `task-runtime/`：执行指令、返回结果和任务运行环境记录。
- `runs/smoke-20260923/episodes/<episode_id>/`：工作目录、原生 host 运行记录、提交和 `score.json`；`scoring/worker.stdout.log` 与 `worker.stderr.log` 保存官方评分子进程输出。

输出中的已知 API key、invite 和 runtime token 会脱敏。模型提示和响应本身保留用于 debug，日志与 secrets 均属本地私有文件，不应公开上传。

## 复现记录与验证范围

批次冻结代码副本、benchmark/upstream lock、配置及 TaskSpec hash、Docker 镜像实际 ID，保存在批次目录的 `controller-source/`、`frozen-plan.json` 和 `status.json`。当前批次不随仓库后续修改而改变 host 和评分代码。已有 episode 不自动重跑；重跑需要新批次与新的 R1/R3 房间。

启动前已完成真实模型与 judge 接口检查、SharedNet HTTPS 检查及 Linux 固定源码检查。2026-09-23 Windows 全部 236 项测试中 232 项通过、4 项平台或依赖相关跳过；Linux 队列测试 11 项通过，另有此前评分适配 17 项通过。已实际验证双容器暂停/恢复与 Windows CPU 采样。完整 OpenHands 模型及隔离终端预检通过，日志见 `runs/preflight/openhands-runtime-20260923-113546/`。科学任务的实际结果以运行输出为准。准备阶段的 PDF `output/pdf/rac-smoke-test-plan.zh-CN.pdf` 仍是启动前方案快照，当前执行顺序、并发和状态以本页及运行目录为准。

两个接入诊断批次 `runs/smoke-20260922/`（7 项启动尝试）及 `runs/smoke-20260922-live/`（3 项启动尝试）均已标记 `aborted_startup`，没有模型调用。原始日志及无有效提交的评分状态均保留。修复了 ARK provider 前缀、SharedNet 显示名称长度、Windows 挂载卷复制元数据所需的 FOWNER 权限，以及回滚重复执行的问题。三个真实 host 模型客户端均另外完成了极小的接口验证。

两个 ARK / Discovery R1、R3 房间中可能已有启动诊断的协议消息，未产生科学分析。正式批次给 episode ID 加入批次 hash 后缀；通信层忽略其他 episode ID 的协议消息，防止旧启动诊断进入模型上下文。其余分配不变。后续重新开展科学实验仍应新建独立房间。

启动入口为 `scripts/start_smoke_queue.ps1`，它启动隐藏 WSL 进程并记录 Windows PID；对已有状态的批次拒绝重复启动。当前批次已经启动时，不要再次执行。

## 配额问题已解除及历史诊断

2026-09-22 实际 Azure 响应显示 DeepSeek-V4-Pro 为 10,000 tokens/分钟、10 次请求/分钟，完整请求被 429 拒绝，证据见 `runs/preflight/azure-rate-limit.json`。用户调整后，2026-09-23 返回限额已变为 1,500,000 tokens/分钟、1,500 次请求/分钟，响应头证据见 `runs/preflight/azure-quota-20260923.json`。该探测同时发现 Azure 不接受 OpenHands 附带的 `prompt_cache_key`；模型代理移除该缓存提示后，完整模型与工具预检成功，随后启动本批次。

本批次通过 `scripts/start_smoke_queue.ps1` 启动。当前准备和验证记录见 `runs/launch-readiness.json`，实时执行状态见本批次的 `status.json`。已经运行时不要重复启动。

ARK 的 1.6 GB Conda 克隆在 Windows 挂载卷中严重受 I/O 限制；改为每项独立的匿名 Linux 数据卷后完整克隆通过。该卷只供 host 使用，不向任务容器预装科学依赖，host 退出时随容器清理。公开 OpenHands 扩展固定于 `47a8aa806d3ad2416ca3ee08c946558553413c83`，启动时经本地 Git 镜像加载。限流响应的 Retry-After 和限额头会保存；被明确拒绝的 429 不虚记 token 消耗或费用。
