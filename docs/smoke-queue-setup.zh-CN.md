# 三个 host 的首测配置

2026-09-23：两个任务、六份实验配置、三份队列清单和十二个 SharedNet 房间已准备。用户提高 Azure 配额后，完整模型预检通过，批次 `smoke-20260923` 已启动；先 EvoScientist，再 ARK，最后 Agent Laboratory，最多同时运行 2 项。准备和验证记录见 `runs/launch-readiness.json`，实时进度见批次 `status.json`。本文保留房间与输入准备记录，运行和日志说明见 [队列运行说明](running-smoke-queue.zh-CN.md)。

## 已选任务

| Benchmark | Task ID | 输入准备情况 |
|---|---|---|
| DiscoveryBench | `nls_bmi/metadata_0/0/0` | 原始 train 中的 BMI / 时间偏好问题；现有 prepared bundle 的公开输入 hash 已验证 |
| CORE-Bench | `capsule-4180912` | Python / CPU 的代谢状况分类任务；按固定 HAL Hard 规则准备，两个数值问题 |

CORE capsule 官方下载地址为 `https://corebench.cs.princeton.edu/capsules/capsule-4180912.tar.gz`，压缩包 56,216 字节，SHA256 为 `7b445396c4528f23932c51d4f760023f7f4d2133cf09ede77a83aaa1ee2a9d16`。已检查公开脚本的依赖及代码，未运行其训练或复现脚本，因此 CPU 可行性来自静态筛选，实际耗时及兼容性尚未实测。

CORE 的 `dev` 是从固定 HAL test 中划出的本地开发分区；后续 eval 必须排除这个 capsule。私有参考 JSON、原始 capsule 和下载记录保存在 Git 忽略的 `upstreams/benchmark-data/corebench/`。提供给 host 的目录仅为 `prepared_tasks/corebench/dev/capsule-4180912/`，不含官方 Hard 规则排除的结果、环境定义和运行脚本。未为 agent 预装 capsule 依赖或预先修复原始代码。

## 配置及顺序

六份 `configs/<benchmark>.<host>.json` 已填入对应任务。每个 host 的清单为：

- `configs/queues/ark.smoke.json`
- `configs/queues/agent_laboratory.smoke.json`
- `configs/queues/evo_scientist.smoke.json`

每份清单顺序：Discovery N0 → Discovery R1 → Discovery R3 → CORE N0 → CORE R1 → CORE R3。固定 seed 0、每个条件运行一次、每个 episode 的 agent 费用上限 USD 40；其他预算见 [预算说明](queue-budget-settings.zh-CN.md)。清单包含配置 hash、prepared TaskSpec hash、独立评分所需的 controller 路径、每个 episode 的 SharedNet 配置路径和预定日志目录。全部路径以仓库根目录为基准。

清单保留准备时的 `queue_plan_only` 标记，由 `scripts/run_smoke_queue.py` 读取执行。实际运行按 EvoScientist → ARK → Agent Laboratory 分组；同一 host 的两个 benchmark 并行，各自依次执行 N0 → R1 → R3。每项结束或报错后尝试独立评分，再进入该 benchmark 的下一项；评分失败也继续，不自动重试。CPU 持续过载时短暂停下较后启动项，详见运行说明。`run-one` 本身不会读取队列清单。实际日志位置为 `runs/<batch>/logs/<episode_id>/`，取代清单中准备阶段的 `log_dir` 路径；实时状态中记录实际位置。

## SharedNet：12 个房间均已配置

用户已确认根目录 `.env` 中的现有房间尚未用于实验。它已复制到 **01：ARK / Discovery / R1**，根目录 `.env` 保持原样。用户补充的 11 个房间按提供顺序分配至 02–12；槽位 08 的完整 token 已补发并替换。12 个 Room ID 和 invite token 均互不重复、格式合法；未进行网络连接或加入房间。

| 槽位 | Host | Benchmark | 条件 | 准备时状态 |
|---|---|---|---|---|
| 01 | ARK | Discovery | R1 | 已填现有房间 |
| 02 | ARK | Discovery | R3 | 已填，格式合法 |
| 03 | ARK | CORE | R1 | 已填，格式合法 |
| 04 | ARK | CORE | R3 | 已填，格式合法 |
| 05 | Agent Laboratory | Discovery | R1 | 已填，格式合法 |
| 06 | Agent Laboratory | Discovery | R3 | 已填，格式合法 |
| 07 | Agent Laboratory | CORE | R1 | 已填，格式合法 |
| 08 | Agent Laboratory | CORE | R3 | 已填，格式合法 |
| 09 | EvoScientist | Discovery | R1 | 已填，格式合法 |
| 10 | EvoScientist | Discovery | R3 | 已填，格式合法 |
| 11 | EvoScientist | CORE | R1 | 已填，格式合法 |
| 12 | EvoScientist | CORE | R3 | 已填，格式合法 |

对应文件位于 `secrets/sharednet/smoke/`，文件名前缀就是槽位编号。N0 不使用房间。每个 R1/R3 episode 使用一个独立、未用于其他实验的房间；多个 agent 角色共享同一 episode 的房间。重新尝试一个已经启动过的 episode 时应换新房间。

后续需要更换房间时，可以使用以下格式提供信息：

```text
房间 02
ROOM=rom_完整RoomID
TOKEN=rit_完整InviteToken
```

也可以直接填写相应 `.env` 文件中的 `SHAREDNET_ROOM_ID` 和 `SHAREDNET_INVITE`。后者只放完整的 `rit_…` token，保留默认 `SHAREDNET_BASE_URL=https://www.sharednet.ai`；不需要 `clp_…` claim，也不需要发送模型 API key。invite token 只保存在 Git 忽略的本地 secrets 目录，不写入公开队列清单、评分输入或 prepared bundle。用户要求的 PDF 汇总只列 Room ID，不包含 invite token 或 API key。

## 当前验证范围与剩余工作

已验证：六份实验配置均通过离线配置检查；两项 prepared bundle 的文件 hash 一致；三个队列共 18 个 episode、12 个独立房间槽位；费用上限均为 USD 40；全部 12 个房间的 ID/token 格式合法且无重复；本地秘密文件被 Git 忽略。

启动前另已验证：DeepSeek-V4-Pro 和独立 gpt-5.4 judge 的真实接口、Linux 容器的模型代理与 SharedNet HTTPS 连通性、固定 benchmark 源码一致性。2026-09-23 Windows 完整测试 236 项中 232 项通过、4 项平台或依赖相关跳过；Linux 队列测试 11 项及此前评分适配测试 17 项通过。完整 OpenHands 模型与隔离终端预检、真实 Docker 暂停/恢复也通过。接口连通和单元测试不代表 18 项科学实验已经完成；结果、失败原因和评分均由运行目录持续记录。每项 R1/R3 在自身启动时加入对应房间。
