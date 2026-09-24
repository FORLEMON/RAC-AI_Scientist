# DiscoveryBench / CORE-Bench 适配与验收

代码提供任务准备、三个宿主的输入/输出协议、独立评分入口，以及 CORE Hard 的任务执行容器。RCB 原有命令默认行为保持兼容。**离线验证不能代替 Linux/Docker 和真实模型验收**；当前没有新 benchmark 的真实 episode 分数，也没有宣称 24 个宿主/条件组合跑通。

2026-09-22 本地验证：217 项测试，215 通过，2 项因 Windows 的符号链接权限 / Linux Bash 进程组测试条件跳过；包含固定上游源码的官方过滤与评分 smoke。真实 `nls_bmi/metadata_0/0/0` train 任务已通过 CLI 准备及公开输入 hash 检查。没有调用付费模型；本机没有 Docker，因此容器 smoke 未运行。

## 固定协议

| 项目 | DiscoveryBench | CORE-Bench |
| --- | --- | --- |
| 来源 | allenai/discoverybench，见 benchmark.lock.json | princeton-pli/hal-harness，见 benchmark.lock.json |
| 任务范围 | real，no-domain-knowledge | HAL Hard，首批选择 Python、CPU capsule |
| 分区 | 原始 train / test | 从 HAL test 集人为冻结 dev / eval，必须 capsule 不重叠 |
| 任务 ID | topic/metadata_N/group-index/query-index | capsule_id |
| 可见输入 | 当前 question、数据文件、dataset/原始列说明 | 官方 Hard 过滤后的 capsule、task_prompt、问题字符串 |
| 隐藏内容 | hypotheses、true_hypothesis、workflow、其它 queries、派生列、answer key | results、环境定义、运行脚本、REPRODUCING.md，以及官方额外排除的训练数据/缓存/检查点 |
| 提交 | discovery_result.json | report.json |
| 主分数 | 官方 final_score | 官方 task accuracy（全部问题正确才算成功） |

Discovery 的 `workflow` 是评分输入，没有另造 workflow 分数。官方评分函数和 CORE 的过滤、评分类从校验 SHA256 的上游源码加载，函数体保持原样；只隔离无关导入和替换 judge 传输层。Discovery 的 judge 出错、无效 facet 分数不会转成有效零分。CORE 保留官方数值预测区间、字符串、列表和视觉问题规则。

准备目录包含 `task_spec.json` 和公开文件 SHA256 清单；复制到 episode 时不携带 `.env`。评分使用 episode 根目录的冻结 TaskSpec，不信任 agent 可修改的 workspace 副本。原始 scorer 输出可能含标准答案，只留在 scorer/controller 一侧。

## 获取与准备

在 controller/scorer 环境执行；不要把上游 benchmark 仓库或私有答案挂载给宿主：

```bash
python -m pip install -e '.[benchmark-scoring]'
python scripts/bootstrap_benchmarks.py
```

已有 checkout 只验证，不自动 reset。固定 revision 和 scorer hash 位于 `benchmark.lock.json`。CORE 的 `core_test.json.gpg` 按上游 README 解密；下载并解包选中的 capsule 到 controller 私有目录。准备器要求已解包目录，不自动运行 capsule，也不自动解包不可信 tar。数据文件保留在仓库外或 `upstreams/` 下，切勿提交答案。

Discovery 示例（先确认此 ID 在选定 checkout 中存在）：

```bash
rac-ai-scientist prepare-task \
  --benchmark-id discoverybench --task-dir upstreams/discoverybench \
  --split train --task-id nls_bmi/metadata_0/0/0 \
  --output prepared_tasks/discoverybench/train/nls_bmi/metadata_0/0/0
rac-ai-scientist check-workspace prepared_tasks/discoverybench/train/nls_bmi/metadata_0/0/0
```

CORE 示例，`CAPSULE_ID` 替换成实际冻结的 capsule ID：

```bash
rac-ai-scientist prepare-task \
  --benchmark-id corebench --task-dir upstreams/hal_harness \
  --dataset /private/core_test.json --capsules /private/capsules \
  --split dev --task-id CAPSULE_ID \
  --output prepared_tasks/corebench/dev/CAPSULE_ID
```

Discovery 原始 metadata 整份复制会泄漏答案；不能手工把原始 topic 目录当作 prepared task。CORE 不能将 Easy/Medium 的文件或依赖镜像复用到 Hard。GPU 标记 capsule 在当前配置中被拒绝。`check-workspace` 对新 benchmark 检查的是**运行前的 prepared 输入目录**，不是已经生成新文件的 episode。

## 提交和宿主行为

Discovery：

```json
{"hypothesis":"由数据支持的具体发现", "workflow":"实际执行的分析步骤", "evidence":[{"claim":"具体结论", "artifact":"outputs/result.csv"}]}
```

CORE：

```json
{"原始问题字符串，逐字匹配": 42, "另一条原始问题": "答案"}
```

禁止用 Markdown 替代 JSON，不调用额外模型补写或修复提交。提交缺失、重复 JSON 键、非有限数值、CORE 键不匹配均记录为 `invalid_submission`。

- N0：各宿主完整原生生命周期运行一次。
- R1：原生固定后继 + SharedNet。
- R2：共享运行时路由。
- R3：增加 scoped contract；仅 finalization 阶段要求合法 JSON，验证保持 advisory，不触发停止、回滚或额外重试。

宿主的原生论文/报告产物可以继续生成，benchmark 提交单独检查。Agent Laboratory 通过原生 execute_code 写 JSON；EvoScientist 的 native rubric 使用对应 benchmark 的交付要求；ARK 每个原生角色收到相同交付说明。

## CORE 任务执行隔离

`Dockerfile.task-runtime` 只包含基础 Python/pip/Bash。先构建，再固定 image ID：

```bash
docker build -f docker/Dockerfile.task-runtime -t rac-task-runtime .
docker image inspect rac-task-runtime --format '{{.Id}}'
```

将输出的 `sha256:...` 填入配置。每个 episode 新建容器，安装依赖的时间属于该 episode。控制器检查初始 Python 包仅有 pip/setuptools/wheel，记录镜像 ID、初始包清单、CPU/内存限制。镜像仍须使用此 Dockerfile 构建并审查；包清单检查不能识别所有系统级预装依赖。

推荐部署：controller 管 Docker，宿主容器只接收 task runtime 的 HTTP endpoint 和随机 token。宿主和 task 容器以**相同绝对路径**挂载该 episode workspace；宿主不挂 Docker socket、私有 benchmark 仓库、答案或其它 episode。宿主还需一个只读 prepared bundle，以及自己的 episode 元数据写入目录。

```bash
# controller：用保密环境变量向 controller 和宿主传递同一个随机 token
# 日志路径放到 episode 之外，避免与 run-one 的空目录检查冲突。
python -m rac_ai_scientist.task_runtime.serve \
  --workspace /runs/core-smoke/workspace --log-dir /runtime-logs/core-smoke \
  --image sha256:REPLACE_WITH_IMAGE_ID \
  --wall-seconds 7200 --bind 0.0.0.0 --port 8765
```

`RAC_TASK_RUNTIME_TOKEN` 必须事先设置；endpoint 仅向受信任的实验网络开放。单机 Linux 调试也可在 `run-one` 使用 `--runtime-image sha256:...`，由 CLI 控制容器生命周期。

宿主执行工具适配点：Agent Laboratory 的 `execute_code` 及已导入别名；EvoScientist 的 `_execute_prepared_command`，保留外层验证；ARK 的 OpenHands Python CLI 包装器，替换 TerminalExecutor 和 LocalWorkspace 的执行入口。不支持的 API/独立二进制 OpenHands 会报错，不回退到本地执行。每次命令采用新的同步 Bash shell，工作目录默认为 workspace，文件和安装的依赖在 episode 内持久化；不支持跨调用交互 stdin，后台进程在命令结束时清理。此执行协议必须用于所有四个条件并记录，不能混用后直接比较。

CORE host 运行示例：

```bash
rac-ai-scientist run-one --host ark --condition N0 \
  --task-dir /prepared/corebench/dev/CAPSULE_ID \
  --run-root /runs --episode-id core-smoke \
  --runtime-url http://CONTROLLER:8765 \
  --max-cost-usd 5 --max-input-tokens 100000 --max-output-tokens 20000 \
  --max-agent-calls 100 --max-wall-seconds 3600 --max-hops 30
```

预算是示例，正式对照需统一冻结。`AGENT_MODEL_NAME`、`AGENT_API_KEY`、`AGENT_API_BASE` 沿用已有约定。R1–R3 另需每个 episode 唯一的 SharedNet room/invite；N0 不用 SharedNet。Discovery 使用相同 run-one 命令替换 prepared 目录，可选相同执行隔离；其各条件必须使用一致环境。

## 独立评分

先结束宿主进程，外部 controller 服务也需停止；然后在 controller 或独立 `Dockerfile.benchmark-scorer` 镜像运行。私有 source/dataset 只挂载给 scorer。

```bash
# Discovery 设置 JUDGE_PROVIDER=openai 或 azure、JUDGE_MODEL_NAME、JUDGE_API_KEY。
# Azure 还需 JUDGE_API_BASE 和 JUDGE_API_VERSION。
rac-ai-scientist score-episode --episode-dir /runs/discovery-smoke \
  --benchmark /private/discoverybench --score-timeout 1800

rac-ai-scientist score-episode --episode-dir /runs/core-smoke \
  --benchmark /private/hal_harness --dataset /private/core_test.json
```

新评分器在独立 Python 子进程中运行，只读取冻结 TaskSpec、解析后的提交和私有参考，不执行 agent 代码。Discovery 的 judge 需支持官方 Chat Completions 参数（包括 temperature / max_tokens）；不兼容的模型应显式失败，不静默改评分算法。judge token/call/wall usage 独立记录；当前传输未提供美元计价，不应把缺失费用当成零。

`episode.json` 分别保存 `status`、`native`、`submission_status`、`scoring_status`。`score.json` 状态为 `scored` / `invalid_submission` / `scorer_failed`，后两类 `total_score=null`。分析时保留全部 planned episode，报告成功覆盖率和失败原因，不把缺失评分混成有效零分或悄悄从分母删除。CORE 同时保留原始 written/vision 计数；两个 benchmark 不混为一个总均分。

## 验收顺序

1. 运行离线测试：`python -m unittest discover -s tests -v`。源码缺失时官方 smoke 会跳过；先 bootstrap 并安装 scoring extra 才能覆盖官方数值评分。Discovery 的离线 smoke 使用假的 judge 响应，仅验证算法接入，不代表真实 LLM judge 已验收。
2. Linux 先执行 `python scripts/smoke_task_runtime.py --image sha256:实际镜像ID`，验证挂载、初始依赖、命令超时杀进程、文件同步、禁止跨 episode 复用和清理；再验证三个原生执行入口的实际调用，确认执行日志和 hook hash 出现在 controller 日志目录。
3. 每个 benchmark 先跑 1 个 ARK N0；在独立 scorer 中做 reference/oracle 与真实输出对照。CORE oracle 仅用于 scorer，绝不能复制给 agent。
4. 扩展到每 benchmark 3 宿主 × 4 条件，共 24 个 smoke cell。记录所有失败，不自动补跑成“成功”。
5. 通过后冻结约 10 个 Discovery train dev 任务、5 个 CORE CPU dev capsule，再跑每 seed 180 个 pilot episode。测试集不参与适配；CORE eval capsule 与 dev 严格不重叠。

`configs/discoverybench.example.json`、`configs/corebench.example.json` 的任务列表、模型和预算故意留空，待真实 smoke 确认可行后冻结。填入任务后使用原有 `doctor` / `plan`；新 episode ID 含 benchmark 和 split，任务目录按 `prepared_tasks/<benchmark>/<split>/<task-id>` 查找。计划展开本身不会执行模型调用。
