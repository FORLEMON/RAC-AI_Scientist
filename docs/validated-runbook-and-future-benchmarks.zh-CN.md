# RAC × AI Scientist：已验证运行手册与后续 Benchmark 适配建议

> 更新日期：2026-09-23  
> 用途：供合作者理解当前已经跑通的实验方法、尚未验证的边界，以及未来适配 DiscoveryBench 和 CORE-Bench 的建议。  
> 安全说明：本文不包含任何 API key、SharedNet invite/token、私人 endpoint 或 SSH 私钥内容。

## 1. 当前实验范围

当前正式实验只包含三个 active host：

| Host | Compose service | CLI `--host` |
|---|---|---|
| ARK | `ark` | `ark` |
| Agent Laboratory | `agent-laboratory` | `agent_laboratory` |
| EvoScientist | `evo-scientist` | `evo_scientist` |

AI-Researcher、AutoResearchClaw 和 data-to-paper 的 adapter 仅保留用于历史复现，不应继续安排新的正式实验。

当前 condition ladder 只有 N0–R3：

| Condition | 定义 |
|---|---|
| N0 | 完整 host-native fixed workflow；不连接 SharedNet，也不进入 RAC phase loop |
| R1 | 在 native fixed successor 顺序上增加 SharedNet runtime communication |
| R2 | R1 + shared runtime routing |
| R3 | R2 + scoped work contract + artifact-grounded advisory verification |

R4、R5 已退出当前实验，不应再出现在 CLI、队列或实验表中。

R3 verifier 的结论可以是 `supported`、`refuted` 或 `inconclusive`。三者都必须写入 ledger 并作为下一跳 advisory context，但 verifier 本身不得触发 STOP、RETRY、REROUTE、RECOVER 或 rollback。只有真实 host terminal state、provider terminal error、HTTP 402、生命周期预算耗尽或 hard hop limit 才能终止 episode。

## 2. 已验证状态与边界

为避免把局部 smoke test 误报成完整跑通，当前状态分为三类。

### 2.1 已验证

- ARK、Agent Laboratory、EvoScientist 是当前 active hosts。
- N0 使用 host 原生顶层 lifecycle；R1–R3 使用共享 RAC runner。
- ResearchClawBench task 可以被清理为不含 `target_study` 的 prepared task。
- N0 不读取 SharedNet 配置。
- R1–R3 使用独立 SharedNet Room；Room ID 可以进入 provenance，invite/token 不可以。
- Agent Laboratory 的 CPU-only runtime 依赖栈已在 server-local harness 中通过完整离线 import smoke test。
- Azure GPT-5.5 的 ARK/OpenHands 请求路径曾通过真实 agent run 验证；不能只用直接 LiteLLM 请求代替 OpenHands smoke test。
- DeepSeek V4 Pro 必须通过独立 Pro relay，不能只修改 ARK 日志中的 model label。
- ResearchClawBench judge 当前默认使用 Azure GPT-5.5 Chat Completions；历史 GPT-5.4 直连 smoke test 已成功，但不能替代 GPT-5.5 端到端验证。

### 2.2 仅在特定 server-local 路径验证

- Agent Laboratory N0 的 sanitized harness 路径依赖服务器上的专用 image、relay、clean integration source 和 credential file。
- 这些 server-local harness/image 修复不等于仓库 Compose 路径已经在所有机器上完全复现。
- DeepSeek V4 Pro 的 compose override 和 relay 配置是服务器本地运维文件，不应默认提交到公共仓库。

### 2.3 尚需正式确认

- Azure GPT-5.5 judge adapter 的完整 `episode → scorer image → score.json` 路径仍需按结果完整性规则持续验证。
- 每个 active host 的每个新 task 都必须重新进行 runtime preflight；一个 task 成功不能证明其他 task 的依赖也齐全。
- DiscoveryBench 和 CORE-Bench 目前仅完成方案选择，尚未下载、适配或运行。

## 3. 基础目录与安全规则

服务器示例根目录：

```bash
export SRC_ROOT=/mnt/data0/ldav/src
export REPO="$SRC_ROOT/RAC-AI_Scientist"
export RUN_ROOT="$REPO/runs"
```

核心规则：

1. `.env`、credential JSON、SharedNet invite/token 绝不进入 Git、实验报告或 provenance。
2. 只允许输出“变量是否非空”的诊断，不打印值。
3. ResearchClawBench evaluated host 永远不能读取 `target_study`。
4. 每次正式 run 使用新的 episode ID；失败 run 也保留用于审计。
5. 每次正式 R1–R3 cell 使用全新的 SharedNet Room。
6. 打包结果时始终排除所有 `.conda_env/`。
7. 长时间 run 使用 `tmux`，不要依赖 SSH 会话持续在线。
8. 终端编辑器统一使用 `vim`。

## 4. 标准 ResearchClawBench 单 episode 流程

### 4.1 Bootstrap 与 provenance 检查

在仓库根目录运行：

```bash
python3 scripts/bootstrap.py \
  --only ark \
  --only agent_laboratory \
  --only evo_scientist \
  --only researchclawbench

PYTHONPATH=src python3 -m rac_ai_scientist.cli verify-upstreams \
  --only ark \
  --only agent_laboratory \
  --only evo_scientist \
  --only researchclawbench
```

不要仅凭 image 名称或 `doctor-host` 判断新 image 是否包含最新代码。凡 provenance、lock file、Dockerfile 或 host patch 有变化，都必须要求：

```bash
docker compose build "$SERVICE"
test "$?" -eq 0
```

### 4.2 选择 cell 和预算

下面只是模板，预算不应跨 task/host 生搬硬套：

```bash
export TASK=Math_000
export CONDITION=R3
export SEED=0
export SERVICE=agent-laboratory
export HOST=agent_laboratory

export MAX_COST_USD=25
export MAX_INPUT_TOKENS=60000000
export MAX_OUTPUT_TOKENS=1300000
export MAX_AGENT_CALLS=900
export MAX_WALL_SECONDS=21600
export MAX_HOPS=14

export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export PREPARED="$REPO/prepared_tasks/${TASK}_${HOST}_${CONDITION}_${SEED}_${STAMP}"
export PREPARED_TASK="$PREPARED"
export EPISODE="${HOST}-${TASK,,}-${CONDITION,,}-s${SEED}-${STAMP}"
mkdir -p "$RUN_ROOT"
```

正式比较必须满足：同一 host/task/seed 的 N0–R3 使用相同 evaluated model、工具、权限、输入和 lifecycle budget。Router/verifier 消耗也计入同一个 lifecycle budget。

如果研究问题只关注 hop budget，可以把 cost/token/agent-call 上限设置得足够高，使它们在正常 run 中不成为 binding constraint，但仍应记录这些上限和实际 usage。不能删除 accounting，也不能让不同 condition 使用不同的隐藏上限。

更稳妥的预算制定方式是：

1. 用非约束性高上限跑该 host/task 的 N0。
2. 从 N0 的真实 wall time、input/output tokens、calls 和 hops 取得基线。
3. 为 R1–R3 设置与 N0 生命周期接近的共同 ceiling，并提前声明 headroom。
4. 不要用 `Math_000` 的预算代表所有 task。

### 4.3 创建 target-free prepared task

```bash
PYTHONPATH=src python3 -m rac_ai_scientist.cli prepare-task \
  --task-dir "$REPO/upstreams/researchclawbench/tasks/$TASK" \
  --output "$PREPARED"

PYTHONPATH=src python3 -m rac_ai_scientist.cli check-workspace "$PREPARED"
```

禁止直接把完整的 `tasks/<TASK>` 挂载给 evaluated host，因为其中包含 `target_study`。

### 4.4 SharedNet 配置

N0 跳过本节。R1–R3 在 prepared task 的 `.env` 中保存本次 run 独有的配置：

```bash
vim "$PREPARED/.env"
```

```dotenv
SHAREDNET_ROOM_ID=rom_example
SHAREDNET_INVITE='ROOM=rom_example TOKEN=rit_REDACTED BASE=https://www.sharednet.ai'
SHAREDNET_BASE_URL=https://www.sharednet.ai
```

要求：

- `.env` 不复制进 evaluated workspace。
- Room ID 必须和 invite 中的 Room 一致。
- invite/token 不写入 episode metadata、ledger 或日志。
- 任一 probe/失败尝试触碰 Room 后，正式 cell 必须换新 Room。
- SharedNet member name 最长 64 字符。必须提前检查 `rac:<episode-id>` 和 `<episode-id>:<capability-id>`；建议使用短 episode ID，例如 `al-m000-r1-260920103015`。

### 4.5 零模型 preflight

```bash
docker compose build "$SERVICE"
test "$?" -eq 0

docker compose run --rm "$SERVICE" \
  doctor-host --host "$HOST" </dev/null

docker compose run --rm "$SERVICE" \
  check-workspace /input/task </dev/null
```

这一步不应调用模型。任何一步失败都应先修环境，不应启动付费 run。

### 4.6 启动与监控

```bash
export SESSION="${HOST}-${CONDITION,,}-${STAMP}"
tmux new-session -s "$SESSION"
```

在 tmux 内运行：

```bash
set -o pipefail
docker compose run --rm "$SERVICE" run-one \
  --host "$HOST" \
  --condition "$CONDITION" \
  --task-dir /input/task \
  --sharednet-env-file /input/task/.env \
  --run-root /runs \
  --episode-id "$EPISODE" \
  --seed "$SEED" \
  --max-cost-usd "$MAX_COST_USD" \
  --max-input-tokens "$MAX_INPUT_TOKENS" \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" \
  --max-agent-calls "$MAX_AGENT_CALLS" \
  --max-wall-seconds "$MAX_WALL_SECONDS" \
  --max-hops "$MAX_HOPS" \
  2>&1 | tee "$REPO/${EPISODE}.run.log"
```

N0 必须去掉：

```text
--sharednet-env-file /input/task/.env
```

常用监控：

```bash
tail -F "$REPO/${EPISODE}.run.log"
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
test -f "$RUN_ROOT/$EPISODE/episode.json" && \
  python3 -m json.tool "$RUN_ROOT/$EPISODE/episode.json"
```

不能仅因日志暂时安静就判定失败；部分 upstream 会缓冲输出。单个 agent timeout 应依照 host 的原生语义标记为 failed，而不应被误提升为整个 R1 episode 的异常终止。每次修改 timeout handling 后都应保留对应 regression test。

### 4.7 独立评分

```bash
docker compose build scorer
test "$?" -eq 0

docker compose run --rm scorer score-episode \
  --episode-dir "/runs/$EPISODE" \
  --benchmark /opt/benchmark
```

评分器是唯一允许访问 `target_study` 的组件。即使 run 是 failed 或 budget-exhausted，只要生成了可评分 workspace，也应尝试评分并保留失败信息。

必须区分：

- 合法的 `total_score: 0`
- 因报告缺失、429、解析失败或 judge 调用失败产生的 `total_score: null`

任何 `Scoring failed.`、`Failed to parse scoring response.`、HTTP 400/401/404/429 或 top-level `error` 都表示评分不完整，不能当成真实零分。

### 4.8 归档

```bash
mkdir -p "$REPO/exports"
tar --exclude='*/.conda_env' \
  -czf "$REPO/exports/${EPISODE}.tar.gz" \
  -C "$RUN_ROOT" "$EPISODE"
```

最低限度应保留：

- `episode.json`
- `score.json`
- `coordination.jsonl`
- launcher/runtime/scorer logs
- `report/`
- code、plots、tables 和其他实验输出
- artifact hashes 和 usage records

## 5. Provider 路径中的已知关键点

### 5.1 Azure GPT-5.5 + ARK/OpenHands

当前 Azure resource 使用 Chat Completions，而 OpenHands/LiteLLM 会因为 model name 含 `gpt-5` 自动转到 Responses API。已验证的 image 必须同时具备两层保护：

1. 从 OpenHands `RESPONSES_API_MODELS` 中移除/绕过 `gpt-5` 自动判定。
2. 在实际 `litellm_completion(...)` 调用传入 `_skip_responses_api_bridge=True`。

直接 LiteLLM `completion()` 成功只证明 endpoint/key/deployment 可用，不能证明 OpenHands agent path。正式运行前需要真实 OpenHands-backed smoke call。

### 5.2 DeepSeek V4 Pro relay

Pro 和 Flash 必须使用不同 relay。Relay 会根据自己的 `AZURE_AI_MODEL` 重写请求，单独修改 ARK model label 不足以证明实际调用了 Pro。

Pro 路径要求：

```text
relay service: azure-ai-relay-pro
relay model:   DeepSeek-V4-Pro
ARK API base:  http://azure-ai-relay-pro:8000/v1
ARK model:     openai/DeepSeek-V4-Pro
```

无付费调用的验证：

```bash
PRO_RELAY=rac-ai_scientist-azure-ai-relay-pro-1
docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' "$PRO_RELAY"
docker exec "$PRO_RELAY" printenv AZURE_AI_MODEL
```

必须分别得到 `healthy` 和 `DeepSeek-V4-Pro`。

一个已经建立的 ARK `Math_000` N0 Pro 队列使用过以下 ceiling：USD 70、100M input tokens、1.8M output tokens、1500 agent calls、43200 wall seconds、33 hops。这只是该 cell 的运维记录，不是其他 host/task 的推荐通用预算。

## 6. Agent Laboratory CPU-only server-local harness

Agent Laboratory upstream 在受限、离线 runtime 中缺少若干依赖。已跑通 image 包含：

- `sentence-transformers==3.3.1`
- build-time 缓存的 `all-MiniLM-L6-v2`
- `tensorflow-cpu==2.18.0`
- `USE_TF=0`，避免 Transformers 进入不兼容的 Keras 3 backend
- build-time 缓存的 `cl100k_base`

离线 smoke test 必须导入完整 workflow，不能只测试独立 package。由于 `ai_lab_repo` import 会初始化 Flask/SQLAlchemy，不能从只读 `/opt/host` 直接 import；应先复制到可写目录：

```bash
docker run --rm \
  --network none \
  --entrypoint bash \
  rac-agent-laboratory:runtime-complete-v2 \
  -lc 'rm -rf /tmp/qualification-native-host &&
       cp -a /opt/host /tmp/qualification-native-host &&
       cd /tmp &&
       PYTHONPATH=/tmp/qualification-native-host python -c "
import tiktoken
e = tiktoken.get_encoding(\"cl100k_base\")
print(\"TIKTOKEN_OFFLINE_OK\", e.name)
from ai_lab_repo import LaboratoryWorkflow
print(\"AGENT_LABORATORY_IMPORT_OK\", LaboratoryWorkflow.__name__)
"'
```

Harness 会对固定控制日志使用 exclusive creation。重跑前应归档旧日志而不是无记录删除：

```bash
OLD_LOG="$HARNESS/control/logs/agent_laboratory/N0.log"
LOG_ARCHIVE="$HARNESS/control/logs/archive/agent_laboratory"
mkdir -p "$LOG_ARCHIVE"
if [ -f "$OLD_LOG" ]; then
  mv -- "$OLD_LOG" "$LOG_ARCHIVE/N0.$(date -u +%Y%m%dT%H%M%SZ).log"
fi
```

`retrieve_full_paper_text` 的单次 90 秒 timeout stack snapshot 曾是非致命的；判断 run 是否健康应结合后续 phase marker、relay HTTP 200 和已知 usage，不能只看一条 timeout stack。

## 7. ResearchClawBench judge 配置

正式 judge 默认使用 Azure GPT-5.5 Chat Completions：

```dotenv
JUDGE_PROVIDER=azure
JUDGE_API_BASE=<Azure endpoint root>
JUDGE_API_KEY=<private key>
JUDGE_MODEL_NAME=gpt-5.5
JUDGE_API_VERSION=2025-04-01-preview
JUDGE_MAX_COMPLETION_TOKENS=1000
JUDGE_MAX_WORKERS=1
```

要求：

- 使用 Azure SDK 的 `chat.completions.create(...)`。
- 使用 deployment name `gpt-5.5`，而不是 `azure/gpt-5.5`。
- 使用 `max_completion_tokens`，不使用 `max_tokens`。
- checklist 串行执行，`JUDGE_MAX_WORKERS=1`，避免 quota burst。
- 支持文本和 image data-URL parts。
- HTTP/解析错误必须传播，不能静默转成 0 分。
- 不修改 pinned ResearchClawBench upstream；provider adapter 位于 RAC integration layer。

历史 Azure GPT-5.4 直连 Chat Completions smoke test 只能证明旧 deployment 路径可用。GPT-5.5 的正式评分必须要求 scorer 退出码为 0、`score.json` 无顶层错误、每个 checklist item 都有分数和非空 reasoning，且日志中没有 HTTP/解析失败标记。

## 8. 未来 Benchmark 适配的共同原则

DiscoveryBench 和 CORE-Bench 的适配不应修改 N0–R3 语义。建议新增 benchmark abstraction，而不是在 host bridge 内写 benchmark-specific policy：

```text
Benchmark adapter
  ├── materialize_task(source, prepared_dir)
  ├── validate_public_workspace(prepared_dir)
  ├── required_outputs()
  ├── score_episode(episode_dir, private_benchmark_data)
  └── normalize_score(raw_score)

Existing experiment layer
  ├── host-native N0
  ├── shared R1–R3 runner
  ├── lifecycle accounting
  ├── artifact hashing
  └── episode provenance
```

共同约束：

1. benchmark gold/reference/answer 只进入独立 scorer。
2. 同一 benchmark task 的 N0–R3 使用同一公开输入和 output contract。
3. 不用额外 LLM 自动“替 host 修正格式”，否则会引入未计费、未受控的 agent。
4. 如果必须做格式转换，应使用确定性 parser；无法转换时标记 invalid/incomplete。
5. 保留 benchmark 原生指标，同时可以额外提供统一的 normalized score；不能只保留统一分数。
6. 固定 benchmark revision、dataset split、scorer revision 和依赖镜像 digest。
7. CPU subset 的选择规则必须预注册，不能看结果后再挑任务。
8. 先运行 oracle/reference smoke，再运行 host 的零模型 preflight，最后才启动付费 cell。

## 9. DiscoveryBench 适配建议

### 9.1 为什么适合

DiscoveryBench 是数据驱动发现 benchmark。每个 task 由 discovery goal、dataset 和 metadata 组成，要求 agent 同时进行统计分析与语义推理，输出 hypothesis 和 analysis workflow。它通常可以在 CPU 上用 pandas、NumPy、SciPy、statsmodels 或 scikit-learn 完成，和当前 workspace-based host 很匹配。

它测量的是“端到端数据驱动发现”，不是从选题、文献、实验到整篇论文的完整生命周期。这个边界应在论文中明确。

### 9.2 建议输入边界

Evaluated host 可见：

- discovery goal
- dataset(s)
- public metadata/column descriptions
- 相同的工具、网络和依赖政策

Evaluated host 不可见：

- gold hypothesis
- gold workflow
- evaluator prompt/reference decomposition
- test split 的其他答案信息

### 9.3 建议输出 contract

```text
workspace/
├── report/report.md
├── analysis/
│   ├── analysis.py
│   └── outputs/
└── discovery_result.json
```

建议 `discovery_result.json` 至少包含：

```json
{
  "hypothesis": "...",
  "workflow": "...",
  "evidence": [
    {
      "claim": "...",
      "artifact": "analysis/outputs/result.csv"
    }
  ]
}
```

`report.md` 供人工审计和 R3 artifact verification 使用；官方 scorer 主要读取结构化 hypothesis/workflow。所有 condition 都必须收到相同 output requirement。

### 9.4 Scoring 建议

- 优先复用官方 DiscoveryBench scorer，或者使用 AstaBench 中固定版本的 DiscoveryBench implementation。
- scorer 在 evaluated host 外运行。
- 固定 evaluator model、prompt、temperature、并发数和重试政策。
- 分别保存 hypothesis score、workflow/facet score 和 aggregate score。
- 对 scorer provider error 使用 `null + error`，不能使用零分替代。

### 9.5 首轮任务选择

建议先使用 real-data validation subset，筛选约 10 个：

- 无 GPU
- 输入数据可以本地完整提供
- 不依赖登录网站或私有数据库
- oracle/baseline 在服务器 CPU、内存和 wall-time 上稳定
- 尽量覆盖多个 domain 和 workflow type

不要把 synthetic 和 real-data task 混成一个未经分层的总体均值。

## 10. CORE-Bench 适配建议

### 10.1 为什么适合

CORE-Bench 测量论文计算复现：agent 读取 `task.txt`，进入论文对应的 CodeOcean capsule，处理依赖、运行代码、分析结果并生成结构化 `report.json`。它比 DiscoveryBench 更接近“重现已有论文”，并且官方 harness 提供 `--no_gpu` 过滤。

正式实验建议使用 no-GPU Hard subset。Easy 主要是在预计算输出中抽取答案，不足以代表端到端计算复现；Medium 涉及 Docker-in-Docker，可能需要 `--privileged`，会明显增加部署和安全复杂度。

### 10.2 建议输入边界

Evaluated host 可见：

```text
workspace/
├── task.txt
└── capsule-<id>/
    ├── code/
    ├── data/
    └── public supporting files
```

Evaluated host 不可见：

- reference answers
- grader-only metadata
- hidden test expectations
- benchmark scorer implementation中可能泄露答案的内容

### 10.3 建议输出 contract

CORE-Bench 原生要求 `environment/report.json`，键是 task questions、值是答案。适配后仍应保留这个原生文件，不要只生成 `report.md`：

```text
workspace/
├── report.json
├── report/report.md
├── reproduction/
│   ├── scripts/
│   └── logs/
└── capsule-<id>/...
```

其中 `report.json` 用官方 grader；`report.md`、脚本和日志用于人工审计、R3 verification 和 failure analysis。

### 10.4 环境策略

不要在正式 run 中临时下载 capsule 或任意版本依赖。建议：

1. 预下载并校验 capsule hash。
2. 为每个选定 task 预构建固定 runtime image。
3. 记录 base image digest、language/runtime 版本和依赖 lock。
4. 先执行 reference/oracle smoke test。
5. 正式 run 禁止访问原论文官方复现仓库之外的答案来源，并遵守 benchmark 的网络政策。

原始 CORE-Bench harness 已不再积极维护，官方建议使用 Holistic Agent Leaderboard 路径。适配时可以复用其 dataset/scorer，但不必让它接管当前 N0–R3 runner。

### 10.5 首轮任务选择

建议先选约 5–10 个 Python、no-GPU Hard task：

- capsule 能在普通 Docker/CPU 上构建
- 不依赖 CUDA
- 不需要私有数据或交互式登录
- oracle runtime 在预注册 wall-time 内
- 数据和 image 尺寸适合现有服务器
- 结果可以通过明确问题和数值/文本答案评分

R task、需要嵌套 Docker、超大模型训练或超大数据下载的 task 放到第二阶段。

## 11. 建议实施顺序（当前不执行）

现阶段不做适配。等当前代码和实验流程稳定后，建议按以下顺序推进：

1. 冻结当前 ResearchClawBench N0–R3 revision，并完成 judge 端到端验证。
2. 抽象 benchmark-neutral task materialization 和 scorer interface。
3. 接入 DiscoveryBench validation subset；先做一个 task、一个 host、N0 smoke。
4. 完成三 host 的 DiscoveryBench N0 smoke，再扩展到 R1–R3。
5. 接入 CORE-Bench no-GPU Hard 的一个 Python capsule。
6. 固定 capsule runtime image，并通过 oracle/reference smoke。
7. 完成三 host N0，再扩展到 R1–R3。
8. 最后才扩大任务数和 seeds。

不要一开始创建完整笛卡尔积。即使每个 benchmark 只选 10 个 task，两个 benchmark × 10 tasks × 3 hosts × 4 conditions 已经是 240 episodes/seed。

## 12. 最低验收清单

一个正式 cell 至少必须满足：

- [ ] benchmark、dataset、host 和 RAC revision 已记录
- [ ] prepared workspace 未包含 gold/reference/private scorer data
- [ ] image build exit code 为 0
- [ ] upstream/capsule hash 验证通过
- [ ] host runtime preflight 通过且没有模型调用
- [ ] episode ID 和 SharedNet member names 长度合法
- [ ] R1–R3 使用全新 Room，N0 不读取 SharedNet
- [ ] lifecycle budget 与对应 comparison cells 相同
- [ ] `episode.json`、usage、artifacts、logs 完整保留
- [ ] scorer 在独立环境执行
- [ ] scorer error 不被写成数值零分
- [ ] archive 排除了 `.conda_env/`
- [ ] failed/budget-exhausted episode 未从实验分母中删除

---

本文描述的是当前可复现的运维和实验边界，而不是所有路径都已完成正式验证。任何新模型、新 host image、新 benchmark task 或 scorer revision 都应重新经过 build、zero-model preflight、最小 paid smoke 和独立评分验证。
