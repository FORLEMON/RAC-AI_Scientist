# RAC × AI Scientist

This repository is the integration and evaluation layer for applying Runtime
Agent Coordination (RAC) to three active AI-scientist hosts—ARK, Agent
Laboratory, and EvoScientist—and evaluating them on ResearchClawBench. Earlier
data-to-paper, AI-Researcher, and AutoResearchClaw integrations remain for
historical reproducibility but are excluded from new experiments.

DiscoveryBench (real, no domain knowledge) and CORE-Bench (HAL Hard, CPU)
adapters are also available. They add sanitized task bundles, JSON submissions,
private scoring workers, and an isolated task execution runtime. See the
[Chinese setup and acceptance runbook](docs/benchmark-adapters.zh-CN.md) and
`benchmark.lock.json`. Local Docker environments and native execution hooks have
offline acceptance checks; real model-driven benchmark episodes still require
validation. Existing ResearchClawBench commands remain the default.

Windows users: see [the local WSL2/Docker environment guide](docs/local-host-environments.zh-CN.md)
for the three installed host environments and the `scripts/hosts.ps1` launcher.

The repository deliberately keeps upstream projects separate. The integration
package owns the shared N0–R3 policy, schemas, accounting, host bridges, and
experiment runner. A host bridge may serialize native state, invoke an existing
capability, and return artifacts and usage; it must not contain routing,
acceptance, recovery, or benchmark-specific policy.

## Experimental conditions

| ID | Enabled coordination mechanisms |
|---|---|
| N0 | Native fixed workflow |
| R1 | SharedNet runtime communication over the native fixed workflow |
| R2 | R1 + runtime routing |
| R3 | R2 + scoped work contracts + advisory artifact-grounded verification |

Conditions are cumulative. Within a host/task/seed comparison, model, tools,
permissions, input artifacts, and lifecycle budget must be identical. Router and
verifier usage is charged to the same lifecycle budget.

R4 and R5 are retired condition labels and are intentionally rejected by the
current CLI. Their former contract, verification, and recovery behavior has
been simplified into R3: the work contract scopes the requested change, the
verifier records artifact-grounded findings, and that verdict becomes advisory
context for the next runtime selection. A negative verdict does not stop the
episode, discard artifacts, or force a retry.

Every N0 episode bypasses the RAC episode runner. The integration layer invokes
the host-owned top-level lifecycle once and records only the native boundary,
artifacts, usage, and terminal state: ARK `Orchestrator.run()`, Agent Laboratory
`LaboratoryWorkflow.perform_research()`, data-to-paper `run_all_steps()`,
AutoResearchClaw `execute_pipeline()`, and one complete EvoScientist Deep Agent
job. ARK retains native caps of three development and three paper-review
iterations. AI-Researcher's published Level-1 launcher is coupled to its own ML
benchmark schema and nested Docker layout; its N0 compatibility path keeps the
native MetaChain agents and fixed Level-1 ordering while mapping a sanitized
ResearchClawBench workspace into that flow. It is reported explicitly as a
compatibility-native run, not as an unmodified invocation of the upstream CLI.
R1--R3 continue to use the capability-level RAC runner for every active host.
All three active bridges use a fresh SharedNet Room as the communication plane
while RAC remains the routing, contract, verification, and stopping control
plane. R1 keeps each host's declared fixed successor at every hop; runtime
routing begins at R2. R3 verification is advisory: every verdict is recorded
and forwarded to the next selected agent, while artifacts are retained and the
verdict itself never stops, retries, or rolls back a step.

## Repository boundary

The eight source directories currently collected beside this file are local
read-only snapshots and are ignored by Git. `upstream.lock.json` records their
provenance. Fresh checkouts should be placed under `upstreams/` by the bootstrap
script; generated runs go under `runs/`.

```text
configs/                  shared experiment and host-interface declarations
src/rac_ai_scientist/     shared RAC policy, schemas, runner, and bridges
tests/                    offline contract and invariant tests
scripts/                  bootstrap and developer commands
upstreams/                ignored upstream checkouts
runs/                     ignored generated episodes and ledgers
```

## Safe offline quickstart

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
rac-ai-scientist doctor --config configs/experiment.example.json
python -m unittest discover -s tests -v
```

PowerShell activation is `.venv\Scripts\Activate.ps1`. `doctor` is read-only and
does not call a model. Live execution remains disabled until all upstream
revisions, host environments, credentials, task selection, and budget fields are
explicitly configured.

To materialize resolved upstream revisions, run `python scripts/bootstrap.py`.
The collected dependencies are revision-pinned. The private/unpublished RAC
research snapshot is optional because the runnable experiment profile lives in
this repository; `--allow-floating` remains an explicit, non-reproducible
development escape hatch.

AI-Researcher's upstream tree contains filenames that are illegal on Windows
(`:` and `?`). The collected Windows snapshot records its portable extraction
hash and seven omitted template files separately from the canonical Git tree.
For a byte-complete bootstrap of that host, clone/build on a Linux filesystem;
the adapter does not depend on those omitted writing-template filenames.

After filling a copied experiment config, expand the complete Cartesian product
without calling a model:

```bash
rac-ai-scientist doctor --config configs/experiment.json
rac-ai-scientist plan --config configs/experiment.json
```

The JSONL plan has stable episode IDs and a config hash, so a scheduler can
resume without silently duplicating cells.

## Starting a live run

The following procedure is the canonical single-episode workflow on a Linux
host with Docker Compose. Run every formal host/condition/seed cell in a fresh
episode directory. Do not reuse a SharedNet Room between formal R1--R3 cells.

### 1. Bootstrap and verify the selected upstreams

From the repository root, materialize only the active hosts and benchmark that
you need. The example below prepares all three active hosts:

```bash
python3 scripts/bootstrap.py \
  --only ark \
  --only agent_laboratory \
  --only evo_scientist \
  --only researchclawbench

python3 scripts/bootstrap_host_archives.py  # verifies active hosts and fetches pinned PaperBanana
python3 scripts/cache_host_models.py       # MiniLM, tiktoken and verified Miniforge installer

PYTHONPATH=src python3 -m rac_ai_scientist.cli verify-upstreams \
  --only ark \
  --only agent_laboratory \
  --only evo_scientist \
  --only researchclawbench
```

Use the following exact Compose service, CLI host ID, and build-context
variable combinations:

| Host | Compose service | `--host` | Build-context variable |
|---|---|---|---|
| ARK | `ark` | `ark` | `ARK_CONTEXT` |
| Agent Laboratory | `agent-laboratory` | `agent_laboratory` | `AGENT_LABORATORY_CONTEXT` |
| EvoScientist | `evo-scientist` | `evo_scientist` | `EVO_SCIENTIST_CONTEXT` |

### 2. Configure provider and judge credentials

Create the repository `.env` from your private provider configuration. At
minimum, Compose needs the evaluated model settings below. Configure the judge
variables as well if the episode will be scored:

```dotenv
AGENT_API_BASE=<provider endpoint root>
AGENT_API_KEY=<private evaluated-model key>
AGENT_MODEL_NAME=<provider/model identifier>

JUDGE_PROVIDER=<judge adapter name>
JUDGE_API_BASE=<judge endpoint root>
JUDGE_API_KEY=<private judge key>
JUDGE_MODEL_NAME=<judge deployment name>
JUDGE_API_VERSION=<judge API version>
JUDGE_MAX_WORKERS=1
```

Never commit this file. Do not print populated keys or SharedNet invites in
logs. Compose reads the repository `.env` automatically, so it normally does
not need to be sourced into the interactive shell.

### 3. Select one host, task, condition, and lifecycle budget

This example selects Agent Laboratory, `Math_000`, R3, and seed 0. Change
`SERVICE`, `HOST`, and the matching context variable together when selecting a
different host. Use exactly the same lifecycle budget across compared
conditions.

```bash
export REPO="$(pwd)"
export TASK=Math_000
export CONDITION=R3                 # N0, R1, R2, or R3
export SEED=0
export SERVICE=agent-laboratory
export HOST=agent_laboratory
export AGENT_LABORATORY_CONTEXT=./upstreams/agent_laboratory
export RCB_CONTEXT=./upstreams/researchclawbench

export MAX_COST_USD=25
export MAX_INPUT_TOKENS=60000000
export MAX_OUTPUT_TOKENS=1300000
export MAX_AGENT_CALLS=900
export MAX_WALL_SECONDS=21600
export MAX_HOPS=14

export STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export PREPARED="$REPO/prepared_tasks/${TASK}_${HOST}_${CONDITION}_${SEED}_${STAMP}"
export PREPARED_TASK="$PREPARED"
export RUN_ROOT="$REPO/runs"
export EPISODE="${HOST}-${TASK,,}-${CONDITION,,}-s${SEED}-${STAMP}"
mkdir -p "$RUN_ROOT"
```

The numbers above are an example, not a universal benchmark budget. A large
cumulative output budget is not sent as one provider request: host adapters may
apply a smaller per-request completion cap while accounting all requests
against the lifecycle total.

### 4. Create a target-free task bundle

Never mount `upstreams/researchclawbench/tasks/<task>` directly into an
evaluated host because it contains `target_study`. Materialize a sanitized,
unique bundle instead:

```bash
PYTHONPATH=src python3 -m rac_ai_scientist.cli prepare-task \
  --task-dir "$REPO/upstreams/researchclawbench/tasks/$TASK" \
  --output "$PREPARED"

PYTHONPATH=src python3 -m rac_ai_scientist.cli check-workspace "$PREPARED"
```

For R1, R2, or R3, create a fresh SharedNet Room and store its values only in
`$PREPARED/.env`:

```bash
vim "$PREPARED/.env"
```

```dotenv
SHAREDNET_ROOM_ID=rom_example
SHAREDNET_INVITE='ROOM=rom_example TOKEN=rit_REDACTED BASE=https://www.sharednet.ai'
SHAREDNET_BASE_URL=https://www.sharednet.ai
```

The Room ID must match the Room embedded in the invite. The dotenv file is read
before workspace materialization and is not copied into the evaluated
workspace. Only the non-secret Room ID is recorded in episode provenance. N0
does not read or require SharedNet configuration.

### 5. Build and run the zero-model preflight

Each host has a separate image because their dependency stacks conflict. A
successful unit test does not replace an image build or runtime preflight:

```bash
docker compose build "$SERVICE"
test "$?" -eq 0

docker compose run --rm "$SERVICE" \
  doctor-host --host "$HOST" </dev/null

docker compose run --rm "$SERVICE" \
  check-workspace /input/task </dev/null
```

Both preflight commands are zero-model-call checks. Stop if either fails.

### 6. Start the episode

For a long run, first enter a named tmux session so disconnecting SSH does not
terminate the experiment:

```bash
export SESSION="${HOST}-${CONDITION,,}-${STAMP}"
tmux new-session -s "$SESSION"
```

Inside tmux, restore the exports from steps 3--4 if this is a new shell, then
start the evaluated host:

```bash
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

For N0, omit `--sharednet-env-file`; the native lifecycle must remain
independent of SharedNet. Detach from tmux with `Ctrl-b d`. Reattach or inspect
the run with:

```bash
tmux attach -t "$SESSION"
tail -F "$REPO/${EPISODE}.run.log"
```

Do not infer failure merely from a quiet log: some upstream hosts buffer their
output. Check the container and episode record as well:

```bash
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
test -f "$RUN_ROOT/$EPISODE/episode.json" && \
  python3 -m json.tool "$RUN_ROOT/$EPISODE/episode.json"
```

### 7. Score the episode

Scoring runs in a separate image because only the scorer may read
`target_study`. Build it once per integration revision, then score any episode
that produced `episode.json`, including failed or budget-exhausted episodes:

```bash
docker compose build scorer
test "$?" -eq 0

docker compose run --rm scorer score-episode \
  --episode-dir "/runs/$EPISODE" \
  --benchmark /opt/benchmark
```

A valid numeric zero is different from an incomplete score. Provider failures,
parse failures, and missing reports must produce `total_score: null` with an
error explanation rather than silently becoming zero.

### 8. Archive the evidence

Preserve `episode.json`, `score.json`, `coordination.jsonl`, logs, reports,
code, and outputs. Exclude every generated `.conda_env/` runtime directory:

```bash
mkdir -p "$REPO/exports"
tar --exclude='*/.conda_env' \
  -czf "$REPO/exports/${EPISODE}.tar.gz" \
  -C "$RUN_ROOT" "$EPISODE"
```

Keep failed attempts for audit and always use a new episode ID for a retry.
`run-one` intentionally refuses to overwrite an existing episode.

## Integrity rules

- The evaluated host never receives `tasks/<id>/target_study`; only the external
  ResearchClawBench scorer may read it.
- Every episode records the host and RAC source revisions, configuration hash,
  task, seed, condition, budget, usage, terminal status, and artifact hashes.
- An agent's completion statement is not evidence. R3 records an external
  artifact-grounded verdict, but the verdict is advisory and never discards
  the agent's persisted work.
- Failed and budget-exhausted episodes remain in the denominator.
- Host-specific capability names and filesystem paths may appear in bridge data;
  decisions over those declarations live only in the shared package.

## RAC source relationship

The collected RAC source describes itself as a research alpha. Its reference
`baselines/algorithms/rac.py` intentionally omits retry/reroute policy, spawn
templates, disclosure measurement, and in-turn deadlines, while this paper's
N0–R3 study requires communication, routing, contracts, and advisory verification.
Consequently, its schemas and mechanism invariants are treated as the
design source, but the complete longitudinal condition profile is implemented
and tested here. This avoids importing an exploratory baseline and claiming it
already implements mechanisms it explicitly does not.

See `docs/architecture.md` for the implementation boundary and staged delivery
plan.

## Experiment data

The September 2026 DiscoveryBench results, sanitized logs, analysis artifacts,
and scoring audit are indexed in [output/README.zh-CN.md](output/README.zh-CN.md).
The complete archive retains failed, interrupted, cancelled, and unstarted
records separately from scored submissions. These are single-task, seed-0 smoke
tests, not a benchmark-wide comparison.
