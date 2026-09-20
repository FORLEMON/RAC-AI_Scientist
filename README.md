# RAC × AI Scientist

This repository is the integration and evaluation layer for applying Runtime
Agent Coordination (RAC) to three active AI-scientist hosts—ARK, Agent
Laboratory, and EvoScientist—and evaluating them on ResearchClawBench. Earlier
data-to-paper, AI-Researcher, and AutoResearchClaw integrations remain for
historical reproducibility but are excluded from new experiments.

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

One fully specified episode is launched with:

```bash
rac-ai-scientist run-one \
  --host ark --condition R3 \
  --task-dir upstreams/researchclawbench/tasks/Astronomy_000 \
  --max-cost-usd 20 --max-input-tokens 1000000 \
  --max-output-tokens 200000 --max-agent-calls 40 \
  --max-wall-seconds 14400 --max-hops 20
```

The command requires `AGENT_MODEL_NAME` and `AGENT_API_KEY`. For every active
host's R1--R3 episode, create a fresh Room and put its settings in
`<task-dir>/.env`:

```dotenv
SHAREDNET_ROOM_ID=rom_example20260917R3
SHAREDNET_INVITE='ROOM=rom_example20260917R3 TOKEN=rit_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA BASE=https://www.sharednet.ai'
```

Use `--sharednet-env-file` when the per-run file lives elsewhere. A command-line
`--sharednet-room-id` overrides the file, and the selected room must match the
room embedded in the invite. This dotenv file is read before materialization
and is not copied into the evaluated workspace. The room id is recorded for
provenance, while the invite token is never written to the episode. N0 neither
reads nor requires SharedNet configuration. The command creates a fresh
episode directory, copies only host-visible benchmark inputs, writes an
append-only coordination ledger, and refuses to reuse an existing episode ID.

## Isolated host images

Each host has a separate image because their dependency stacks conflict. On
this collected workspace, point Compose at the existing snapshots:

```powershell
$env:ARK_CONTEXT = "./ARK"
$env:AGENT_LABORATORY_CONTEXT = "./AgentLaboratory-main"
$env:DATA_TO_PAPER_CONTEXT = "./data-to-paper-main"
$env:AI_RESEARCHER_CONTEXT = "./AI-Researcher-main"
$env:EVO_SCIENTIST_CONTEXT = "./EvoScientist-main"
$env:AUTO_RESEARCH_CLAW_CONTEXT = "./AutoResearchClaw-main"
$env:RCB_CONTEXT = "./ResearchClawBench-main"
docker compose build ark
docker compose run --rm ark doctor-host --host ark
```

The three added services follow the same pattern. For example:

```powershell
docker compose build evo-scientist
docker compose run --rm evo-scientist doctor-host --host evo_scientist
```

In a fresh GitHub clone, run `python scripts/bootstrap.py` first and keep the
default contexts. Before a live run, create a target-free bundle on the host and
mount only that bundle—never mount the benchmark's full `tasks/` directory into
an evaluated host:

```powershell
rac-ai-scientist prepare-task `
  --task-dir "ResearchClawBench-main/tasks/Astronomy_000" `
  --output "prepared_tasks/Astronomy_000"
$env:PREPARED_TASK = "./prepared_tasks/Astronomy_000"
```

The live container reads `/input/task` and writes to the mounted `/runs`
directory:

```bash
docker compose run --rm <service> run-one --host <host-id> --condition R3 \
  --task-dir /input/task --run-root /runs \
  --max-cost-usd 20 --max-input-tokens 1000000 \
  --max-output-tokens 200000 --max-agent-calls 40 \
  --max-wall-seconds 14400 --max-hops 20
```

Active service/host-id pairs are `ark`/`ark`,
`agent-laboratory`/`agent_laboratory`, and
`evo-scientist`/`evo_scientist`.

After the evaluated host exits, score it in the separate judge image. Only this
image contains `target_study`:

```bash
docker compose build scorer
docker compose run --rm scorer score-episode \
  --episode-dir /runs/<episode-id> --benchmark /opt/benchmark
```

Building images downloads substantial upstream dependencies. `doctor-host` is
the required zero-model-call gate after a build; image builds and live bridges
have not been certified merely by the offline unit suite.

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
