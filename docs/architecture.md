# Architecture and experimental boundary

## One policy, three active hosts

For R1--R3, the common runner owns the episode loop. Every active host publishes
typed work requests, results, and verification advisories in a caller-selected
SharedNet Room. At each hop the runner requests a normalized checkpoint from the
host bridge. The checkpoint contains the objective, native stage, persisted
artifact manifest, unresolved issues, remaining budget, and capability cards.
The selected condition determines which common mechanisms may run.

```text
ResearchClawBench task workspace
              |
              v
      N0 host scheduler OR R1--R3 common runner
                         -----> append-only usage/decision ledger
              |
      shared N0--R3 policy
              |
              v
       HostBridge protocol
        |-- ARK
        |-- Agent Laboratory
        `-- EvoScientist
```

The data-to-paper, AI-Researcher, and AutoResearchClaw bridges remain available
only to reproduce historical runs; they are excluded from the active matrix and
do not join SharedNet.

The bridge owns translation, not policy. It may:

1. map native roles/stages/products to capability cards;
2. map native files/products to typed artifact records;
3. invoke a requested native capability with a supplied work contract;
4. report provider usage and native errors.

It may not rank capabilities, choose retries, define acceptance thresholds,
classify issues for routing, or decide when the episode stops.

## Condition boundary

- **N0** uses the host's complete native scheduler without passing phase
  transitions through the shared RAC policy. ARK, Agent Laboratory,
  and EvoScientist enter their top-level native
  lifecycle once. For ARK this is one `Orchestrator.run()` lifecycle with native
  caps of three development iterations and three paper-review iterations.
  The integration layer records only the native run boundary, artifacts, usage,
  and terminal state.
- **R1** adds SharedNet runtime communication while preserving the host's fixed
  native successor at every hop.
- **R2** selects from admitted capability cards using the shared policy.
- **R3** additionally supplies a minimum-scoped contract and fingerprints
  artifacts before and after execution. Its `supported`, `refuted`, or
  `inconclusive` verdict is recorded and injected into the next agent context,
  but never stops, retries, reroutes, or rolls back the invocation.

## Benchmark boundary

The host-side runner accepts only a pre-sanitized task bundle containing the
task description, `data/`, and `related_work/`, then copies those inputs into a
new episode workspace. `target_study/` is never copied or mounted into an
evaluated host container. After the host terminates, a separate scorer image
receives the terminal report and the original benchmark-owned checklist/target
material.

## Environment boundary

The six hosts cannot safely share one Python environment. ARK uses modern
LiteLLM/OpenAI dependencies, Agent Laboratory has a large pinned ML stack, and
data-to-paper 1.1.22 pins the pre-1.0 OpenAI client and PySide. AI-Researcher,
EvoScientist, and AutoResearchClaw add separate MetaChain, DeepAgents/LangGraph,
and ResearchClaw pipeline stacks. Each host therefore runs in its own
image/virtual environment. The small integration package is
installed inside each image and imports only that image's host bridge; no two
host dependency stacks share a process. Episode state and coordination traces
cross runs only as JSON/JSONL and persisted artifacts.

## Delivery stages

1. Freeze upstream provenance and common schemas/invariants.
2. Implement N0 recording adapters and benchmark-safe workspace materialization.
3. Expose each host's existing roles as resumable capability calls.
4. Enable R1--R3 strictly through the shared policy.
5. Add isolated images, one-command smoke runs, and resume-safe batch execution.
6. Freeze task selection before inspecting condition outcomes and publish hashes,
   ledgers, analysis scripts, and failure-complete reports.
