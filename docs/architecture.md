# Architecture and experimental boundary

## One policy, three host interfaces

The common runner owns the episode loop. At each hop it requests a normalized
checkpoint from the host bridge. The checkpoint contains the objective, native
stage, persisted artifact manifest, unresolved issues, remaining budget, and
capability cards. The selected condition determines which common mechanisms may
run.

```text
ResearchClawBench task workspace
              |
              v
      common episode runner  -----> append-only usage/decision ledger
              |
      shared N0--R5 policy
              |
              v
      HostBridge protocol
       /       |       \
     ARK   Agent Lab   data-to-paper
```

The bridge owns translation, not policy. It may:

1. map native roles/stages/products to capability cards;
2. map native files/products to typed artifact records;
3. invoke a requested native capability with a supplied work contract;
4. report provider usage and native errors.

It may not rank capabilities, choose retries, define acceptance thresholds,
classify issues for routing, or decide when the episode stops.

## Condition boundary

- **N0** delegates transition choice to `HostBridge.native_next` and records the
  same normalized trace without changing the host workflow.
- **R1** selects from admitted capability cards using the shared policy.
- **R2** additionally supplies a minimum-scoped contract derived from the chosen
  card and checkpoint.
- **R3** fingerprints declared writable artifacts before and after execution and
  produces `supported`, `refuted`, or `inconclusive`.
- **R4** may preserve verified partial effects after timeout/empty response and
  reconstruct a safe next checkpoint.
- **R5** maintains stable issue records and uses their type, repetition, and
  acceptance checks to reroute, reverify, escalate, or stop.

## Benchmark boundary

The host-side runner accepts only a pre-sanitized task bundle containing the
task description, `data/`, and `related_work/`, then copies those inputs into a
new episode workspace. `target_study/` is never copied or mounted into an
evaluated host container. After the host terminates, a separate scorer image
receives the terminal report and the original benchmark-owned checklist/target
material.

## Environment boundary

The three hosts cannot safely share one Python environment. ARK uses modern
LiteLLM/OpenAI dependencies, Agent Laboratory has a large pinned ML stack, and
data-to-paper 1.1.22 pins the pre-1.0 OpenAI client and PySide. Each host therefore
runs in its own image/virtual environment. The small integration package is
installed inside each image and imports only that image's host bridge; no two
host dependency stacks share a process. Episode state and coordination traces
cross runs only as JSON/JSONL and persisted artifacts.

## Delivery stages

1. Freeze upstream provenance and common schemas/invariants.
2. Implement N0 recording adapters and benchmark-safe workspace materialization.
3. Expose each host's existing roles as resumable capability calls.
4. Enable R1--R5 strictly through the shared policy.
5. Add isolated images, one-command smoke runs, and resume-safe batch execution.
6. Freeze task selection before inspecting condition outcomes and publish hashes,
   ledgers, analysis scripts, and failure-complete reports.
