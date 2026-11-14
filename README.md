# Free model aggregator

An entry for make.fun competition `free-model-aggregator-01`: the best
free AI model aggregator.

## What it does

A single-file Python 3.13 server (standard library only, `main.py`) that
serves two client APIs — OpenAI Chat Completions and Anthropic Messages,
JSON and streaming — over ten free-tier providers on three wires
(OpenAI, Anthropic, Gemini). It:

- reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider
  directory, the sealed quality table and the workload manifest;
- probes every provider's catalog route in its own wire and builds the
  one catalog document (`python3 main.py --list` prints it, and the
  register body carries the same document, byte for byte);
- registers its port and catalog with the env, serves the client's
  twenty requests one at a time, and exits 0 when the env says the
  workload is done;
- ranks the free models meeting each request's requirements by the
  quality table's rating, tries providers in that order (one attempt
  per provider per request), and fails over to the next candidate when
  a provider refuses. It learns budgets online: a provider that
  answers "budget is spent" or "is out" is treated as dead for the
  rest of the trial, while a scheduled refusal is transient and never
  kills a provider. A shadow budget keeps the best providers from
  being burned on light requests while heavier ones are still to come;
- translates requests and answers between wires, preserving exactly
  what the provider echoes back (system prompt, user text, tool
  names, max_tokens, stop sequences, image bytes, JSON mode), and
  returns the provider's usage counts and stop reason mapped to the
  client's wire.

Every choice is a function of the brief, the requests and the
providers' answers so far: no wall clock, no unseeded randomness.
Two runs of the same trial leave byte-identical state exports.

## Local grading

Practice scores with the kit's own env and scorer (no Docker needed):

```sh
./run-trial.sh 0   # practice trial 0 (also 1 and 2)
```

Results on the three practice trials:

| trial | score | quality | served | wire | failover | catalog |
|-------|-------|---------|--------|------|----------|---------|
| 0     | 99    | 99      | 100    | 100  | 100      | 100     |
| 1     | 99    | 99      | 100    | 100  | 100      | 100     |
| 2     | 99    | 99      | 100    | 100  | 100      | 100     |

Quality 99 is the practice ceiling: the providers' failure schedules
force a handful of fallbacks just below the best rating on every
instance (a 100 would need every request served by its best model,
which the schedules make impossible).

## Files

- `main.py` — the entrypoint (the whole server, one file)
- `LICENSE` — MIT
- `.makefun/owner.json` — the entrant's payout wallet
