# Adaptive mobility memory

Research on adaptive personal and collective memory for next-location prediction under behavioral change.

This repository is being populated prospectively from an existing research workspace. It currently contains address-processing verification code, **not a complete AgentMove reproduction** or a validated memory-routing method.

## Final Tokyo address audit

`scripts/audit_final_l4.py` checks the final 3,090-response batch, previously completed batch IDs, and exact coverage of the 83,190 frozen Tokyo POI requests. It verifies paired ordering, request equality, prompt hashes, token counts, normal stop reasons, duplicate IDs, official join errors, normalization and recorded costs. It records output checksums. Earlier batches' detailed request/normalization audits are separate evidence; this final script does not repeat all of them.

Run on Linux with Python 3, pandas and NumPy, in the existing reproduction environment (without `python -O`). It limits itself to two available CPUs and lowers its scheduling priority. No GPU/model inference is performed.

```bash
python scripts/audit_final_l4.py \
  --run "$L4_ROOT/chunk017" \
  --seed "$SEED_DIR" \
  --requests "$FROZEN_REQUESTS" \
  --normalization "$FROZEN_NORMALIZATION" \
  --output "$AUDIT_DIR/completed_audit.json"
```

Create the audit directory first; existing output files are never overwritten. The requests must match the frozen SHA256 embedded in the script. `run_result.json` records the prior batch directories, which must remain accessible. The seed contains 100 prior responses. Each batch contains `raw_responses.jsonl`, `parsed_responses.jsonl`, `selection.json`, `run_result.json`, and `events.jsonl`, with a sibling batch log. Paths are supplied by the operator and are not server credentials.

**Trusted input:** the normalization JSON contains a `statement` copied from the audited official preprocessing code and executes as Python. Only supply the trusted frozen project artifact, never a downloaded or unreviewed statement. The upstream code revision is `61a3e47667f2862c81478bfa442595b766f2f4d6`.

Passing this audit establishes structural consistency and POI coverage, not geographic correctness or prediction accuracy. The current address generation uses local Hugging Face FP16 greedy batch-1 inference; equivalence to the official provider has not been established. Map-date and peripheral-index deviations also remain. Full session construction, collective graph, prediction evaluation and official ablations are still pending.

Credentials, model weights, raw datasets and large model outputs are intentionally external to this repository. Existing exploratory memory experiments have not established a stable benefit from dynamic source selection.

## Full-city address join

`scripts/build_city_join.py` joins all 83,190 accepted Tokyo POI address records to the frozen city check-ins. It executes the normalization/join/export statements extracted from the checksum-verified official `processing/trajectory_address_match.py`, without its API imports or generation loops. It checks exact POI coverage, zero dropped check-ins, preservation of original columns and CSV round-trip consistency. Missing address field values retain upstream behavior.

Run on Linux with pandas 2.1.4, without `python -O`. Use a **new**, nonexistent output directory under an existing parent. The worker requires 16 GiB available RAM and 2 GiB free disk; it bounds itself to two CPU cores, 8 GiB process address space and 900 CPU seconds. No GPU or model calls are needed. Apply an outer wall-time limit:

```bash
timeout --signal=TERM --kill-after=10s 960s python scripts/build_city_join.py \
  --source "$PREPROCESSED_CITY/Tokyo_filtered.csv" \
  --upstream "$AGENTMOVE/processing/trajectory_address_match.py" \
  --seed "$SEED_DIR" --chunks "$L4_ROOT" --output "$NEW_JOIN_DIR"
```

Inputs are the frozen raw-city CSV, seed responses and `chunk001` through `chunk017` (with terminal `run_result.json` and `parsed_responses.jsonl`). The source and upstream checksums are pinned in the script. Outputs are `Tokyo_addr_dict.json`, the address-enriched `Tokyo_filtered.csv`, and `result.json` with counts, checksums, resource measurements and input manifests. Raw inputs and enriched check-ins stay outside Git. A passing join does not establish geographic correctness; session construction, prediction and ablations remain separate steps.
