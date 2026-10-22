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
