# Adaptive mobility memory

Research on adaptive personal and collective memory for next-location prediction under behavioral change.

**[工作文档：论文指标、复现结果与后续计划](docs/WORK_LOG.md)** — the maintained project record. It separates published AgentMove scores, accepted preprocessing results, engineering checks and exploratory memory experiments.

**[2026-10-26阶段收尾与恢复交接](docs/CLOSEOUT_20261026.md)** — the user's five selected papers (BudgetMem, ShareMem, CAMEL, NextLocMoE, SHARE) now have full-text reading notes and a comparative synthesis in the local research run. All five PDFs are saved locally; the workflow is ready for slides, with no deck delivered yet. The recurring task remains paused. New experiments remain subject to the literature/direction/results approval gates; prepared controls are not an approval to execute. See [the current work log](docs/WORK_LOG.md).

The completed 200-user prediction comparison uses **local Meta-Llama-3-8B-Instruct** for AgentMove and LLM-ZS, then the same model for ablations. See [the frozen model choice and planned settings](configs/tokyo_llama3_baseline.json). This is a substitute-model framework reproduction; the paper's main table uses GPT-4o-mini.

This repository is being populated prospectively from an existing research workspace. It contains full-city address/session/input verification, completed Tokyo Llama3 framework comparisons, explicit module-ablation reconstructions, a paper-text filtering comparison and a completed 600-user memory-reader diagnostic. Exact paper-score/protocol alignment and stable adaptive memory routing remain unverified. The next evidence/budget controls have passed prompt and cache validation but have no new predictions.

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

Passing this audit establishes structural consistency and POI coverage, not geographic correctness or prediction accuracy. The current address generation uses local Hugging Face FP16 greedy batch-1 inference; equivalence to the official provider has not been established. Map-date and peripheral-index deviations also remain. Full session construction, collective graph and the first prediction comparison have since completed; official ablations remain pending (see the maintained work document).

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

## Official session checkpoint

`scripts/build_session_checkpoint.py` runs the frozen official `get_dataset` loader and the prefix of `get_trajectories` through session grouping, unique session IDs and address normalization. Source hashes are required. An independent chronological scan verifies every session against the anchored 72-hour rule. The manifest records per-user 70/10/20 session partitions, the first 80% used as LLM history, and test sessions with at least four stays. Users with fewer than three sessions are marked excluded. This stage does not construct the prediction dictionaries, apply the final 200-query selection or build CKE.

```bash
timeout --signal=TERM --kill-after=10s 2400s python scripts/build_session_checkpoint.py \
  --source "$JOIN_DIR/Tokyo_filtered.csv" \
  --upstream "$AGENTMOVE" --output "$NEW_SESSION_DIR"
```

Requires Linux, pandas 2.1.4, NumPy 1.26.4, 32 GiB available memory and 5 GiB free disk. It uses two CPU affinity cores at low priority, a 16 GiB address-space limit and 1,800 CPU seconds; no GPU or model calls. The output directory must not exist. `status.json` records progress, `session_rows.pkl` preserves the DataFrame checkpoint, `session_manifest.csv` records boundaries, and `result.json` reports the successful audit and checksums. Only a result with `official_session_checkpoint_pass` is accepted. Pickle is an executable serialization format: load only this trusted, checksum-verified local checkpoint. Checkpoints and per-user manifests remain outside Git.

## Prediction inputs and collective graph

`scripts/prepare_prediction_inputs.py` resumes the verbatim Dataset suffix from the trusted session checkpoint, audits every generated historical/context/target field, executes the frozen SocialWorld graph builder, and uses the official Agent sampler for 200 users with 3–50 eligible test sessions and one query each. It does not run model inference. The input pickle and upstream source hashes are pinned. Unused API imports are excluded using AST extraction; the executed Dataset, graph and selection statements are preserved.

```bash
python scripts/prepare_prediction_inputs.py --launch \
  --checkpoint "$SESSION_DIR" --upstream "$AGENTMOVE" --output "$NEW_INPUT_DIR"
```

Use Linux with pandas2.1.4, NumPy1.26.4 and NetworkX3.3; require48GiB available RAM and10GiB disk. `--launch` creates a new directory, detaches a worker with two CPU affinity cores, nice19,24GiB address-space limit,7200 CPU seconds and9000 wall seconds. Monitor `launch.json`, `status.json`, `run.log` and the supervisor/worker PIDs. Do not repeat the launch for an active run. An absent result is not success; only accept `prediction_inputs_and_graph_pass` after the worker exits.

Outputs include official processed JSON dictionaries, `Tokyo_graph.gml`, the frozen query selection, a descriptive trajectory-count comparison and aggregate `result.json`. Keep raw outputs outside Git. Seed1234 is an explicit reproducibility setting for the upstream DL training sample; the Agent's final selection is ordered. The graph retains upstream inclusion of target-user history, undirected edges and unsorted neighbors. The audit counts selected queries exposed to later collective-history records across users: it does not silently repair this official behavior or claim globally causal evaluation. Resolve/document this and the paper/code filtering differences before comparing scores. Reload the saved graph consistently for subsequent conditions because serialization can affect neighbor iteration order.

## Accepted input checkpoint and evaluator contract

The full Tokyo input preparation completed in726.60s, peakRSS3.42GiB, with zero model calls:10,917 audited inputs,60,053 graph nodes,323,106 edges and200 selected users. All six artifact hashes were rechecked after completion; the detached supervisor did not retain its final exit code. See [aggregate input report](reports/tokyo_prediction_inputs_20261022.json).

All200 query times precede some records in the official collective-history pool. This is a pool-level time violation, not proof that each retrieved neighbor or prediction used a future event. Preserve the original-code condition and separately construct timestamp-valid evidence before making causal or adaptive-memory claims.

Run the synthetic evaluator audit against the frozen upstream file:

```bash
python scripts/audit_official_evaluator.py --source "$AGENTMOVE/evaluate/evaluations.py" --output "$NEW_AUDIT_JSON"
```

It extracts only the hash-pinned evaluator class, uses synthetic IDs and no API or model. Python standard library is sufficient; `np.log2` is supplied by `math.log2` for these scalar cases. [Recorded cases](reports/evaluator_contract_20261022.json) show untruncated official NDCG, retained empty predictions, and denominator loss for missing output keys or unreadable files. Future evaluation must report both official-compatible values and fixed-manifest Acc@1/5 plus strict NDCG@5. Synthetic scores are not reproduction results.

## Timestamp-filtered evidence and the 200-query model run

Follow the frozen [comparison protocol](docs/TOKYO_200_PROTOCOL.md). First prepare time-filtered evidence on CPU (Linux, pandas, NumPy and NetworkX from the existing environment):

```bash
timeout --signal=TERM --kill-after=20s 1800s python scripts/build_causal_bundle.py \
  --inputs "$ACCEPTED_INPUTS" --checkpoint "$SESSION_DIR" \
  --upstream "$AGENTMOVE" --output "$NEW_CAUSAL_DIR"
```

The directory must be new. It uses two CPU affinity cores, nice19 and a24GiB address-space cap. Accept only `causal_bundle_pass`; preserve `causal_bundle.json` and per-query provenance outside Git. The [aggregate evidence audit](reports/tokyo_causal_evidence_20261022.json) records source/time/order checks. Event-time validity does not establish historical validity of the modern map or benchmark membership.

Use absolute paths for all path variables below because detached workers change their working directory. Validate the model transport without loading model weights or making model calls:

```bash
python scripts/run_llama3_baseline.py --launch --preflight-only \
  --upstream "$AGENTMOVE" --inputs "$ACCEPTED_INPUTS" --model "$LLAMA_MODEL" \
  --token-cache "$TOKEN_CACHE" --causal-bundle "$NEW_CAUSAL_DIR/causal_bundle.json" \
  --weight-manifest "$PRIOR_WEIGHT_MANIFEST" --include-official-cke \
  --gpu 0 --output "$NEW_PREFLIGHT_DIR"
```

After `preflight_pass` and exit0, use the same command with a **different new output directory** and omit `--preflight-only` to run200 queries in all three conditions. `--include-official-cke` labels the static graph as a non-causal diagnostic. The default includes causal-CKE AgentMove and LLM-ZS. WKG is cached/shared across framework conditions; its failure does not suppress an otherwise runnable LLM-ZS query.

The runner verifies inputs, prior weight sizes/mtimes and model metadata, uses local-only HF FP16/greedy decoding, and disables network. It requires at least32GiB available RAM and19GiB free on a GPU below10% utilization, checks again before model loading, takes a cooperative GPU lock and caps its CUDA allocator at80%. CUDA virtual mappings preclude a meaningful process address-space cap here; host RAM is checked, not claimed to be hard limited. The supervisor enforces a six-hour timeout and records `exit.json`. Monitor `status.json`, `run.log`, calls and error files; never duplicate an active run. Outputs and failed-query placeholders retain the fixed200 denominator. Completion is not established by launch success. Keep raw outputs local; publish aggregate metrics only after checking all planned conditions and failures.

After successful completion, independently recompute metrics and paired comparisons (Python + NumPy, no model calls):

```bash
python scripts/validate_llama3_results.py --self-test
python scripts/validate_llama3_results.py --run "$COMPLETED_RUN" \
  --bundle "$NEW_CAUSAL_DIR/causal_bundle.json" --output "$NEW_SUMMARY_JSON"
```

Unfinished or failed jobs are rejected before performance is read. The output path must not exist. The validator checks the fixed600 result files and labels, preserves200 users per condition, compares independently computed scores with the runner, and reports paired bootstrap intervals, exact Acc@1 discordance tests with three-pair Holm adjustment and separate WKG costs. See the dated analysis section in the protocol. It does not rerun the model or establish paper-level equivalence.

## Accepted Tokyo 200-query results (2026-10-23)

[Results, paired uncertainty, output diagnostics and costs](docs/TOKYO_200_RESULTS.md): original-CKE AgentMove / event-time-filtered AgentMove / LLM-ZS achieved Acc@1 15.5% / 15.5% / 12.5% and Acc@5 35.5% / 36.5% / 34.5%. All 600 outputs passed independent scoring. Acc@1 differences against LLM-ZS were not statistically significant. This is a substitute-model framework result; paper-level alignment remains open. The completed job must not be relaunched.

## Ablation input audit and next protocol

[The source audit and reconstruction protocol](docs/TOKYO_ABLATION_PROTOCOL.md) document the absence of separate ablation branches in the pinned full-agent prompt function, differences between LLM-ZS and an ablation base, exact reconstruction of400 full prompts, and verification of200 WKG cache entries. The [aggregate audit](reports/tokyo_ablation_input_audit_20261023.json) reports actual section retention after clipping. No new inference was performed. Run `scripts/audit_ablation_inputs.py` with the existing completed run, causal_v2 bundle and pinned upstream source as documented in the protocol. The next ablations will be labeled reconstructions, with an unchanged full-condition identity check before launch.

The subsequent implementation passed all1,200 prompt-variant checks and launched the reconstructed ablations on GPU0 (2026-10-23). It adds1,000 prediction calls and reuses200 full predictions and the accepted WKG cache. See the protocol's launch checkpoint and [configuration](configs/tokyo_ablation_200.json). `run_llama3_baseline.py --ablation-source` enables this mode; `ablation_adapter.py` must be alongside the runner. After completion, `validate_ablation_results.py` recomputes metrics and six fixed contrasts with all200 users per condition. Initial successful calls are not final results.


## Completion verified - 2026-10-23

The GPU0 ablation job completed with exit0, 200 users and 1000 new calls. All 1200 condition outputs passed independent scoring and identity checks; 200 full outputs and WKG caches were reused. Acc@1 for base/STM/WKG/CKE/STM+WKG/full is 13.5/19.0/13.5/13.0/15.0/15.5 percent. None of the six prespecified Acc@1 contrasts survives Holm adjustment. See [final results](docs/TOKYO_ABLATION_RESULTS.md) and the aggregate JSON. This supersedes the running checkpoints above; do not relaunch inference.
