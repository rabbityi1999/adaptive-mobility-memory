# Tokyo module ablations: source audit and reconstruction protocol

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run (offline input audit and next-run specification)
- Origin Date: 2026-10-23 (Asia/Shanghai)
- Verification Status: ANALYZED
- Version Label: ablation_input_audit_v2

## Completed verification

[Aggregate audit](../reports/tokyo_ablation_input_audit_20261023.json) and [audit script](../scripts/audit_ablation_inputs.py) establish the following for the accepted 200-user run:

1. All 600 saved prediction files still match the previously validated output-tree hash. All recorded effective prompts match their prompt hashes and the original clipping operation. No prediction was rerun, repaired or rescored.
2. All 400 full-framework prompts can be reconstructed exactly from the pinned prompt function, deterministic Memory readout and cached spatial information. For the original static-CKE condition, the CKE string is taken from the accepted original prompt; this audit does not repeat graph construction. For the filtered condition it comes from the accepted time-filtered bundle.
3. All 200 WKG cache entries match the two corresponding generation outputs and the pinned SpatialWorld formatter. They are eligible for reuse without new WKG calls, provided the manifest, input, model and source hashes remain fixed.
4. The pinned `prompt_generator_agent` implements only `agent_move_v6`. The full prompt reads `historical_info` and `user_profile` from Memory; the computed `context_info` field is not inserted as a separate summary. Recent context stays remain in the history-data section.
5. LLM-ZS includes historical/context times and IDs, whereas the full framework also has category/address fields and a different instruction template. Its prompt differs from the empty-module full template for all 200 queries. LLM-ZS is not an interchangeable ablation base.

The server source directory has no resolvable Git HEAD; provenance is established here by exact file hashes against the previously pinned source snapshot, not a fresh Git checkout claim. This audit concerns that snapshot, not unpublished author code. In particular, it cannot determine the exact prompts behind the paper's Table 2. Reconstructed ablations must be labeled as such.

## Actual retained evidence

Counts below describe the original non-whitespace payload span after the recorded prompt clipping; they do not measure attention or comprehension. “Empty before clipping” is distinct from evidence removed by clipping. A lost span does not imply that all overlapping facts elsewhere in the prompt disappeared.

| Condition / payload | Fully retained | Partially retained | Completely removed | Empty before clipping |
|---|---:|---:|---:|---:|
| Original CKE / WKG |172|18|10|0|
| Original CKE / CKE |185|4|5|6|
| Original CKE / STM summaries |195|5|0|0|
| Filtered CKE / WKG |172|18|10|0|
| Filtered CKE / CKE |184|4|5|7|
| Filtered CKE / STM summaries |195|5|0|0|

All 200 history-data and output-instruction payloads remain intact in both framework conditions. The ten WKG losses and five CKE losses are counts within each 200-query condition; do not sum them as independent users or infer their intersection from these aggregates. The initial audit mistakenly required every source payload to be nonempty and stopped on an empty CKE block; the corrected audit distinguishes empty source from clipping loss, passed boundary checks and exited 0. The failed audit did not write a result or call a model. Prior predictions and scores were unchanged.

## Next inference: explicitly reconstructed ablations

Use the same accepted 200 users, local Llama3 FP16 greedy model, model/tokenizer files, ordered manifest, time-filtered CKE bundle and WKG cache. This is an observed benchmark cohort; no prompt selection or tuning based on its outcomes, and no claim of new independent validation. There is no published Tokyo Table 2 row to replicate.

The controlled base retains the full framework's task, address/category-enriched history and context, masked target and output specification. Remove only module sections and their module-specific instructions. It therefore remains a personal-history baseline, not a memory-free or LLM-ZS baseline.

| Reconstructed condition | STM summary/profile | WKG | Filtered CKE | Prediction execution |
|---|---|---|---|---|
| base |off|off|off|200 new calls|
| +STM |on|off|off|200 new calls|
| +WKG |off|on|off|200 new calls|
| +CKE |off|off|on|200 new calls|
| +STM+WKG |on|on|off|200 new calls|
| full |on|on|on|reuse 200 accepted filtered-CKE outputs after exact prompt/input/model/hash checks|

Keep the original static-CKE result and LLM-ZS result as separately labeled references. Do not silently mix original and filtered CKE within the ablation table. Do not modify the original upstream files.

Reconstruction rule: starting from the full `agent_move_v6` prompt, remove the entire WKG/CKE/STM section, including its heading and payload, when disabled. Remove instruction 3 when WKG is disabled and instruction 4 when STM is disabled. Preserve all other text, relative order, raw history fields and prediction parsing. For full, the transformation must be the identity, byte for byte. Build this adapter and prove the full identity and disabled-section exclusion for all 200 queries before launching inference. Record each condition's original and effective prompt hash and retained payload spans; preserve the existing GPT-token threshold and last-6,000-character clipping.

Deleting a section changes prompt length and can change how much other evidence survives clipping. Consequently these comparisons estimate the effect of the implemented module-plus-prompt configuration under the existing interface; they are not a perfectly token-matched attribution of a memory mechanism. Do not tune a new clipping policy on this benchmark. An interface change belongs to a separate development protocol.

Budget: 1,000 new prediction calls, zero new WKG generations, with 200 full predictions reused. Distinguish incremental execution cost from standalone condition cost (include historical WKG generation for any standalone WKG-enabled condition). Each condition has a fixed denominator of 200; keep empty/error outputs as misses. No correctness-based retries. Prefer GPU0 after a fresh availability check and cooperative lock; retain the established two-CPU/nice19 and six-hour execution limit.

Before launch, the new runner must also verify the accepted run's terminal status, manifest, output-tree hash, source hashes, WKG cache hashes and model fingerprint evidence. A validation or launch failure is not a completed experiment; retain diagnostics and do not duplicate active jobs. At the initial source-audit checkpoint the adapter was not implemented; the implementation and launch checkpoint below supersede that status.

## Analysis fixed for the next run

Report Acc@1, Acc@5 and strict NDCG@5 for all six conditions, plus untruncated official NDCG, empty/malformed/parse-fallback counts, retained evidence spans, tokens and time. Use all fixed users, not only successful or covered cases.

Predefine six comparisons: each of +STM, +WKG, +CKE, +STM+WKG and full against the controlled base, and full against +STM+WKG. Report paired-user percentile bootstrap 95% intervals with 10,000 draws and seed20261023 for all three metrics; Acc@1 exact discordance p-values with Holm adjustment over these six comparisons. Intervals are descriptive, not simultaneous; the ordered sample is not a random population sample. Disclose effect sizes and all planned comparisons regardless of significance. No new routing-method claim follows from module ablations alone.

## Audit reproduction

Use the existing Python/pandas environment (pandas2.1.4); no GPU or model load. Output must be a new path. Only aggregate counts/hashes are written.

```bash
python scripts/audit_ablation_inputs.py --self-test
PYTHONHASHSEED=100 python scripts/audit_ablation_inputs.py \
  --run "$COMPLETED_RUN" --bundle "$CAUSAL_V2_BUNDLE" \
  --upstream "$PINNED_UPSTREAM" --output "$NEW_AUDIT_JSON"
```

Inputs are trusted local completed-run artifacts, including `validated_summary.json`, the 600 predictions, `calls.jsonl` and `wkg_cache.jsonl`. The script executes only pinned pure prompt/Memory/formatter definitions through AST extraction, blocks network, and uses two CPU-affinity cores at low priority. It does not verify map accuracy or cure the outstanding paper/code filtering and trajectory-count mismatch.

## Implementation and launch checkpoint — 2026-10-23 09:30 CST

The [adapter](../scripts/ablation_adapter.py) and optional `--ablation-source` mode in the existing runner implement the frozen deletion rules. Default three-condition runner behavior is preserved when that option is absent. All 1,200 prompt variants passed the offline preflight; all 200 full original/effective prompt identities, tokenizer lengths and source model/software/hash checks matched the prior run. Source predictions and WKG were not modified. [Aggregate preflight](../reports/tokyo_ablation_preflight_20261023.json), [configuration](../configs/tokyo_ablation_200.json).

The preflight exited0 without loading weights or making model calls. At launch GPU0 had24,121MiB free and0% utilization. Inference is now running on GPU0, supervisor1618786 / worker1618806, output `/mnt/xuyi/agentmove_ablation_200_20261023`. Its first user completed all five new conditions (5 calls) with no observed execution error. This is an initial health checkpoint, not a completed result. Full-condition files in the new directory are explicitly reused, even though their original content retains `execution_status=completed`.

Run with absolute paths, a new output directory and the same accepted baseline source:

```bash
python scripts/ablation_adapter.py
python scripts/run_llama3_baseline.py --launch --preflight-only \
  --upstream "$PINNED_UPSTREAM" --inputs "$ACCEPTED_INPUTS" --model "$LLAMA_MODEL" \
  --token-cache "$TOKEN_CACHE" --causal-bundle "$CAUSAL_V2_BUNDLE" \
  --weight-manifest "$WEIGHT_MANIFEST" --ablation-source "$COMPLETED_BASELINE" \
  --gpu 0 --output "$NEW_ABLATION_PREFLIGHT_DIR"
```

Only after `preflight_pass` plus exit0, remove `--preflight-only` and provide a **different new** output directory to launch inference. The runner repeats the gates and all identities, takes the cooperative GPU lock, preserves the existing six-hour timeout and records `status.json`, `exit.json`, a per-query prompt manifest and each successful call's retained payload spans. No WKG model calls are made. The adapter and runner must be in the same directory.

After terminal completed status and exit0:

```bash
python scripts/validate_ablation_results.py \
  --run "$COMPLETED_ABLATION_RUN" --output "$NEW_VALIDATED_SUMMARY_JSON"
```

The independent validator checks1,200 files, exact source-result reuse, labels and the frozen prompt manifest, and recomputes all metrics and six predefined contrasts. It separates new execution costs from historical reused full/WKG costs. Keep `ablation_adapter.py` and `validate_llama3_results.py` beside it. The unfinished-run gate was exercised successfully: it exited1 before reading performance and wrote no summary. Analysis implementation was completed after launch with no interim accuracy read, while the contrast family and statistical settings were already fixed in the preceding protocol.

Retention changes from deleting sections are visible before inference: WKG alone is fully retained in199 cases and partial in1; STM+WKG retains WKG fully182, partially13 and loses5; full retains it fully172, partially18 and loses10. CKE alone preserves all193 nonempty payloads (7 empty source). This verifies a prompt-length interaction, not an accuracy improvement. The adapter does not change the clipping rule in response to these diagnostics.

## Independent synthetic acceptance checks — 2026-10-23 12:25 CST

While GPU0 continued inference (175/200 users,875 calls at the latest check), four local tests passed without reading real partial predictions or changing runner/validator code. The synthetic completed fixture contains200 users per arm with analytically known rank-one hits, one failed new prediction and200 reused full outputs. It checks exact expected scores, failure retention in the200 denominator, six planned contrasts and a known exact/Holm p-value. Other tests ensure unfinished jobs are rejected before metadata/scores are read, changed labels fail, and modified reused output fails even if its score is unchanged. These are software acceptance checks, not additional empirical results.

```bash
python -m unittest discover -s tests -p test_ablation_validation.py -v
```

Requires Python and NumPy; fixtures are generated in an automatically cleaned local temporary directory. No model, network, server credentials or real trajectories are used. All four tests passed in23.60seconds. The earlier implementation commit1b35112 was subsequently pushed and independently verified on remote main.


## Completion verified - 2026-10-23

The GPU0 ablation job completed with exit0, 200 users and 1000 new calls. All 1200 condition outputs passed independent scoring and identity checks; 200 full outputs and WKG caches were reused. Acc@1 for base/STM/WKG/CKE/STM+WKG/full is 13.5/19.0/13.5/13.0/15.0/15.5 percent. None of the six prespecified Acc@1 contrasts survives Holm adjustment. See [final results](TOKYO_ABLATION_RESULTS.md) and the aggregate JSON. This supersedes the running checkpoints above; do not relaunch inference.
