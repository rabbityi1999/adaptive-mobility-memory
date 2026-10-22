# Tokyo 200-query Llama3 comparison — 2026-10-22

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-22 (Asia/Shanghai)
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

## Question and fixed comparison

Measure the frozen AgentMove framework with local Meta-Llama-3-8B-Instruct against LLM-ZS on the same 200 selected Tokyo users, and measure the effect of limiting CKE evidence to records already observed at prediction time. No routing mechanism or new-model superiority is claimed by this experiment.

Keep the accepted query manifest, SHA256 `526e73882604a4ebdb6632c3f3b0bfcf42ef842953299d5ee82a29fad1bddefb`, unchanged: one query per user, fixed upstream 3–50 eligible-session rule and ordered selection. Use the accepted address/session/input checkpoint. These are benchmark queries; do not describe them as a newly untouched holdout or tune parameters using their outcomes.

Conditions:

1. `agent_move_v6_officialCKE`: original static CKE, labeled **non-causal original-code diagnostic**. This retains the audited later-history exposure and cannot support strict temporal research claims.
2. `agent_move_v6_causalCKE`: same official prompt, personal memory and WKG, with CKE evidence limited to the last observed context timestamp.
3. `llmzs`: original LLM-ZS prompt, same model and query inputs.

Cache each query's two WKG responses once and share them between framework conditions. Record WKG generation costs separately; do not charge unused WKG to LLM-ZS or claim an independent rerun of WKG per condition. Source changes, model transport, fixed hash seed, serialized graph reload and cached calls are explicit execution choices.

## Temporal evidence construction

Start from the fixed official historical corpus; preserve its offline user/session membership. For each query, retain only check-ins at or before the last observed context check-in. Filter transition events using both endpoint timestamps. Build neighbor order solely from retained events in the original corpus order; exclude context venues. Select node attributes from the final retained row in official iteration order. Preserve the source's effective eleven-neighbor limit (`max_neighbors=10` breaks after appending the eleventh). This is an explicitly modified CKE condition, not the same graph as the official control.

Match the official GML save/reload adjacency order using node insertion ranks computed from retained events. The unfiltered version must reproduce original retrieved neighbor order for every query. The optimized incident-edge procedure is independently checked against full filtered graph construction, GML round-trip and unchanged upstream formatter for query indices 0, 99 and 199. Every returned edge and node-attribute source must satisfy the cutoff. Save per-query provenance locally, publish aggregates only. Modern address metadata and offline cohort membership remain fixed; this is event-time filtering, not a fully historical city-map reconstruction.

Audit the original returned neighbors as well: edges only first observed after cutoff, edges with later contributions, and node attributes whose final assignment comes from later records. A later assignment does not establish that its attribute value differs from an earlier observation. Do not convert these counts into accuracy inflation claims.

## Model and execution

Local HF FP16, greedy decoding, batch size 1, seed100, maximum1000 new tokens, tokenizer chat template and EOS/end-of-turn stops. Preserve upstream token-count clipping: when its GPT tokenizer counts more than2000 tokens, keep the last6000 characters. Log both original/effective prompt and Llama token lengths. Do not equate this behavior to a hard2000-Llama-token input cap.

Prefer GPU0, then other authorized available GPUs; inspect free memory and utilization before launch, do not interrupt other workloads. Use existing weights and environment, no paid APIs. Record code/model fingerprints, source hashes, software versions, time, memory and each call. Keep runtime bounded and check process health plus progress. Failed outputs stay in the planned200-query denominator; no response retries selected by correctness.

## Reporting

Report per condition Acc@1, Acc@5, strict NDCG@5 on the frozen manifest, alongside official-compatible untruncated NDCG and parsing behavior. Count missing/failed planned predictions as misses; separately disclose invalid IDs, empty lists, non-five-item responses, runtime failures and truncation. Official evaluator edge cases are recorded in `reports/evaluator_contract_20261022.json`.

Once outputs are complete, compute paired differences and user-level bootstrap intervals on the same200 users, plus discordant counts for Acc@1. No significance or improvement claim from partial progress. Report shared and condition-specific token/time costs. Historical five-query engineering checks and memory-pilot outputs remain separate.

The main paper uses GPT-4o-mini; current Llama3 results are a substitute-model framework study. Paper/code filtering and trajectory-count discrepancies, present-day map metadata and local transport differences remain disclosed in WORK_LOG. This protocol does not resolve them or justify paper-level equivalence.

## Analysis implementation fixed on 2026-10-23

While inference was still running (144–147/200 queries), the analysis implementation was fixed without reading interim performance. This is a dated implementation decision, not a claim of preregistration before model outputs existed. Use `scripts/validate_llama3_results.py` only after completed status and exit0. It verifies600 expected result files against the frozen query list and labels, recomputes four metrics independently from saved predictions, and compares them with the runner's final report.

For all three condition pairs, use paired user bootstrap with seed20261023 and10,000 draws; report percentile95% intervals for Acc@1, Acc@5 and NDCG@5 differences. Report exact two-sided discordance tests for Acc@1 and Holm-adjusted p-values across the three pairs. The metric confidence intervals are descriptive, not multiplicity-adjusted simultaneous intervals. The ordered cohort is not a random population sample; bootstrap uncertainty assumes exchangeable users conditional on this cohort. Report effect sizes and discordant counts regardless of p-values. No tuning, selective subgroup claims or significance claims from partial data.

Audit per-stage successful call counts, token totals, clipping and time, retaining WKG separately. Distinguish an output-score recomputation from repeating model inference. Synthetic rank6/empty/invalid extraction and failure-denominator checks passed; an attempted analysis of the active run was correctly rejected before reading predictions, with no summary written.
