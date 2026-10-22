# Tokyo 200-query Llama3 results — 2026-10-23

The three-condition run completed on GPU 0 with exit code 0. All 600 prediction files, their labels, the fixed 200-user manifest and evidence-bundle hashes passed independent score recomputation. This verifies saved-output scoring, not an independent model rerun. [Machine-readable results](../reports/tokyo_llama3_200_results_20261023.json); [frozen protocol and reproduction commands](TOKYO_200_PROTOCOL.md); [maintained paper comparison](WORK_LOG.md).

## Results

Same 200 users, one query per user, local Meta-Llama-3-8B-Instruct, FP16, greedy decoding, batch 1, output limit 1,000 tokens. Each condition retains all 200 queries in its denominator, including empty predictions. Strict NDCG@5 and the untruncated official NDCG happen to agree on these outputs.

| Condition | Acc@1 | Acc@5 | NDCG@5 |
|---|---:|---:|---:|
| AgentMove, original static CKE | 31/200 (15.5%) | 71/200 (35.5%) | 0.258397 |
| AgentMove, event-time-filtered CKE | 31/200 (15.5%) | 73/200 (36.5%) | 0.262485 |
| LLM-ZS | 25/200 (12.5%) | 69/200 (34.5%) | 0.238575 |

Original static CKE is a non-causal original-code diagnostic. The filtered condition limits evidence events to the last observed context timestamp, but still uses modern addresses and fixed offline cohort membership. Neither condition establishes full historical simulation or paper-level protocol equivalence.

## Paired differences

Differences below are A minus B; Acc intervals are percentage points. Bootstrap resamples paired users 10,000 times with seed 20261023. Acc@1 also uses an exact two-sided discordance test, with Holm adjustment across all three pairs. The analysis code was fixed during inference before intermediate performance was read; this was not an experiment-before-launch preregistration.

| A minus B | Acc@1 difference [95% CI] | Acc@5 difference [95% CI] | NDCG@5 difference [95% CI] | Acc@1 exact p / Holm p |
|---|---|---|---|---|
| Filtered CKE minus LLM-ZS | +3.0 [-2.0, +8.0] | +2.0 [-4.5, +8.5] | +0.02391 [-0.02289, +0.07018] | 0.34493 / 1.0 |
| Filtered CKE minus original CKE | 0.0 [0.0, 0.0] | +1.0 [0.0, +2.5] | +0.00409 [0.0, +0.01033] | 1.0 / 1.0 |
| LLM-ZS minus original CKE | -3.0 [-8.0, +2.0] | -1.0 [-7.0, +5.5] | -0.01982 [-0.06623, +0.02741] | 0.34493 / 1.0 |

Both framework conditions have 17 Acc@1 wins and 11 losses against LLM-ZS; their per-user Acc@1 correctness is identical. This cohort does not establish a reliable framework advantage. The filtered/static Acc@1 bootstrap interval is degenerate because every paired correctness difference is zero; it is not proof of population equivalence. Filtered/static higher-rank differences are small and their intervals include zero. No claim that future information improved scores is supported.

The cohort follows the fixed upstream ordered sampler, not a population random sample. Intervals assume exchangeable users within this benchmark; metric intervals are descriptive, not simultaneous coverage guarantees. The 1,000 calls are not 1,000 independent examples.

## Output diagnostics and costs

| Condition | Execution errors | Empty predictions | Prediction length other than 5 | Raw-response fallback | Malformed ID occurrences | Clipped prompts |
|---|---:|---:|---:|---:|---:|---:|
| Original CKE | 0 | 1 | 5 | 83 | 100 | 129 |
| Filtered CKE | 0 | 1 | 4 | 79 | 105 | 128 |
| LLM-ZS | 0 | 0 | 3 | 97 | 12 | 0 |

Malformed ID occurrences count extracted strings that fail the 24-hex-character format; they are neither the number of affected queries nor a check of venue existence. Raw-response fallback counts a parser fallback marker, not necessarily an empty prediction. Invalid strings and empty predictions were not repaired or dropped. Prompt clipping preserves the frozen upstream GPT-tokenizer threshold and last-6,000-character behavior; it is not a 2,000-Llama-token guarantee. These diagnostics motivate inspecting the interface before attributing the score gap to memory selection, without tuning on this observed cohort.

| Stage | Calls | Input tokens | Output tokens | Inference seconds |
|---|---:|---:|---:|---:|
| Shared WKG | 400 | 125,002 | 77,267 | 2,311.42 |
| Original CKE prediction | 200 | 411,497 | 75,860 | 2,338.68 |
| Filtered CKE prediction | 200 | 411,507 | 76,276 | 2,349.84 |
| LLM-ZS prediction | 200 | 139,529 | 101,135 | 3,011.25 |
| Actual shared execution total | 1,000 | 1,087,535 | 330,538 | 10,011.18 |

Whole-run elapsed time was 10,062.44 seconds (about 2 h 47 min); peak allocated GPU memory was 17.32 GiB. No generation reached the 1,000-new-token cap. Standalone AgentMove costs must include its 400 WKG calls as well as its 200 prediction calls; the two framework conditions shared WKG in this comparison. LLM-ZS does not incur WKG cost. Preprocessing and address-generation costs are separate historical stages.

## Evidence and next experiment

Fixed upstream commit: `61a3e47667f2862c81478bfa442595b766f2f4d6`. The aggregate report records the runner, validator, query manifest, evidence bundle and 600-file output-tree SHA256 fingerprints. Private raw outputs remain in `/mnt/xuyi/agentmove_llama3_200_20261022`; `status.json`, `exit.json`, `evaluation.json` and `validated_summary.json` are the terminal evidence. Do not rerun this completed job.

The paper Tokyo main table uses GPT-4o-mini: AgentMove 18.5% / 46.5% / 0.331 and LLM-ZS 17.5% / 41.0% / 0.299. Those are reference scores, not matched-model targets. Paper/code user filtering, trajectory-count definitions, map provenance and sampling differences remain documented in WORK_LOG.md. Figure 5 does not provide a precise tabulated Llama3 target. This milestone is a substitute-model framework reproduction with completed prediction results, not recovery of the paper's reported score.

Next, freeze and run module ablations under the same model and query protocol (base, STM, WKG, CKE, STM+WKG, full framework), first verifying each upstream prompt's actual module composition. Reuse only demonstrably identical cached components; do not assume LLM-ZS equals the ablation base. Keep the original static-CKE comparison and event-time-filtered research setting explicitly separate. Tokyo ablations are additional experiments, not reproduction of the paper's Table 2 city rows. No new inference job was started during this result-verification step. Later memory-selection development requires a separate development split; these 200 queries are now observed evidence.
