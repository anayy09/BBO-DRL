# BBO-DRL Q1 Elevation: Round 2 Prompt

## Context

Fix.md has been executed. The paper now has 30-run Monte Carlo, BBO-only and DQN-only ablations,
Local-Only SLA bug fixed, Wilcoxon+Bonferroni significance testing, N=5000, CI weight ablation,
ε-convergence figure, Privacy Guard ROC, and MIT-BIH real-trace evaluation.

The issues below are what remains between the current paper and a defensible Q1 submission.
Do not change the algorithm design. Do not fabricate numbers. Every result must come from
actual simulation runs.

---

## Fix A: Add PSO+DQN hybrid baseline

This is the highest-priority item. Table III currently shows BBO-only is statistically
indistinguishable from PSO on all metrics. A reviewer will immediately ask: if the coupling
is what drives the privacy advantage, why not PSO+DQN? Without this baseline, the argument
that BBO specifically contributes something is not defensible.

**Implementation:**
- Build a PSO+DQN variant using the same structure as BBO-DRL (Algorithm 1), with one change:
  replace the BBO inner search (lines 4-13) with a PSO search over the same K-node subspace AK.
- Use the same PSO hyperparameters as the standalone PSO baseline (same swarm size, same
  update rule, same number of iterations). Do not re-tune.
- Keep the DQN architecture, state, reward, and top-K pre-filtering identical to BBO-DRL.
- Run PSO+DQN at all five task scales (N in {100, 500, 1000, 2000, 5000}) with 30 Monte Carlo
  replicates. Use the same seed protocol: sr = 42 + 1000r + N.
- Add PSO+DQN to Table III, Table V (MIT-BIH), Figure 2 (bar chart), Figure 3 (latency vs scale),
  and Figures 5-6 (Pareto plots).
- Run Wilcoxon rank-sum tests between BBO-DRL and PSO+DQN on all four metrics at N=1000.
  Apply Bonferroni correction with the updated family size (6 baselines × 4 metrics = 24 tests).
  Update Table III footnote to reflect family size 24.
- Update results/table3_n1000.csv to include PSO+DQN row.

The expected finding: if BBO-DRL and PSO+DQN perform similarly, report it honestly and
reframe the contribution as "hybrid DRL+bio-inspired search" rather than BBO specifically.
If BBO-DRL outperforms PSO+DQN on privacy, that is a strong result. Report what the data shows.

---

## Fix B: Run the all-high-CI ICU scenario for weight function ablation

The current Table IV shows the four weight schemes (flat, step, linear, non-linear) are
statistically indistinguishable at N=1000 on a mixed-CI workload (low 20%, medium 60%, high 20%).
The paper's discussion section speculates this separation would appear on a workload skewed to
high CI. Test that claim.

**Implementation:**
- Create a new task stream configuration: CI distribution = high 100% (all tasks arrive with
  Phi_i drawn uniformly from [0.8, 1.0]). All other parameters stay the same (N=1000,
  same four task types by proportion, same hardware).
- Run all four weight conditions (flat, step, linear, non-linear) using BBO-DRL on this
  all-high-CI stream, 30 Monte Carlo replicates each.
- Run Wilcoxon rank-sum between non-linear and each other scheme on all four metrics.
  Apply Bonferroni correction (3 schemes × 4 metrics = 12 tests).
- Report results in a new Table (Table VI): "CI Weight Scheme Ablation under All-High-CI Load
  (Phi in [0.8, 1.0], N=1000, 30 runs)." Same format as Table IV.
- Add a new sub-figure to Fig 7 showing the all-high-CI ablation bars alongside the mixed-CI bars.

If the non-linear scheme still does not separate from flat/step/linear on the all-high-CI
workload, remove the non-linear CI weight functions from the paper's contributions list.
Report the full negative result in Section V-F. Do not tune parameters to manufacture separation.

Deliverable: results/table6_highci_weights.csv and updated figures/fig7_weight_ablation.pdf
(two panels: mixed-CI and all-high-CI side by side).

---

## Fix C: Measure and report wall-clock scheduling latency

The paper claims O(N_pop * T_BBO * M) complexity is "suitable for real-time clinical deployment"
but no timing data is reported. This claim must be grounded in measurements.

**Implementation:**
- Instrument the scheduler loop in Algorithm 1 to record wall-clock time from line 1
  (state observation) to line 15 (dispatch decision). Use Python's time.perf_counter().
  Do not include the Bellman update (lines 16-18) — that happens after dispatch.
- For each of the 30 runs at N=1000, record the per-task scheduling time for every task.
  This yields 30,000 timing samples total.
- Compute mean, median, 95th percentile, and max scheduling time across all tasks and runs.
- Do the same for PSO+DQN and BBO-only at N=1000 for comparison.
- Add a row to Table I or a new small table (e.g., after Table I): "Per-task scheduling overhead"
  with columns: Algorithm, Mean (ms), p95 (ms), Max (ms).
- Report these numbers in Section V-A (Simulation Setup) and reference the SLA deadline
  (150 ms for ECG). If mean scheduling overhead is under 5 ms, the real-time claim holds.
  If it is above 10 ms, the text needs to qualify the claim.

Deliverable: results/scheduling_overhead.csv with columns [algorithm, run, task_idx, time_ms].

---

## Fix D: Fix the AUC inconsistency in Figure 8

Figure 8 caption reads "AUC = 0.987". Section V-G text says AUC = 0.995. One of these is wrong.

**Action:**
- Re-run the MedSec-25 Privacy Guard evaluation from scratch using the exact same stratified
  sample (9,998 flows: 9,776 attack, 222 benign) and the entropy threshold tau = 0.55.
- Compute AUC using sklearn.metrics.roc_auc_score on the full detection channel output
  (not just the binary prediction at tau). This gives the correct AUC.
- Also recompute TPR, FPR, precision, F1 at tau = 0.55.
- Regenerate Figure 8 with the correct AUC in the legend and caption.
- Update Section V-G text to match.

Do not choose whichever value is higher. Report what the computation produces.

Deliverable: updated figures/fig8_privacy_guard_roc.pdf, updated results/medsec25_guard.csv
with columns [threshold, TPR, FPR, precision, F1, AUC].

---

## Fix E: Report per-comparison p-values in Table III

The current Table III caption says "all reported p-values are < 10^-3." This is a blanket
statement, not per-pair reporting. Q1 venues expect individual p-values.

**Action:**
- For every (algorithm, metric) pair in Table III, compute the Bonferroni-corrected p-value
  from the Wilcoxon rank-sum test between that row and BBO-DRL.
- Format each cell as: mean ± std (p=X.XXe-Y vs BBO-DRL).
- If all corrected p-values are below 1e-3, they may still be reported as "< 10^-3" per pair
  in the table body, but the footnote must state the correction method and family size
  explicitly (24 tests after adding PSO+DQN).
- Regenerate the table in LaTeX. Update results/table3_n1000.csv to include a p_corrected
  column for each metric.

---

## Fix F: Measure and report scheduling overhead decomposition

The paper reports end-to-end task latency (transmission + queue + compute at destination node)
but does not separate how much of BBO-DRL's 74.9 ms comes from the BBO inner search versus
the physical routing.

**Action:**
- Decompose mean end-to-end latency at N=1000 for BBO-DRL into three components:
  (1) transmission latency Di/Ri,j, (2) queuing delay T_queue, (3) compute at destination Ci/fj.
- Report these three components for BBO-DRL and PSO as a stacked bar in a new subfigure
  or a supplementary table.
- This clarifies whether BBO-DRL's latency penalty over PSO comes from routing to a slower
  node (compute component) or a more distant node (transmission component).

Deliverable: results/latency_decomposition.csv with columns
[algorithm, N, mean_tx_ms, mean_queue_ms, mean_compute_ms, mean_total_ms].

---

## Fix G: Add explanation comments to simulation code for two anomalies

Two results in the current paper will draw reviewer questions that require in-code documentation
to trace and in-text explanation to answer. No new runs are required for this fix — only
documentation and in-text clarification.

**Anomaly 1: DQN-only energy (35.17 mJ vs BBO-DRL 7.36 mJ).**
- Trace through the simulation: when DQN-only is cold (epsilon near 1.0), what nodes does
  it route to? If it is routing to cloud during exploration (random action), the transmission
  energy Ei_offload = Pt * Ttx_ij is high because Ttx is large for cloud. Log the per-run
  mean node selection distribution for DQN-only at N=1000 early vs late in the run.
- Produce results/dqn_only_routing_dist.csv: columns [run, task_quintile (1-5), node_fraction_0
  through node_fraction_5] showing where DQN-only routes tasks across five equal-sized
  segments of each run. This should show early-run routing to high-energy nodes.
- Add one paragraph to Section V-D explaining the energy mechanism with reference to this data.

**Anomaly 2: MIT-BIH zero SD for stateless baselines (Table V).**
- Confirm in the code that PSO, ACO, and HS-HHO always route to cloud for every task in the
  MIT-BIH trace. If confirmed, add a footnote to Table V: "SD = 0.00 for PSO/ACO/HS-HHO/Cloud-
  Only because the all-ECG trace with uniform 150 ms deadline causes all stateless optimizers
  to converge to cloud routing on every run; variance enters only through channel re-seeding,
  which has negligible effect on cloud-link latency in the model."
- Do not patch the SD values. Document the behavior.

---

## Constraints

- All results must come from simulation runs. No placeholder values.
- Fix A (PSO+DQN) is highest priority. Do not skip it.
- If Fix B produces a null result on all-high-CI (non-linear still does not separate), document
  that explicitly and note in the deliverable log that the non-linear contribution claim should
  be removed from the paper.
- Fix the AUC before updating any text. The text should reflect the measured value, not the
  other way around.
- Maintain seed protocol: sr = 42 + 1000r + N. Log seeds for all new runs.
- If PSO+DQN proves to match BBO-DRL on privacy, that changes the framing of the paper's
  main claim. Flag this in a deliverable note (deliverables/framing_note.txt) and do not
  silently continue. The manuscript rewrite will need to adjust.

---

## Deliverables

1. results/table3_n1000.csv — updated with PSO+DQN row and per-comparison p-values.
2. results/table5_mitbih_trace.csv — updated with PSO+DQN row.
3. results/table6_highci_weights.csv — all-high-CI weight ablation (30 runs).
4. results/medsec25_guard.csv — recomputed TPR, FPR, precision, F1, AUC at tau=0.55.
5. results/scheduling_overhead.csv — per-task wall-clock overhead for BBO-DRL, PSO+DQN, BBO-only.
6. results/latency_decomposition.csv — tx/queue/compute breakdown at N=1000.
7. results/dqn_only_routing_dist.csv — per-quintile node routing fractions for DQN-only.
8. figures/fig2_metric_comparison.pdf — updated bar chart with PSO+DQN.
9. figures/fig3_latency_vs_scale.pdf — updated with PSO+DQN curve.
10. figures/fig5_energy_latency_pareto.pdf — updated with PSO+DQN point.
11. figures/fig6_latency_privacy_pareto.pdf — updated with PSO+DQN point.
12. figures/fig7_weight_ablation.pdf — two-panel: mixed-CI and all-high-CI side by side.
13. figures/fig8_privacy_guard_roc.pdf — regenerated with correct AUC.
14. deliverables/framing_note.txt — assessment of whether PSO+DQN result requires reframing
    the main claim, and whether non-linear weight contribution should be removed.

Do not update the manuscript. Produce numerical results and figures only.
The manuscript rewrite follows once these results are confirmed.
