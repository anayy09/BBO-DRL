# BBO-DRL Q1 Elevation: Claude Code Prompt

## Context

The BBO-DRL paper has a working simulation, trained CI module, and baseline results. The issues below are not about rewriting — they are specific, targeted fixes to the simulation code and experimental pipeline that are required to produce defensible Q1 results. Do not change the algorithm design. Do not fabricate numbers. Every result must come from actual simulation runs.

---

## Fix 1: Resolve the Monte Carlo contradiction

The paper currently runs **5 Monte Carlo trials per configuration** but the conclusion claims **30 runs**. Choose one and enforce it everywhere.

**Action:**
- Set `N_RUNS = 30` globally in `config.py` (or equivalent).
- Re-run all experiments: all baselines, all task scales (100, 500, 1000, 2000, and add 5000).
- Update Table III, all figures, and all text references to reflect the actual run count.
- Re-compute all means and standard deviations from scratch. Do not carry over old numbers.

---

## Fix 2: Add BBO-only and DRL-only ablation baselines

These two baselines are mandatory for any hybrid paper. Add them to the simulation:

**BBO-only:**
- Same BBO population and TBBO as the full system.
- No DQN. Replace the top-K pre-filtering step (line 3 in Algorithm 1) with a full search over all candidate nodes (A_K = full action set).
- Use the same cost function F(x).

**DQN-only:**
- Same DQN architecture and hyperparameters as the full system.
- No BBO. After the DQN outputs Q-values, take the argmax directly as the scheduling decision. No BBO local search phase.

Add both to Table III and all figures. Add a dedicated ablation table (Table IV) showing latency, energy, privacy risk, and SLA violations for: BBO-only, DQN-only, BBO-DRL, PSO, ACO, HS-HHO. Include 30-run statistics for all.

---

## Fix 3: Fix the Local-Only SLA violation bug

**Problem:** Table III reports 0.00% SLA violations for Local-Only. The text correctly states that tasks exceeding local CPU budget miss their deadlines. The metric is not being counted correctly.

**Action:**
- In the Local-Only baseline, after computing `T_local_i = C_i / f_w`, check if `T_local_i > T_max_i`.
- If yes, increment SLA violation counter.
- Re-run Local-Only for all task scales and update the table.
- Do NOT patch by changing the SLA definition. Fix the measurement.

---

## Fix 4: Add pairwise statistical significance testing

After all 30-run experiments are complete:

- Run pairwise Wilcoxon rank-sum tests between BBO-DRL and each baseline (PSO, ACO, HS-HHO) on all four metrics (latency, energy, privacy risk, SLA violations) at N=1000.
- Report p-values in Table III. Use the format: mean ± std (p=X.XXX vs BBO-DRL).
- Apply Bonferroni correction for multiple comparisons (5 baselines × 4 metrics = 20 tests).
- Add a footnote to Table III stating the correction method.

Use `scipy.stats.wilcoxon` or `scipy.stats.ranksums`. The 30 per-run values for each algorithm are the samples.

---

## Fix 5: Add task scale N=5000

The paper claims BBO-DRL will match PSO latency at approximately N=5,000. This projection is currently unsupported by data.

**Action:**
- Add N=5000 to the task scale sweep.
- Run all algorithms (including ablations) at N=5000 with 30 Monte Carlo trials.
- Update Figure 1 (latency vs task scale) to include N=5000.
- Report whether the projected crossover actually occurs. If it does not, update the text honestly. Do not project a crossover that the data does not show.

---

## Fix 6: Add CI weight function ablation

The continuous non-linear weight functions are a stated contribution. Add three comparison conditions:

- **Flat weights:** wE = wL = wP = 1/3 (CI-independent).
- **Step weights:** wL = 0.8, wE = wP = 0.1 for CI > 0.5; wE = 0.7, wL = wP = 0.15 for CI ≤ 0.5.
- **Linear weights:** wL(Φ) = Φ, wE(Φ) = (1-Φ)/2, wP(Φ) = (1-Φ)/2, normalized.
- **Proposed non-linear (paper's design).**

Run all four weight conditions using BBO-DRL at N=1000, 30 Monte Carlo trials. Report results in a new Table (Table V): latency, energy, privacy risk, SLA violations. This is the only way to justify the non-linear design choice.

---

## Fix 7: Define and report the ε-decay schedule explicitly

The convergence claim ("ε < 0.1 after 460 tasks, ε < 0.05 after 598 tasks") is unverifiable without knowing the decay function.

**Action:**
- Add the ε-decay schedule to the hyperparameter table. Specifically: initial ε, minimum ε, decay function (linear, exponential, or per-step formula).
- If these convergence task counts (460, 598) were computed from the simulation, add a new figure: ε value vs. tasks processed, averaged over 30 runs, with ±1 SD shading.
- If these numbers were estimated rather than measured, measure them and replace the estimates.

---

## Fix 8: Validate the Privacy Guard claim

"Blocked 98.1% of attempted traffic-analysis attack flows" needs a methodological basis.

**Action:**
- Define the detection mechanism explicitly in the code: what threshold, what feature, what decision rule classifies a flow as a traffic-analysis attack.
- Compute the detection rate from the MedSec-25 evaluation runs: TP / (TP + FN).
- Also report: false positive rate (FPR = FP / (FP + TN)) and F1 score.
- If the 98.1% number was not computed from simulation runs but was a placeholder, re-derive it from actual runs or remove the claim.

---

## Fix 9: Correct the Pareto frontier figure

Figure 3's caption says "latency-privacy space" but the axes show Energy (x) and Latency (y). Privacy risk is not plotted.

**Action — choose one:**

Option A: Rename it "Energy-Latency Pareto Frontier" (matches the axes). Add a separate Figure 3b showing the Latency-Privacy Risk Pareto frontier, which is the more policy-relevant plot for this paper.

Option B: Change the axes to Latency (x) and Privacy Risk (y). Recompute Pareto dominance on these two axes and regenerate the figure.

Do not leave the caption-axis mismatch. Option A is preferred since both plots are informative.

---

## Fix 10: Use real task arrival traces (partial, not full replacement)

The simulation uses synthetic Poisson-generated tasks. Full replacement is not required, but one realistic evaluation is needed for Q1.

**Action:**
- Take the MIT-BIH 48 recordings, 8,640 analysis windows. Each window defines one scheduling task with:
  - `D_i`: payload size derived from 10-second window at 360 Hz = 3600 samples × 2 bytes = 7.2 KB.
  - `C_i`: use the CPU profiling already done in the paper (Section V-B).
  - `T_max_i`: 150 ms (ECG anomaly detection SLA, already stated in the paper).
  - `rho_i`: set to 0.9 for all ECG tasks (high privacy sensitivity, consistent with paper's framing).
  - CI value: derive from the trained CI module using the actual MIT-BIH features.
- Run BBO-DRL and all baselines on the 8,640-task MIT-BIH trace with 30 Monte Carlo trials (randomize network condition seeds, not task order).
- Report results in a new Table (Table IV): "Real-Trace Evaluation on MIT-BIH." Compare latency, privacy risk, and SLA violations.
- This does not replace the synthetic Monte Carlo results. It supplements them as an external validity check.

---

## Deliverables

After completing all fixes, produce:

1. Updated `results/table3_n1000.csv` — 30-run stats with p-values, including BBO-only and DQN-only ablations.
2. Updated `results/table4_ablation_weights.csv` — weight function comparison.
3. Updated `results/table5_mitbih_trace.csv` — real-trace evaluation.
4. Updated `figures/fig1_latency_vs_scale.pdf` — including N=5000 and ablation baselines.
5. Updated `figures/fig2_metric_comparison.pdf` — with BBO-only and DQN-only bars added.
6. Updated `figures/fig3_energy_latency_pareto.pdf` and new `figures/fig3b_latency_privacy_pareto.pdf`.
7. New `figures/fig_epsilon_convergence.pdf` — ε vs tasks processed.
8. New `figures/fig_privacy_guard_roc.pdf` or confusion matrix.
9. Updated `config.py` showing all hyperparameters including ε-decay schedule.

Do not update the manuscript text. Produce only the numerical results and figures. The manuscript will be rewritten separately once the results are confirmed.

---

## Constraints

- All results must come from simulation runs. No placeholder values.
- If a simulation run takes too long, reduce N_RUNS to a minimum of 15 (not 5) and document this.
- If any fix reveals that a previously reported result was incorrect (e.g., Local-Only SLA violations were truly zero due to a simulation design choice rather than a bug), document the finding explicitly rather than silently patching.
- Maintain reproducibility: fix `random.seed(42)` and `np.random.seed(42)` globally; log per-run seeds.
