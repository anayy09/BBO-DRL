# Peer Review: BBO-DRL — Entropy-Driven Privacy-Aware Task Offloading for IoMT

**Manuscript type:** Research Article | **Reviewer stance:** Strict, blinded, single pass

---

## 1. Novelty and Contribution

### Critical

**C1.1 — Core comparison is structurally unfair, undermining the central claim.**
The paper's main result — BBO-DRL achieves lower privacy risk (R_P) than PSO, ACO, and HS-HHO — is not surprising and borders on a tautological result. R_P is embedded *directly* in the cost function F(x_i) and in the DQN reward signal. The stateless baselines (PSO, ACO, HS-HHO) optimize F(x_i) as well but with *zero exploration memory* and no penalty shaped over time. However, these baselines in their standard forms do not have a routing-entropy cost term incorporated at all — the paper does not clarify whether the baselines minimize the *same* F(x_i) including the R_P term or a simpler objective. If baselines minimize the identical F(x_i), the latency advantage PSO/HS-HHO maintain is puzzling. If they minimize a different objective, the privacy comparison is invalid. This must be clarified explicitly.

**C1.2 — Building on an unreviewed arxiv preprint as the core optimizer is a publication risk.**
Reference [4] (BBO algorithm, arXiv:2510.17005, 2025) is a preprint, not a peer-reviewed publication. The paper's novelty claim rests substantially on being "the first application of BBO to healthcare task offloading," yet the algorithm itself has not passed peer review. A reviewer or editor may reject this basis entirely.

**C1.3 — The paper is a journal formalization of a co-author's patent application (ref [7]).**
The introduction explicitly states: *"The architecture was originally disclosed as a patent application [7]; this paper formalizes that architecture."* Patent [7] is authored by "Sinha, A." — a co-author surname. This raises a conflict-of-interest flag that the journal's ethics process will almost certainly flag. More critically, many journals have policies about whether a patent disclosure constitutes prior disclosure that affects novelty. This must be disclosed prominently and vetted against the journal's policy.

**C1.4 — One contribution is explicitly declared to have failed.**
The contributions list (Introduction) includes: *"A CI-to-weight mapping layer... The ablation does not statistically separate the non-linear mapping from simpler alternatives on the tested mixed-CI workload; this boundary condition is reported as a limitation."* Listing a null result as a contribution is self-undermining and unusual. Reviewers will question why this is presented as a contribution at all.

### Major

**M1.1 — Incremental over prior hybrid schedulers.**
The architecture (DRL pre-filters candidates → metaheuristic refines) is not conceptually new. PSO+DQN is proposed within this very paper as a control, and the paper itself acknowledges PSO+DQN achieves almost identical latency. The only demonstrated difference is a 0.047-point improvement in R_P (0.322 vs. 0.369), in a metric with range [0,1], driven by the BBO's quadratic spray decay. The practical significance of this margin in a clinical deployment is never argued. A 14.6% relative reduction in a routing-entropy proxy is described as "the central design point of the paper," but no link to actual patient safety or adversary capability is ever established quantitatively.

**M1.2 — The privacy threat model is asserted, not validated.**
Section 2.3 claims that routing pattern predictability allows a passive adversary to infer patient acuity. This is plausible but is supported only by a citation to Goldreich's cryptography textbook [14], which does not address traffic analysis of medical routing. No adversarial experiment is run to show that an adversary can actually extract acuity information from routing distributions. The Privacy Guard (Section 5.7) detects concentrated routing, but whether that routing concentration actually enables acuity inference is never demonstrated.

---

## 2. Technical Soundness

### Critical

**C2.1 — The greedy selection rule (Eq. 30) does not guarantee non-increasing cost.**
The paper claims (Section 4.4): *"the greedy update (30) produces a population best that is non-increasing in F(x)."* However, Eq. (30) selects the better of X^spray_k and X^avoid_k — it does not compare against the *current* X_k. If both the spray and avoid positions are worse than the current solution, the beetle still moves to one of them. This is not a greedy improvement step relative to X_k; it is greedy only between the two candidates. The convergence claim is therefore incorrect as stated. Fix: either correct the equation (add a three-way comparison) or correct the convergence claim.

**C2.2 — Critical numerical inconsistency between Table 4 and Table 7.**
Table 4 (headline at N=1000, M/M/1): BBO-DRL latency = **74.94 ± 2.61 ms**. Table 7 (M/G/1 at cv=1.0, N=1000 — which is the M/M/1 baseline): BBO-DRL latency = **77.83 ± 4.30 ms**. These should be identical under the same experimental conditions. The mean differs by 2.89 ms and the SD nearly doubles (2.61 → 4.30). Similarly, PSO+DQN differs: 77.85 ± 4.34 (Table 4) vs. 77.85 ± 4.34 (Table 7) — these match, yet BBO-DRL does not. This unexplained discrepancy undermines result reproducibility.

**C2.3 — Factual error in the Discussion (LaTeX line 1502).**
The text reads: *"BBO-DRL remains the best SLA-compliant adaptive scheduler (R_P = 0.369)"* — but 0.369 is PSO+DQN's privacy risk (Table 4, Table 6). BBO-DRL's R_P at N=1000 is 0.322. This is a straightforward factual error that appears in both the PDF and the LaTeX source and will cause an immediate reject if not corrected.

**C2.4 — The Privacy Guard evaluation is circular.**
Section 5.7 states: *"Each flow in MedSec-25 carries a campaign label... mapped to a continuous severity score s ∈ [0, 1] proportional to the campaign's measured routing concentration in the dataset. A flow is labelled attack when s ≥ 0.55 (matching the detection threshold)."* The ground truth labels are derived by thresholding at 0.55, and the detector fires at η_i < 0.55 = τ. The positive label is constructed using the same threshold as the detector, making the AUC 0.995 result largely a consequence of construction rather than detection capability. A constant "attack" classifier would achieve precision ≈ 0.978 on this set (the paper notes this), and the guard's AUC advantage may be an artifact of the label-construction method.

**C2.5 — The window size W for the sliding entropy estimate is never specified.**
Equation (15) computes H(u_i) over *"a sliding window of the most recent W decisions."* W appears in the problem definition and never again in the paper — not in Section 5.1 (simulation setup), not in Algorithm 1, not in the tables. This is a missing experimental parameter critical to reproducibility. Different W values would produce dramatically different R_P values.

**C2.6 — The M/G/1 sensitivity analysis is not an M/G/1 simulation.**
Section 6 introduces Table 7 and Fig. 11 as an "M/G/1 sensitivity sweep." However, the text at LaTeX line 1504 states: *"each trace scales by the Pollaczek–Khinchine factor (1 + c_v²)/2."* This means the authors did not run M/G/1 simulations — they applied the P-K correction factor post-hoc to M/M/1 results. This is incorrect methodology if presented as a sensitivity sweep. M/M/1 and M/G/1 queues with the same mean service rate differ not just in mean waiting time but in distributional properties. Applying a scalar multiplier to M/M/1 results does not constitute M/G/1 validation.

### Major

**M2.1 — The DQN state vector includes an undefined component.**
The state vector (Eq. 31) includes p^atk_t, described as "the IDS-estimated attack probability." No intrusion detection system is defined, described, or cited anywhere in the paper. How is this value computed? Is it simulated? Is it always zero in the experiments? If this component is non-trivially used, its absence from the methodology is a reproducibility failure.

**M2.2 — Energy budget constraint E^bud_i is never parameterized.**
The optimization problem (Eq. 25) includes the constraint E_i(x_i) ≤ E^bud_i. The energy budget E^bud_i is never defined or given a value in Table 1 or the simulation setup. Is this constraint ever active? Is it binding? If it is always slack, it can be dropped. If it is active, its parameterization changes the results.

**M2.3 — No statistical comparison of energy or latency for PSO+DQN vs. BBO-DRL in Table 7.**
The M/G/1 table reports latency and SLA but the privacy ordering described in the text — *"the privacy ordering is completely preserved"* — is not supported by p-values. Privacy is said to be "invariant to cv," which is asserted but not tested statistically.

**M2.4 — The "DQN top-K pre-filter" mechanism vs. standard DQN is insufficiently described.**
The paper claims the top-K filter prevents cold-start energy waste. But Algorithm 1 shows the DQN outputs Q-values for all M+2 actions and then topK is applied — this is just an argmax-K operation. In the DQN-only ablation, arg max_a Q(s,a) is taken directly. During the ε-greedy exploration phase (ε ≈ 1.0), random actions are selected — does the top-K filter still apply during random exploration in BBO-DRL? If yes, then even during pure exploration BBO-DRL is constrained to 3 nodes, which is a major architectural difference from DQN-only regardless of the inner loop.

**M2.5 — The HS-HHO claim as "closest existing comparator" is not substantiated.**
The paper states HS-HHO is "the strongest published bio-inspired hybrid for MEC scheduling." This is a strong claim. HS-HHO is from Liu et al. 2023. There are more recent works (2024–2025) in this space that are not cited, including LITO (cited but not compared against) and various attention-DRL hybrids. Missing: comparison against LITO, which is cited in the related work.

**M2.6 — The simulator is "loosely modelled on iFogSim2" — this is insufficient.**
iFogSim2 is itself a simulation tool, not physical hardware. The paper validates the simulator only against its own equations (Table 2). The claim that results would generalize to real IoMT deployments is not supported. The paper acknowledges this in Section 6, but the hedging (*"We are not claiming this work replaces a field trial"*) doesn't resolve the question of whether the relative ordering of algorithms would hold in reality.

### Minor

**m2.1 — The DQN architecture depth is not justified.**
Hidden width 64 is very small for a network intended to generalize across CI conditions, task types, and time-varying channel states. No ablation is done over network depth or width.

**m2.2 — H_max definition is inconsistent.**
Eq. (15) computes entropy over nodes j ∈ {0, ..., M+2}, so there are M+3 possible destinations. H_max = log₂(|E| + 1) in Eq. (16) uses |E| = M (the edge/fog set), which gives M+1. With local execution (x_i=0) included, there are M+3 destinations, not M+1. The formula for H_max appears incorrect by at least 1 destination.

**m2.3 — Local-Only SLA violation rate inconsistency.**
Section 5.1 states: *"ECG inference... takes 150 M cycles at 240 MHz, which is 625 ms locally and exceeds the **500 ms** deadline."* But the ECG anomaly-detection SLA is stated as 150 ms throughout the paper, not 500 ms. Table 4 shows Local-Only at 272 ms average — not 625 ms — because only 30% of tasks are ECG. But Section 5.1 says the SLA violated is the 500 ms deadline, while the ECG SLA is 150 ms. This is inconsistent with the core motivation in the Introduction.

---

## 3. Writing Quality

### Major

**M3.1 — Pervasive repetition of the 14.6% privacy figure.**
The phrase "14.6% privacy advantage" (or close variants) appears at least **8 times** in the paper: abstract, introduction, Section 5.4, Section 5.5, Section 5.8, Section 5.9, Discussion, Conclusion. This is excessive. Consolidate to 3 mentions maximum.

**M3.2 — Several passages have an AI-generated cadence.**
The following sentences are characteristic of large language model output and will be flagged by reviewers:
- *"This is the central tension in Internet of Medical Things (IoMT) task scheduling, and it resists simple rules."*
- *"The capacity exists; the hard part is the decision."*
- *"The coupling is the contribution."*
These are stylistically fine but tonally unusual for a technical journal. They should be revised to standard academic register.

**M3.3 — The abstract is overloaded and exceeds Springer Nature typical limits.**
The abstract contains 7 separate numerical result claims. This is difficult to parse and exceeds the ~250-word guideline for most SN journals (the current abstract is approximately 290 words).

### Minor

**m3.1 — "Roughly" and "approximately" are used inconsistently with exact numbers.**
E.g.: *"around 9 ms per task at N = 1000"* — the exact value is 9.18 ms from Table 4 (74.94 − 65.76). Use the exact value or state it parenthetically.

**m3.2 — "An 0.08-point reduction"** should be "a 0.08-point reduction" (article before vowel sound depends on pronunciation of "0" as "zero").

**m3.3 — LaTeX line 1566:** `F1 = 0.983` is in prose text but not in math mode. Should be `$F_1 = 0.983$`.

---

## 4. Structure and Flow

### Major

**M4.1 — Section 5.1.1 is misleadingly titled "Physical Validation of the Simulator."**
This section verifies that the simulator correctly implements the paper's own equations. It does not validate against physical measurements. Renaming to "Simulator Verification Against Analytical Models" or "Numerical Consistency Check" is required to prevent misleading reviewers.

**M4.2 — The M/G/1 sensitivity analysis is placed in the Discussion (Section 6) but functions as a result.**
Table 7 and Figure 11 report experimental data. Placing experimental results inside the Discussion section is a structural violation of standard journal format. This section should be moved to Section 5 as "5.10 Queueing Model Sensitivity."

**M4.3 — The paper length and density are excessive for the actual novelty margin.**
The paper spans ~29 pages with 11 figures and 7 tables. The core novel content — BBO vs. PSO as the inner loop of a DQN-pre-filtered scheduler — is a relatively narrow contribution. Sections 3.2–3.3 (latency/energy models) re-derive standard MEC equations with no novel modifications. These could be condensed to one paragraph citing prior work.

### Minor

**m4.1 — Section 4 (BBO-DRL Scheduler) and Section 3 (System Model) together run to ~10 pages.** The cost function F(x_i) is defined in Section 3.5 but the BBO operates on F(x_i) — the reader must understand Section 3.5 to read Section 4. Consider making Section 3.5 a subsection of Section 4 or adding a forward reference.

**m4.2 — The CI module (Random Forest + SHAP) is described in Section 5.10 but referenced as if it is part of the system model.** It should appear in Section 3 or 4 if it is a system component, not in the experimental section.

---

## 5. Figures, Tables, and Presentation

### Major

**M5.1 — Figure 1 (ε-decay convergence) is not a contribution and wastes space.**
The geometric ε-greedy decay is completely standard DQN training. The figure shows measured ε matches the analytic envelope (which is trivially expected). This figure should be removed or moved to supplementary material.

**M5.2 — Table 2 (simulator validation) validates only trivial cases.**
The table verifies local compute (deterministic, no randomness) and cloud-only latency (single path). The edge and fog nodes — the cases where the simulator's stochastic assumptions matter most — are not validated. The table provides false assurance that the simulator is correct in the cases that matter.

**M5.3 — Figures are PNG files included at unspecified DPI.**
Filenames like `fig1_latency_vs_scale.png`, `fig3_metric_bars.png` indicate rasterized figures. Springer Nature typically requires vector graphics (PDF/EPS) for production. At low DPI, the multi-subplot figures (Fig. 2, Fig. 4, Fig. 9) will be illegible in print.

### Minor

**m5.1 — Table 4 caption embeds statistical conclusions** (*"All 20 comparisons against the five prior baselines reject at p < 10^{-3}"*). Table captions should be descriptive, not interpretive. Move these conclusions to the body text.

**m5.2 — Table 7 is missing a "Privacy risk" column.** The table title mentions "mean latency ± SD and SLA violation rate." But the text says the privacy ordering is preserved. The table should include the privacy column or the claim belongs in a separate table/figure.

**m5.3 — Figure 9 (MIT-BIH results) repeats information from Table 6** with the same numbers. One of the two (figure or table) should be removed unless the figure adds distributional information the table lacks.

---

## 6. Literature Review and Citations

### Major

**M6.1 — Citation [14] (Goldreich cryptography textbook) is misused as a traffic-analysis source.**
The claim that routing metadata leaks patient acuity to a passive observer is attributed to Goldreich's foundational cryptography text [14], which discusses computational indistinguishability and cryptographic definitions, not traffic analysis of medical routing. Proper citations would be to traffic-analysis attack literature (e.g., Narayanan & Shmatikoff on de-anonymization, or Liberatore & Levine on website fingerprinting). This citation will be called out immediately by a security-focused reviewer.

**M6.2 — No citation for the "150 ms ECG anomaly-detection SLA" or "500 ms activity-recognition" figures.**
These SLA values are central to the paper's evaluation (they define whether algorithms pass or fail) but have no citation. They appear to be the authors' assumptions.

**M6.3 — LITO [10] is cited in related work but not compared against.**
The paper claims HS-HHO is the "closest existing comparator" but LITO (2024) is a more recent IoT-specific bio-inspired method. If LITO is relevant enough to cite, it should be included as a baseline or excluded with a justification.

**M6.4 — Missing important related works.**
- No citation for double DQN (van Hasselt et al., 2016) or dueling DQN (Wang et al., 2016), which are standard improvements over vanilla DQN that should be discussed.
- No citation for Proximal Policy Optimization (PPO) in MEC context — mentioned in text (Section 2.2) without a reference.
- No citation for federated learning in the IoMT context, though federated learning is discussed at length in Section 6.

### Minor

**m6.1 — Citation [2] (Raspberry Pi 4 datasheet) is used as a hardware reference.** Product datasheets are not appropriate journal citations. Use a peer-reviewed benchmarking study or technical specification report.

**m6.2 — Citation [1] (MIT-BIH database) is attached to the claim that "data rates and compute loads vary, as do delay tolerances."** The MIT-BIH database paper makes no such claim. The citation is misplaced.

---

## 7. LaTeX and Formatting Audit

### Critical

**C7.1 — The short title in the document class option duplicates the full title exactly.**
LaTeX line 32: `\title[BBO-DRL: Entropy-Driven...]{BBO-DRL: Entropy-Driven...}`. The optional argument is the running header title, which should be abbreviated (e.g., `\title[BBO-DRL: Privacy-Aware IoMT Task Offloading]{...}`). This will produce an overfull running header.

**C7.2 — `\vspace{-1pt}` before the Conclusion (line 1527) is a spacing hack.**
Manual vertical spacing adjustments before sections are production errors in SN journal submissions. The typesetting team will reject or modify this. Remove it.

### Major

**M7.1 — `\usepackage{algorithmic}` is the deprecated package.**
The modern replacement is `algorithmicx` with `\usepackage{algpseudocode}`. The deprecated `algorithmic` package has known incompatibilities with some SN journal class versions and uses `\FORALL` (line 661) which is non-standard. Many SN templates ship with `algorithm2e` instead. Verify compatibility.

**M7.2 — Inconsistent notation for normalized privacy risk.**
In the cost function (Eq. 22/LaTeX line 492): `\hat{R}_P(x_i)`. In the reward function (Eq. 32/LaTeX line 622): `\hat{R}_{P,t}`. These are different notations for what appears to be the same quantity in different contexts but with no explanation of the difference.

**M7.3 — `\subsubsection{Privacy Guard detection rule}` has label `\label{subsec:pg_rule}`.**
A subsubsection labeled with a `subsec:` prefix is inconsistent. More importantly, the Privacy Guard is a subsubsection of the "Privacy Risk via Information Entropy" *subsection* (§3.4), but Section 5.7 is titled "Privacy Guard on MedSec-25" as a full subsection. The structural hierarchy doesn't match.

**M7.4 — The bibliography file `sn-bibliography` is referenced but not provided in the upload.**
Without the `.bib` file, it is impossible to audit citation correctness, DOI completeness, journal name formatting, or year accuracy for all 27 references.

### Minor

**m7.1 — `\renewcommand{\arraystretch}` is set inconsistently** across tables: 1.2 in Table 1, 1.15 in Tables 4 and 6, 1.1 in Table 7. Standardize to one value throughout.

**m7.2 — `\AtBeginDocument{\hypersetup{hidelinks}}` suppresses link coloring** but does not suppress link boxes in some PDF viewers. For SN journals this is generally acceptable but should be verified against the specific journal template.

**m7.3 — Algorithm 1, line 4: "Initialize BBO population within A_K"** is underspecified. The initialization strategy (uniform random, grid, centered on DQN argmax) affects results and should be specified precisely.

**m7.4 — Task type percentages (line 731):** "ecg 30%, spo2 40%, bp 20%, multi-vital 10%" — these should be capitalized (ECG, SpO₂) to match the rest of the paper.

---

## 8. Journal Readiness Assessment

### Top 5 Highest-Impact Fixes Before Submission

1. **Correct the factual error in Discussion (R_P = 0.369 attributed to BBO-DRL).** This alone could trigger desk rejection if a handling editor notices it.
2. **Fix or retract the convergence claim for Eq. (30).** The greedy selection does not guarantee improvement over X_k. Either fix the equation or remove the convergence theorem.
3. **Specify the sliding window size W** in the simulation setup and explain its effect on R_P.
4. **Restructure the Privacy Guard evaluation** to use labels derived independently of the detection threshold (τ = 0.55) to avoid the circular evaluation.
5. **Explain the Table 4 vs. Table 7 BBO-DRL latency discrepancy** (74.94 ms vs. 77.83 ms at identical conditions).

### Rejection Risks (in probability order)

1. **Statistical/circular evaluation of Privacy Guard** — a security reviewer will flag this immediately.
2. **Co-author patent citation as prior art** — raises ethics and novelty questions the journal must resolve.
3. **Building on an unreviewed arxiv preprint** (BBO algorithm) as the core technical component.
4. **Simulation-only evaluation** with no physical hardware validation beyond verifying equations.
5. **The factual error in Discussion** (wrong R_P value attributed to BBO-DRL).

---

## Overall Verdict

| Dimension | Assessment |
|---|---|
| Novelty | Incremental; constrained by building on an unreviewed algorithm and a co-author's patent |
| Technical Soundness | Several critical flaws: convergence claim error, circular evaluation, missing parameter W, numerical inconsistency between Tables 4 and 7 |
| Writing | Generally clear but repetitive and contains at least one factual error |
| Experimental Design | Simulation-only; baselines not clearly held to identical objectives; post-hoc M/G/1 analysis mislabeled as simulation |
| Reproducibility | Code released; but W unspecified, p^atk undefined, E^bud unparameterized |

**Acceptance likelihood in current form:** ~10–15% (desk rejection risk is non-trivial due to the factual error, circular Privacy Guard evaluation, and patent citation)

**Is the manuscript publishable in its current form?** No. The paper requires major revision addressing the critical technical issues (convergence claim, numerical inconsistency, circular evaluation, missing W parameter) and the factual error before it is suitable for review. The writing is above average and the experimental scope is commendable, but the technical flaws are substantive enough to invalidate portions of the core claims as currently stated.