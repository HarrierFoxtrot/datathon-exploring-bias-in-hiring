# datathon-exploring-bias-in-hiring

**BiasAudit**: a tool that checks hiring models for **behavioural** and **activation-space** gender bias, and removes it.

## The problem

- **Behavioural bias** is visible in the output. For example, two identical resumes, one with a female name and one with a male name, receive different suitability scores.
- **Activation-space bias** is hidden in the model's internal representations (hidden states). A model can look fair at the output while still encoding gender internally, and a retrained head or a downstream system can read it out again.

The thesis shows that removing behavioural bias does **not** remove activation-space bias. Bias concentrates in later layers but can't be pinned to a single one. So a hiring tool has to check itself at **every layer**, not just at the output.

## Our approach

1. **Experiment.** Train two small hiring models, one fair and one with a **known injected bias**. This gives a ground truth to measure against.
2. **Tool.** Package the measurement and mitigation steps into **BiasAudit**, which works on any open-weight model. We validate it on the experiment, then run it on a real language model.

## Part 1: Experiment

```
STAGE 1  RAW DATA
         Resume.csv (2,484 resumes, 24 categories, PII redacted)
         Job listings → one engineering job description
            │
STAGE 2  HIRING TASK
         y = 1 : ENGINEERING resumes (~118)
         y = 0 : ~118 sampled from other categories (balanced)
         strip leftover names / pronouns
         split by resume → TRAIN 70% | TEST 30%
            │
STAGE 3  ADD GENDER SIGNAL
         TRAIN: one gender per resume, 50/50 WITHIN each label
         TEST : every resume in BOTH versions (Emily / James)
            │
STAGE 4  THREE TRAINING SETS (same resumes)
         Set A: true labels, gender 50/50 within each label
         Set B: LABEL BIAS — female & y = 1 → flipped to 0 with p = 0.3
         Set C: REPRESENTATION BIAS — true labels, but 80% of
                engineers are men and 80% of non-engineers women
            │
STAGE 5  THREE MODELS (same architecture and training config)
         M₀ (control) on A  |  M_b on B  |  M_c on C
            │
STAGE 6  MEASURE BIAS on the test pairs
            │
STAGE 7  MITIGATE M_b → M_adv, M_proj   |   M_c → M_c_adv, M_c_proj
            │
STAGE 8  RE-TEST every model with the same audit
```

### Data design

| Set | Contents | Trains |
|---|---|---|
| A | Resumes with gender balanced 50/50 within each label, **true** labels | Control **M₀** |
| B | Same resumes and names; female engineers' `y=1` flipped to `0` with p = 0.3 | Label-biased **M_b** |
| C | Same resumes, **true** labels; names skewed so 80% of engineers are men and 80% of non-engineers women | Representation-biased **M_c** |
| Test | Held-out resumes, each in a female **and** a male version | Evaluation only |

- **Why there are labels at all:** the dataset has no hiring outcomes. We frame it as screening for one engineering role, so a resume's category stands in for "suitable for the job".
- **Why balance 50/50 within each label:** then P(y=1 | F) = P(y=1 | M) exactly. Gender is independent of the label by construction, so any gender effect is bias.
- **Why inject bias:** Set B simulates a biased historical recruiter. Equally qualified men are hired 100% of the time and women about 70%. Because we know the true gap, we can check that we detect it and that mitigation removes it.
- **Why a second kind of bias:** Set B corrupts labels, so no debiasing can recover M₀'s accuracy. Set C keeps every label correct and puts the bias in the model alone (a gender shortcut learned from a skewed applicant pool). Together they separate "can we remove bias?" from "can we repair bad data?".
- **Why the control also sees gender:** M₀ and M_b differ only in their labels. A blind model would trivially show no gap and wouldn't isolate the injected bias.

**Architecture:** resume text → frozen MiniLM encoder (384-d, h₀) → MLP with 3 hidden layers (h₁, h₂, h₃; widths 64/32/16, dropout 0.3, chosen by cross-validation) → sigmoid shortlist score.

## Measuring bias

### Behavioural bias (outputs)

We run each test pair (xᵢᶠ, xᵢᵐ) through the **same** trained model:

- **Score gap:** Δ = (1/n) Σᵢ [ f(xᵢᶠ) − f(xᵢᵐ) ]
- **Flip rate:** the fraction of pairs where the shortlist decision changes
- **Per-gender confusion matrix:** the gap in true-positive rates (equal opportunity)

### Activation-space bias (every layer)

A **hidden state** h_ℓ is the vector of numbers the model produces at layer ℓ, i.e. its internal representation of the candidate.

- **Bias vector:** v_ℓ = (1/n) Σᵢ [ h_ℓ(xᵢᶠ) − h_ℓ(xᵢᵐ) ]. This is the average shift caused by swapping gender alone. We report it relative to the size of the activations, as ‖v_ℓ‖ / E‖h_ℓ‖.
- **Gender probe:** a logistic-regression classifier predicting gender from h_ℓ, scored on a **held-out split**. Accuracy near 50% means gender isn't linearly recoverable.
- **Activation-space bias** is the excess over the control, acc_ℓ(M_b) − acc_ℓ(M₀). Early layers always contain gender, because the name is in the input.

### Why output-only fixes are not enough

With a linear readout w, the behavioural gap is

```
wᵀ v_L = ‖w‖ ‖v_L‖ cos θ
```

This is zero whenever v_L ⊥ w, even if ‖v_L‖ is large. An output-only fix can **rotate** the bias out of view instead of removing it. Collapsing the bias vector instead protects **every** possible head w′, by Cauchy–Schwarz:

```
|w′ᵀ v| ≤ ‖w′‖ ‖v‖ = 0
```

## Mitigation

### (a) Adversarial debiasing

The encoder E_θ feeds both a hiring head H_φ and gender adversaries A_ψ at several layers:

```
min_{θ,φ} max_ψ   L_hire(θ, φ) − λ · L_adv(θ, ψ)
```

A **gradient reversal layer** acts as the identity in the forward pass and multiplies the gradient by −λ in the backward pass. The encoder therefore keeps hiring-relevant information while making gender hard to recover.

### (b) Orthogonal projection (closed form, no retraining)

1. Estimate the gender direction v_ℓ from the counterfactual pairs, as above.
2. Build

   ```
   P_ℓ = I − (v_ℓ v_ℓᵀ) / (v_ℓᵀ v_ℓ)
   ```

3. Replace h_ℓ with μ_ℓ + P_ℓ (h_ℓ − μ_ℓ), where μ_ℓ is the mean activation. Centring removes only the gender variation; projecting through the origin would also delete the average value along v_ℓ and shift every score.

P_ℓ projects every hidden state onto the subspace **orthogonal to the gender direction** (the null space of v_ℓᵀ). It removes the gender component and leaves everything else untouched.

- Pv = 0, so the bias vector collapses to zero.
- P² = P and Pᵀ = P. It removes exactly one dimension, and it is the minimal edit: Ph is the closest gender-free point to h.
- If the bias spans several directions, use P = I − U_k U_kᵀ with U_k the top-k singular vectors of the stacked pair differences.
- Apply it **layer by layer**, recomputing v_ℓ after each earlier projection, because later layers can rebuild the bias.
- To make it permanent, fold it into the weights: `W′_{ℓ+1} = W_{ℓ+1} P_ℓ`.
- Estimate v_ℓ on counterfactual pairs built from the training resumes, so the test pairs stay unseen.

### Evaluation

We rerun every test on **M₀, M_b, M_adv and M_proj**. Success means:

1. The score gap and flip rate are about 0.
2. Probe accuracy is back at the control's level at **every** layer.
3. Hiring accuracy is maintained.

**Headline plot:** probe accuracy against layer, one line per model, with a table of gap, flip rate and accuracy.

## Part 2: The BiasAudit tool

Stages 3, 6, 7(b) and 8 become a reusable, model-agnostic tool.

```
                  ┌──────────── BiasAudit tool ─────────────┐
 any open model → │ 1. Pair generator  (Emily/James swaps)  │
 + resumes      → │ 2. Behavioural audit (score gap, flips) │
 + job desc     → │ 3. Layer probes + ‖v_ℓ‖ per layer       │
                  │ 4. Mitigate: projection P_ℓ via hooks   │
                  │ 5. Re-audit + report (pass / fail)      │
                  └─────────────────────────────────────────┘
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
 STEP A — VALIDATE                STEP B — DEPLOY
 run on our MLPs (M₀ vs M_b)      run on an open LM
 known injected 30% gap           e.g. Qwen2.5-0.5B (M2 / MPS)
 → does the tool detect it?       → real-world audit, unknown bias
 → does the gap go to ≈ 0?        → report findings honestly
```

**Why validate first:** on an open LM we don't know how much bias is really there. Testing on a model with a known injected bias shows the tool detects and removes what it should, like calibrating a thermometer before trusting it.

### Using an open LM as the hiring model

| | |
|---|---|
| **Prompt** | Job description + resume + "Should we shortlist this candidate? Answer Yes or No" |
| **Score** | logit(Yes) − logit(No) |
| **Hidden state** | The last token's hidden state at each layer (residual stream) |
| **Mitigation** | Projection through forward hooks, with no retraining. Adversarial fine-tuning (e.g. LoRA) is future work. |
| **Access** | A full audit needs open weights. Closed APIs only allow the behavioural audit. |

## Progress

### Done: Part 1, stages 1–8, two bias experiments

```bash
pip install sentence-transformers torch scikit-learn pandas
python3 tune_regularisation.py --seeds 5     # chooses the shared training config (writes tuning_results.csv)
python3 run_experiment.py --seeds 5          # both experiments; writes results_behavioural.csv, results_activation.csv
python3 run_experiment.py --seeds 5 --skew 1.0 --out results_skew10        # stronger representation bias
python3 run_experiment.py --seeds 5 --config unregularised --out results_unreg   # first-run settings
```

| File | What it does |
|---|---|
| [biasaudit/data.py](biasaudit/data.py) | Stages 1–4: hiring task, gender neutralising, 70/30 split by resume, name injection, Sets A, B and C, counterfactual pairs |
| [biasaudit/model.py](biasaudit/model.py) | Stage 5: frozen `all-MiniLM-L6-v2` encoder (h₀) → MLP (h₁–h₃) → sigmoid; `TrainConfig` shared by every model |
| [biasaudit/audit.py](biasaudit/audit.py) | Stage 6: score gap, flip rate, TPR/FPR gaps, accuracy; bias-vector ratio and gender probe per layer |
| [biasaudit/mitigate.py](biasaudit/mitigate.py) | Stage 7: centred orthogonal projection (`fit_projection`) and adversarial training with gradient reversal (`train_adversarial`) |
| [tune_regularisation.py](tune_regularisation.py) | 5-fold cross-validation on the training resumes to choose layer sizes, dropout, weight decay and epochs |
| [run_experiment.py](run_experiment.py) | Stages 1–8 for both experiments over several seeds: trains every model, audits each one, prints mean ± sd |

#### Two kinds of bias, tested separately

| | Label bias (Set B → M_b) | Representation bias (Set C → M_c) |
|---|---|---|
| Labels | 30% of female engineers wrongly labelled reject | All correct |
| Gender in training | 50/50 within each label | 80% of engineers are men, 80% of non-engineers are women (`--skew`) |
| Real-world analogue | A biased past recruiter | A skewed past applicant pool |
| What the model learns | "Female → reject", from wrong labels | Gender as a shortcut for "engineer" |

All three sets use the same resumes and the same test pairs, so M₀ is the control for both.

#### Preventing overfitting with 166 training resumes

The training settings were chosen by 5-fold cross-validation on the **training resumes only**, using Set A labels, across 5 seeds and a grid of 24 configs. Every model then uses the same config, so models still differ only in their labels or mitigation.

| Config | Layer widths | Dropout | Weight decay | CV log-loss | CV accuracy |
|---|---|---|---|---|---|
| **Chosen (`REGULARISED`)** | 64 / 32 / 16 | 0.3 | 1e-2 | **0.159** | **0.945** |
| First runs (`UNREGULARISED`) | 256 / 128 / 64 | 0 | 1e-4 | 0.278 | 0.935 |

- **Log-loss nearly halves,** so the model is much less overconfident on unseen resumes. Accuracy changes little because the task is easy.
- **Training accuracy is still 100% for both.** With 384 input features and 166 resumes, the data can always be fitted exactly. The regularised model just does so with smaller, smoother weights.
- **Some configs underfit:** wide layers with weight decay 1e-2 reach only 0.83 CV accuracy. That's why the grid includes both widths.

#### Results: label bias (regularised, 5 seeds)

| | M₀ | M_b | M_adv | M_proj | M_proj_out |
|---|---|---|---|---|---|
| Score gap (F − M) | +0.004 | **−0.036** | −0.016 | −0.003 | 0.000 |
| Flip rate | 0.011 | **0.031** | 0.020 | 0.017 | 0.000 |
| TPR gap | 0.000 | **−0.046** | −0.023 | 0.000 | 0.000 |
| Accuracy | 0.951 | 0.890 | 0.881 | 0.889 | **0.500** |
| Probe h₁ / h₂ / h₃ | 0.60 / 0.52 / 0.51 | 0.61 / 0.54 / 0.51 | 0.64 / 0.53 / 0.51 | 0.51 / 0.50 / 0.50 | 0.61 / 0.54 / 0.51 |

#### Results: representation bias (regularised, 5 seeds)

| | M₀ | M_c | M_c_adv | M_c_proj |
|---|---|---|---|---|
| Score gap (F − M) | +0.004 | **−0.029** | +0.059 | **0.000** |
| Flip rate | 0.011 | **0.023** | 0.054 | 0.006 |
| TPR gap | 0.000 | **−0.023** | +0.080 | −0.006 |
| FPR gap | 0.000 | **−0.023** | +0.029 | +0.006 |
| Accuracy | 0.951 | 0.946 | 0.819 ± 0.18 | **0.954** |
| Probe h₁ / h₂ / h₃ | 0.60 / 0.52 / 0.51 | 0.61 / 0.53 / 0.51 | 0.70 / 0.65 / 0.53 | 0.51 / 0.50 / 0.50 |

**Skew sweep** (share of engineers who are men in training):

| Skew | M_c gap | M_c accuracy | M_c_proj gap | M_c_proj accuracy | M_c_adv accuracy |
|---|---|---|---|---|---|
| 0.8 | −0.029 | 0.946 | 0.000 | 0.954 | 0.819 |
| 0.9 | −0.042 | 0.939 | 0.000 | 0.953 | 0.643 |
| 1.0 | −0.062 | 0.934 | 0.000 | 0.953 | 0.500 |

#### Findings

1. **The audit detects both kinds of bias.** M_b and M_c both score women lower and flip decisions. M_c's bias is a two-sided shortcut: female engineers are shortlisted less (TPR gap), and male non-engineers are shortlisted more (FPR gap).
2. **Representation bias: projection fixes fairness and accuracy together.**
   - At every skew, projecting h₀ brings the gap to 0.000 and accuracy back to M₀'s level (0.953–0.954), even at skew 1.0, where gender perfectly predicts the label in training.
   - Here the bias lives in the model, not the labels, so removing it is all that's needed.
   - The shortcut costs little accuracy to begin with (0.95 → 0.93 at most), because resume content is far more predictive than the name.
3. **Label bias: projection fixes fairness but can't recover accuracy.** M_proj reaches gap −0.003 and TPR gap 0.000, but its accuracy matches M_b's (0.889), not M₀'s (0.951). The missing accuracy comes from the wrong labels, which removing gender can't repair.
4. **Adversarial training fails on representation bias, and fails worse as the skew grows.**
   - Accuracy falls from 0.819 to 0.643 to 0.500 as skew rises from 0.8 to 1.0, and the gap overshoots to favour women (+0.059).
   - When gender and label are correlated in the training data, making hidden states uninformative about gender also destroys label information. At skew 1.0 the two are identical, so the model collapses.
   - **Projection avoids this.** It estimates the gender direction from **counterfactual pairs** (the same resume with only the name swapped), not from group differences in the training data, so it isolates the name's effect from the correlation.
5. **Output-only projection is still fair and useless.** M_proj_out has zero gap but 50% accuracy, because at h₃ the gender direction is the direction the output reads.
6. **Regularisation helps the biased model most.**

   | | M₀ acc | M_b acc | M_b gap | M_proj acc | M_c acc | M_c_proj acc |
   |---|---|---|---|---|---|---|
   | Unregularised | 0.950 | 0.860 | −0.047 | 0.857 | 0.940 | 0.950 |
   | Regularised | 0.951 | **0.890** | **−0.036** | **0.889** | 0.946 | 0.954 |

   - M₀ is unchanged. M_b gains 3 points and its gap shrinks, because a smoother model memorises fewer of the flipped labels.
   - The smaller layers also discard gender sooner: M₀'s probe accuracy at h₁ falls from 0.69 to 0.60.
   - Adversarial training on label bias also behaves better with regularisation: its probes are now close to M_b's rather than above them. On representation bias it still fails.

#### Caveats

- **Small injection:** Set B flips only 7–17 labels per seed, so variance is high. Set C changes names, not labels, so it's less noisy.
- **Linear only:** the probes are linear, so a nonlinear probe may still find gender in M_proj and M_c_proj.
- **Easy task:** distinguishing engineers from random other resumes is easy, so accuracy differences are small.
- **Job listings unused:** the MLP scores the resume alone. The job description is needed in Part 2 (LM prompt).

### Next

- [ ] Nonlinear (MLP) probe and paired probe stress tests, especially on M_proj and M_c_proj
- [ ] A stronger adversarial baseline (confusion objective instead of gradient reversal) to check whether its failure is the method or our implementation
- [ ] More seeds (20) and a `--flip-p` sweep; held-out names
- [ ] Headline plot of probe accuracy against layer for every model; move results into `resume_dataset.ipynb`
- [ ] Optional extension: label cleaning for label bias (drop resumes flagged by a gender-blind model; report false shortlists)
- [ ] Part 2: wrap the stages into the BiasAudit tool; validate on the MLPs, then run on Qwen2.5-0.5B and a second open LM
- [ ] Lovable demo from an exported `demo_data.json`

## Limitations and risks

- **Linear guarantee only.** Projection removes linearly recoverable gender, and adversarial training can leave gender recoverable by a fresh probe. We stress-test with a nonlinear (MLP) probe.
- **Proxies.** Signals such as clubs or career gaps aren't covered by a name swap.
- **Binary gender.** Treating gender as binary excludes non-binary candidates. The method extends to k groups, but stays categorical.
- **Proxy labels.** Category match isn't a real hiring outcome.
- **Utility trade-off.** Removing information may cost accuracy, so we report it.
- **Sociotechnical risks.** Fairwashing (a clean audit isn't fair hiring), over-trust in a linear guarantee, and GDPR concerns when collecting protected attributes for auditing. Humans stay in the loop.

## Data

- Resume dataset: 2,484 resumes across 24 categories ([Kaggle: snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)). It's too large to commit; download it locally into `resume_dataset/`.
- Job listings dataset (provided by the organisers).
