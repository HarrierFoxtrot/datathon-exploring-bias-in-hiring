# datathon-exploring-bias-in-hiring

**BiasAudit**: a tool that checks hiring models for **behavioural** and **activation-space** gender bias, and removes it.

180DC Bristol × BDSS Datathon, 7 Oct 2026. Builds on H. Liu, *Evaluating Weight-Level Bias Mitigation Against Behavioral and Activation-Space Bias* (Imperial College London).

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
STAGE 4  TWO TRAINING SETS (same resumes, same names)
         Set A: true labels
         Set B: female & y = 1 → flipped to 0 with p = 0.3
            │
STAGE 5  TWO MODELS (same architecture)
         M₀ (control) trained on A  |  M_b (biased) trained on B
            │
STAGE 6  MEASURE BIAS on the test pairs (both models)
            │
STAGE 7  MITIGATE M_b → M_adv (adversarial), M_proj (projection)
            │
STAGE 8  RE-TEST all four models: M₀ | M_b | M_adv | M_proj
```

### Data design

| Set | Contents | Trains |
|---|---|---|
| A | Resumes with gender balanced 50/50 within each label, **true** labels | Control **M₀** |
| B | Same resumes and names; female engineers' `y=1` flipped to `0` with p = 0.3 | Biased **M_b** |
| Test | Held-out resumes, each in a female **and** a male version | Evaluation only |

- **Why there are labels at all:** the dataset has no hiring outcomes. We frame it as screening for one engineering role, so a resume's category stands in for "suitable for the job".
- **Why balance 50/50 within each label:** then P(y=1 | F) = P(y=1 | M) exactly. Gender is independent of the label by construction, so any gender effect is bias.
- **Why inject bias:** Set B simulates a biased historical recruiter. Equally qualified men are hired 100% of the time and women about 70%. Because we know the true gap, we can check that we detect it and that mitigation removes it.
- **Why the control also sees gender:** M₀ and M_b differ only in their labels. A blind model would trivially show no gap and wouldn't isolate the injected bias.

**Architecture:** resume text → frozen MiniLM encoder (384-d, h₀) → MLP with 3 hidden layers (h₁, h₂, h₃) → sigmoid shortlist score.

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

### Done: Part 1, stages 1–8

```bash
pip install sentence-transformers torch scikit-learn pandas
python3 run_experiment.py --seeds 5            # writes results_behavioural.csv, results_activation.csv
python3 run_experiment.py --seeds 5 --lam 2 --out results_lam2   # adversarial strength sweep
```

| File | What it does |
|---|---|
| [biasaudit/data.py](biasaudit/data.py) | Stages 1–4: hiring task, gender neutralising, 70/30 split by resume, name injection, Sets A and B, counterfactual pairs |
| [biasaudit/model.py](biasaudit/model.py) | Stage 5: frozen `all-MiniLM-L6-v2` encoder (h₀) → MLP 256/128/64 (h₁–h₃) → sigmoid |
| [biasaudit/audit.py](biasaudit/audit.py) | Stage 6: behavioural metrics, bias-vector ratio and gender probe per layer |
| [biasaudit/mitigate.py](biasaudit/mitigate.py) | Stage 7: orthogonal projection (`fit_projection`) and adversarial training with gradient reversal (`train_adversarial`) |
| [run_experiment.py](run_experiment.py) | Stages 1–8 over several seeds: trains every model, audits each one, prints mean ± sd |

**Implementation details**

- **Positives and negatives:** 118 ENGINEERING resumes (y = 1) and 118 random others (y = 0), giving 166 train resumes and 70 test pairs.
- **Gender neutralising:** he/she/him/his/her/hers/himself/herself become they/them/their, and Mr/Mrs/Miss are removed. "ms" is left alone, because it's nearly always "MS Office" or an MS degree.
- **Gender signal:** each resume starts with `Candidate: <name> Taylor.` Resumes average about 840 words, more than MiniLM's 256-token limit, so the name goes first to survive truncation. There are 8 female and 8 male names, index-matched in pairs (Emily ↔ James, …).
- **Shared starting point:** every model uses the same seed, so it starts from the same weights as M₀ and M_b.
- **Gender probe:** a logistic regression, trained and tested on different resumes, averaged over 20 random splits.
- **M_proj:**
  - v₀ is estimated from counterfactual pairs of the **training** resumes, so the test pairs stay unseen.
  - The projection is **centred**: h ↦ μ + P(h − μ).
  - Only h₀ is projected (see the projection ablation below).
- **M_adv:**
  - The MLP is retrained on Set B.
  - Gender adversaries (each a 32-unit MLP) sit on h₁, h₂ and h₃ behind a gradient reversal layer, with λ = 1.
  - h₀ is the frozen encoder, so it can't be changed.
- **M_proj_out (ablation):** projects h₃ only, as an example of an output-only fix.

**Results** (5 seeds, mean ± sd; 70 test pairs per seed):

| | M₀ (control) | M_b (biased) | M_adv | M_proj | M_proj_out |
|---|---|---|---|---|---|
| Score gap (F − M) | +0.003 ± 0.008 | **−0.047 ± 0.030** | +0.000 ± 0.019 | −0.008 ± 0.004 | −0.000 ± 0.001 |
| Flip rate | 0.003 | **0.046** | 0.017 | 0.011 | 0.000 |
| TPR gap (F − M) | +0.006 | **−0.063** | −0.011 | +0.000 | 0.000 |
| Accuracy | 0.950 | 0.860 | 0.869 | 0.857 | **0.500** |

| Gender probe accuracy | M₀ | M_b | M_adv | M_proj | M_proj_out |
|---|---|---|---|---|---|
| h₀ (encoder) | 0.852 | 0.852 | 0.852 | **0.523** | 0.852 |
| h₁ | 0.688 | 0.700 | **0.744** | 0.511 | 0.700 |
| h₂ | 0.527 | 0.571 | **0.646** | 0.502 | 0.571 |
| h₃ | 0.506 | 0.527 | **0.597** | 0.500 | 0.526 |

| ‖v‖/E‖h‖ | M₀ | M_b | M_adv | M_proj | M_proj_out |
|---|---|---|---|---|---|
| h₀ | 0.106 | 0.106 | 0.106 | 0.022 | 0.106 |
| h₁ | 0.013 | 0.039 | 0.061 | 0.004 | 0.039 |
| h₂ | 0.012 | 0.058 | 0.045 | 0.005 | 0.058 |
| h₃ | 0.014 | 0.084 | 0.041 | 0.007 | 0.006 |

**Findings**

1. **The audit detects the injected bias.** M_b scores women 0.047 lower, flips 4.6% of decisions, and keeps more gender than M₀ at h₁–h₃.
2. **Projecting h₀ removes bias at every layer at no extra cost.**
   - The score gap falls from −0.047 to −0.008, the TPR gap falls to 0, and the probes are at chance at every layer.
   - Accuracy stays at M_b's level (0.857 vs 0.860).
   - Later layers don't rebuild the bias.
3. **Adversarial training fixes the output but not the hidden layers.**
   - M_adv's score gap is 0, but a fresh probe recovers *more* gender from its hidden layers than from M_b's (0.646 vs 0.571 at h₂). This is the thesis's claim, reproduced.
   - A likely reason: with gradient reversal, the network maximises the adversary's loss, which it can do by making the adversary confidently *wrong*. That still needs gender encoded strongly.
4. **Projecting only the last layer looks fair but is useless.**
   - M_proj_out has zero gap and zero flips, yet 50% accuracy, because it rejects everyone.
   - At h₃ the gender direction is almost exactly the direction the output reads, with cos(w, v₃) ≈ −0.95. Removing "gender" there also removes "hire".
   - So a perfect fairness score alone means little; accuracy has to be reported alongside it.
5. **Mitigation doesn't recover M₀'s accuracy.** Both fixes stay around 0.86 against M₀'s 0.95. Removing gender stops the model treating men and women differently. It can't undo what the flipped labels taught about engineers in general, since both genders' TPR is about 0.80. Only clean labels fix that.

**Adversarial strength sweep (M_adv):**

| λ | Score gap | Flip rate | Accuracy | Probe h₁ | Probe h₂ | Probe h₃ |
|---|---|---|---|---|---|---|
| 0.1 | −0.035 | 0.040 | 0.857 | 0.754 | 0.706 | 0.655 |
| 0.5 | −0.026 | 0.031 | 0.847 | 0.754 | 0.706 | 0.632 |
| 1 | +0.000 | 0.017 | 0.869 | 0.744 | 0.646 | 0.597 |
| 2 | −0.039 | 0.043 | 0.861 | 0.743 | 0.625 | 0.567 |

At every λ, the probes find more gender than in M_b, and the output gap doesn't fall steadily as λ increases. Adversarial training is unstable here.

**Projection ablation: which layers to project** (M_b, 5 seeds):

| Projected layers | Accuracy | Score gap | Flip rate | Probe h₀ / h₁ / h₂ / h₃ |
|---|---|---|---|---|
| h₀ (**used for M_proj**) | 0.857 | −0.008 | 0.011 | 0.52 / 0.51 / 0.50 / 0.50 |
| h₀–h₁ | 0.856 | −0.008 | 0.009 | 0.52 / 0.51 / 0.50 / 0.50 |
| h₀–h₂ | 0.826 | −0.006 | 0.029 | 0.52 / 0.51 / 0.50 / 0.50 |
| h₀–h₃ | 0.764 ± 0.145 | −0.002 | 0.031 | 0.52 / 0.51 / 0.50 / 0.51 |
| h₃ only | 0.500 | −0.000 | 0.000 | 0.85 / 0.70 / 0.57 / 0.53 |

Each extra late layer projected costs accuracy without removing more gender. This changes the plan for the tool: **project where the bias enters, re-audit, and only go deeper if the probes still find gender.**

**Caveats**

- **Small injection:** only 7–17 labels are flipped per seed, so seed-to-seed variance is high.
- **Linear only:** the probes are linear, so a nonlinear probe may still find gender in M_proj.
- **Job listings unused:** the MLP scores the resume alone. The job description is needed in Part 2 (LM prompt).

### Next

- [ ] A stronger adversarial baseline: a confusion objective (push the adversary towards 0.5) instead of gradient reversal, or several adversary steps per model step
- [ ] Nonlinear (MLP) probe and paired probe stress tests, especially on M_proj
- [ ] More seeds (20) and a `--flip-p` sweep; held-out names
- [ ] Headline plot of probe accuracy against layer for all four models; move results into `resume_dataset.ipynb`
- [ ] Part 2: wrap the stages into the BiasAudit tool (project → re-audit → go deeper only if needed); validate on the MLPs, then run on Qwen2.5-0.5B and a second open LM
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
