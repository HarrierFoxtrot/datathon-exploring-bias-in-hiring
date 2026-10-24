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

3. Replace h_ℓ with P_ℓ h_ℓ.

P_ℓ projects every hidden state onto the subspace **orthogonal to the gender direction** (the null space of v_ℓᵀ). It removes the gender component and leaves everything else untouched.

- Pv = 0, so the bias vector collapses to zero.
- P² = P and Pᵀ = P. It removes exactly one dimension, and it is the minimal edit: Ph is the closest gender-free point to h.
- If the bias spans several directions, use P = I − U_k U_kᵀ with U_k the top-k singular vectors of the stacked pair differences.
- Apply it **layer by layer**, recomputing v_ℓ after each earlier projection, because later layers can rebuild the bias.
- To make it permanent, fold it into the weights: `W′_{ℓ+1} = W_{ℓ+1} P_ℓ`.
- Estimate v_ℓ on one half of the test pairs and evaluate on the other half.

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

### Done: Part 1, stages 1–6

```bash
pip install sentence-transformers torch scikit-learn pandas
python3 run_experiment.py --seeds 5     # writes results_behavioural.csv, results_activation.csv
```

| File | What it does |
|---|---|
| [biasaudit/data.py](biasaudit/data.py) | Stages 1–4: hiring task, gender neutralising, 70/30 split by resume, name injection, Sets A and B |
| [biasaudit/model.py](biasaudit/model.py) | Stage 5: frozen `all-MiniLM-L6-v2` encoder (h₀) → MLP 256/128/64 (h₁–h₃) → sigmoid |
| [biasaudit/audit.py](biasaudit/audit.py) | Stage 6: behavioural metrics, bias-vector ratio and gender probe per layer |
| [run_experiment.py](run_experiment.py) | Runs stages 1–6 over several seeds and prints mean ± sd |

**Implementation details**

- **Positives and negatives:** 118 ENGINEERING resumes (y = 1) and 118 random others (y = 0), giving 166 train resumes and 70 test pairs.
- **Gender neutralising:** he/she/him/his/her/hers/himself/herself become they/them/their, and Mr/Mrs/Miss are removed. "ms" is left alone, because it's nearly always "MS Office" or an MS degree.
- **Gender signal:** each resume starts with `Candidate: <name> Taylor.` Resumes average about 840 words, more than MiniLM's 256-token limit, so the name goes first to survive truncation. There are 8 female and 8 male names, index-matched in test pairs (Emily ↔ James, …).
- **Training:** M₀ and M_b share the same seed, so they start from identical weights and differ only in their labels.
- **Gender probe:** a logistic regression, trained and tested on different resumes so it can't memorise their content, averaged over 20 random splits.

**First results** (5 seeds, mean ± sd):

| | M₀ (control) | M_b (biased) |
|---|---|---|
| Score gap (F − M) | +0.003 ± 0.008 | **−0.047 ± 0.030** |
| Flip rate | 0.003 ± 0.006 | **0.046 ± 0.036** |
| TPR gap (F − M) | +0.006 ± 0.013 | **−0.063 ± 0.059** |
| Accuracy | 0.950 ± 0.027 | 0.860 ± 0.057 |

| Layer | Probe acc M₀ | Probe acc M_b | ‖v‖/E‖h‖ M₀ | ‖v‖/E‖h‖ M_b |
|---|---|---|---|---|
| h₀ (encoder) | 0.852 | 0.852 | 0.106 | 0.106 |
| h₁ | 0.688 | 0.700 | 0.013 | 0.039 |
| h₂ | 0.527 | **0.571** | 0.012 | **0.058** |
| h₃ | 0.506 | **0.527** | 0.014 | **0.084** |

The injected bias is detectable both at the output and in the hidden layers, and it grows towards the later layers, as the thesis predicts. h₀ is identical for both models because the encoder is frozen.

**Caveats**

- **Small injection:** only 7–17 labels are flipped per seed, so seed-to-seed variance is high. More seeds or a larger `--flip-p` would tighten it.
- **Job listings unused:** the MLP scores the resume alone. The job description is needed in Part 2 (LM prompt).

### Next

- [ ] Stage 7(b): orthogonal projection P_ℓ, applied layer by layer → M_proj
- [ ] Stage 7(a): adversarial debiasing with gradient reversal → M_adv
- [ ] Stage 8: re-test M₀ / M_b / M_adv / M_proj; headline plot of probe accuracy against layer
- [ ] Nonlinear (MLP) probe stress test
- [ ] Move the results and plots into `resume_dataset.ipynb`
- [ ] Part 2: wrap the stages into the BiasAudit tool; validate on the MLPs, then run on Qwen2.5-0.5B

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
