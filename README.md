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
- **Why balance 50/50 within each label:** then $P(y=1\mid F) = P(y=1\mid M)$ exactly. Gender is independent of the label by construction, so any gender effect is bias.
- **Why inject bias:** Set B simulates a biased historical recruiter. Equally qualified men are hired 100% of the time and women about 70%. Because we know the true gap, we can check that we detect it and that mitigation removes it.
- **Why the control also sees gender:** M₀ and M_b differ only in their labels. A blind model would trivially show no gap and wouldn't isolate the injected bias.

**Architecture:** resume text → frozen MiniLM encoder (384-d, $h_0$) → MLP with 3 hidden layers ($h_1, h_2, h_3$) → sigmoid shortlist score.

## Measuring bias

### Behavioural bias (outputs)

We run each test pair $(x_i^F, x_i^M)$ through the **same** trained model:

- **Score gap:** $\Delta = \frac{1}{n}\sum_i \big(f(x_i^F) - f(x_i^M)\big)$
- **Flip rate:** the fraction of pairs where the shortlist decision changes
- **Per-gender confusion matrix:** the gap in true-positive rates (equal opportunity)

### Activation-space bias (every layer)

A **hidden state** $h_\ell$ is the vector of numbers the model produces at layer $\ell$, i.e. its internal representation of the candidate.

- **Bias vector:** $v_\ell = \frac{1}{n}\sum_i \big(h_\ell(x_i^F) - h_\ell(x_i^M)\big)$. This is the average shift caused by swapping gender alone. We report it relative to the size of the activations, as $\lVert v_\ell\rVert / \mathbb{E}\lVert h_\ell\rVert$.
- **Gender probe:** a logistic-regression classifier predicting gender from $h_\ell$, scored on a **held-out split**. Accuracy near 50% means gender isn't linearly recoverable.
- **Activation-space bias** is the excess over the control, $\text{acc}_\ell(M_b) - \text{acc}_\ell(M_0)$. Early layers always contain gender, because the name is in the input.

### Why output-only fixes are not enough

With a linear readout $w$, the behavioural gap is

$$w^\top v_L = \lVert w\rVert\,\lVert v_L\rVert\cos\theta.$$

This is zero whenever $v_L \perp w$, even if $\lVert v_L\rVert$ is large. An output-only fix can **rotate** the bias out of view instead of removing it. Collapsing the bias vector instead protects **every** possible head $w'$, by Cauchy–Schwarz:

$$|w'^\top v| \le \lVert w'\rVert\,\lVert v\rVert = 0.$$

## Mitigation

### (a) Adversarial debiasing

The encoder $E_\theta$ feeds both a hiring head $H_\phi$ and gender adversaries $A_\psi$ at several layers:

$$\min_{\theta,\phi}\;\max_{\psi}\;\; \mathcal{L}_{\text{hire}}(\theta,\phi) - \lambda\,\mathcal{L}_{\text{adv}}(\theta,\psi)$$

A **gradient reversal layer** acts as the identity in the forward pass and multiplies the gradient by $-\lambda$ in the backward pass. The encoder therefore keeps hiring-relevant information while making gender hard to recover.

### (b) Orthogonal projection (closed form, no retraining)

1. Estimate the gender direction $v_\ell$ from the counterfactual pairs, as above.
2. Build

$$P_\ell = I - \frac{v_\ell v_\ell^\top}{v_\ell^\top v_\ell}$$

3. Replace $h_\ell$ with $P_\ell h_\ell$.

$P_\ell$ projects every hidden state onto the subspace **orthogonal to the gender direction** (the null space of $v_\ell^\top$). It removes the gender component and leaves everything else untouched.

- $Pv = 0$, so the bias vector collapses to zero.
- $P^2 = P$ and $P^\top = P$. It removes exactly one dimension, and it is the minimal edit: $Ph$ is the closest gender-free point to $h$.
- If the bias spans several directions, use $P = I - U_kU_k^\top$ with $U_k$ the top-$k$ singular vectors of the stacked pair differences.
- Apply it **layer by layer**, recomputing $v_\ell$ after each earlier projection, because later layers can rebuild the bias.
- To make it permanent, fold it into the weights: $W_{\ell+1}' = W_{\ell+1}P_\ell$.
- Estimate $v_\ell$ on one half of the test pairs and evaluate on the other half.

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
| **Score** | $\text{logit}(\text{Yes}) - \text{logit}(\text{No})$ |
| **Hidden state** | The last token's hidden state at each layer (residual stream) |
| **Mitigation** | Projection through forward hooks, with no retraining. Adversarial fine-tuning (e.g. LoRA) is future work. |
| **Access** | A full audit needs open weights. Closed APIs only allow the behavioural audit. |

## Limitations and risks

- **Linear guarantee only.** Projection removes linearly recoverable gender, and adversarial training can leave gender recoverable by a fresh probe. We stress-test with a nonlinear (MLP) probe.
- **Proxies.** Signals such as clubs or career gaps aren't covered by a name swap.
- **Binary gender.** Treating gender as binary excludes non-binary candidates. The method extends to $k$ groups, but stays categorical.
- **Proxy labels.** Category match isn't a real hiring outcome.
- **Utility trade-off.** Removing information may cost accuracy, so we report it.
- **Sociotechnical risks.** Fairwashing (a clean audit isn't fair hiring), over-trust in a linear guarantee, and GDPR concerns when collecting protected attributes for auditing. Humans stay in the loop.

## Data

- Resume dataset: 2,484 resumes across 24 categories ([Kaggle: snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)). It's too large to commit; download it locally into `resume_dataset/`.
- Job listings dataset (provided by the organisers).
