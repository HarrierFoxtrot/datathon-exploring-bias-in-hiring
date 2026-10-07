# datathon-exploring-bias-in-hiring

Aiming to address and mitigate **behavioural** and **activation-space** gender bias in hiring models.

180DC Bristol × BDSS Datathon, 7 Oct 2026. Builds on H. Liu, *Evaluating Weight-Level Bias Mitigation Against Behavioral and Activation-Space Bias* (Imperial College London).

## The problem

We focus on two types of gender bias in hiring models.

- **Behavioural bias** is what we can directly observe in the model's output. For example, otherwise identical male-coded and female-coded resumes receive different suitability scores.
- **Activation-space bias** is hidden inside the model's internal representations. A model can look fair at the output while still encoding gender internally, which can be read out again by a retrained head or a downstream system.

The thesis shows that removing behavioural bias does **not** imply removing activation-space bias. Our goal is a hiring model that checks itself for both.

## Pipeline

```
1. Pick one industry (e.g. ENGINEERING)
2. Match resumes to a job description (y=1 same category, y=0 other categories)
3. Redact direct PII (the dataset is already mostly redacted)
4. Create controlled male/female counterfactual versions
   (identical except for a name or pronouns)
5. Inject bias and train two hiring models
6. Measure behavioural bias
7. Extract hidden representations layer by layer
8. Train a gender probe on each layer
9. Compare where gender becomes recoverable (biased vs control)
10. Apply mitigation, then repeat steps 6–9
```

### Data and models

| Split | Contents | Used for |
|---|---|---|
| Train set A | Resumes with one **random** gender each, true labels | Control model **M₀** |
| Train set B | Same resumes and genders, but female `y=1` flipped to `0` with p = 0.3 | Biased model **M_b** |
| Test set | Held-out resumes, each in **both** a female and a male version | Every evaluation |

- **Why random gender in training:** gender is then independent of qualification by design, so any gender effect is bias.
- **Why inject bias:** it simulates a biased historical recruiter and gives a known ground truth. We can check that we detect roughly the bias we put in, and that it disappears after mitigation.
- **Why the control also sees gender:** the two models differ only in their labels. So the difference M_b − M₀ isolates the injected bias, not just the effect of the name being present.

**Architecture:** a frozen sentence encoder (MiniLM, 384-d) followed by an MLP with 3 hidden layers and a sigmoid shortlist score. The MLP hidden layers are the layers we probe.

## 1. Measuring behavioural bias

We use controlled counterfactual resume pairs $(x_i^F, x_i^M)$ and the **same** trained model:

- **Score gap:** $\Delta = \frac{1}{n}\sum_i \big(f(x_i^F) - f(x_i^M)\big)$
- **Flip rate:** the fraction of pairs where the shortlist decision changes
- **Confusion matrix per gender:** compare true-positive rates (equal opportunity)

## 2. Measuring activation-space bias

For each layer $\ell$ we extract activations $h_\ell$ and compute two things.

- **Gender probe:** a logistic-regression classifier predicting gender from $h_\ell$, scored on a **held-out split**. Accuracy near 50% means gender isn't linearly recoverable. High accuracy means the representation contains gender information.
- **Bias vector:** $v_\ell = \frac{1}{n}\sum_i \big(h_\ell(x_i^F) - h_\ell(x_i^M)\big)$, reported as $\lVert v_\ell\rVert / \mathbb{E}\lVert h_\ell\rVert$.

Early layers always contain gender because the name is in the input. So **activation-space bias = excess over the control**, i.e. $\text{acc}_\ell(M_b) - \text{acc}_\ell(M_0)$.

### Why output-only fixes are not enough

With a linear readout $w$, the behavioural gap is

$$w^\top v_L = \lVert w\rVert\,\lVert v_L\rVert\cos\theta.$$

This is zero whenever $v_L \perp w$, even when $\lVert v_L\rVert$ is large. An output-only fix can therefore **rotate** the bias out of view instead of removing it. In contrast, collapsing $\lVert v_\ell\rVert \to 0$ gives

$$|w'^\top v| \le \lVert w'\rVert\,\lVert v\rVert = 0$$

for **any** head $w'$.

## 3. Mitigation

### (a) Adversarial debiasing (main method)

The encoder $E_\theta$ feeds both a hiring head $H_\phi$ and a gender adversary $A_\psi$:

$$\min_{\theta,\phi}\;\max_{\psi}\;\; \mathcal{L}_{\text{hire}}(\theta,\phi) - \lambda\,\mathcal{L}_{\text{adv}}(\theta,\psi)$$

- The hiring head is rewarded for correctly predicting suitability.
- The adversary tries to recover gender from the hidden representation.
- A **gradient reversal layer** sits between encoder and adversary. In the forward pass it is the identity. In the backward pass it multiplies the gradient by $-\lambda$. So the encoder keeps hiring-relevant information while making gender hard to identify.

We apply adversaries at several hidden layers, not just the last one, to target bias at every layer.

### (b) Projection (closed-form alternative)

Given the gender direction $v_\ell$ at layer $\ell$:

$$P_\ell = I - \frac{v_\ell v_\ell^\top}{v_\ell^\top v_\ell}, \qquad h_\ell' = P_\ell h_\ell$$

$P_\ell$ projects every hidden state onto the subspace **orthogonal to the gender direction**, which is the null space of $v_\ell^\top$.

- $P v = 0$, so the bias vector collapses: the new bias is $P_\ell v_\ell = 0$.
- $P^2 = P$ and $P^\top = P$. It removes exactly one dimension and is the minimal edit, since $Ph$ is the closest gender-free point to $h$.
- For a $k$-dimensional bias subspace with orthonormal basis $U_k$ (from SVD): $P = I - U_kU_k^\top$.
- Apply it layer by layer, recomputing $v_\ell$ after each earlier projection, because later layers can rebuild the bias.
- It can be made permanent by folding it into the weights: $W_{\ell+1}' = W_{\ell+1}P_\ell$.

## 4. Evaluation

We rerun the behavioural and activation-space tests on **M₀, M_b, M_b + adversarial, and M_b + projection**, and check that:

1. The score gap and flip rate drop to about 0.
2. Probe accuracy drops to the control's level at **every** layer, not only the output layer.
3. Hiring accuracy is maintained.

**Headline plot:** probe accuracy against layer for all four models, alongside a table of gap, flip rate and accuracy.

## Limitations and risks

- **Linear guarantee only.** Projection removes linearly recoverable gender, and adversarial training is known to leave some gender recoverable by a fresh probe. We stress-test with a nonlinear (MLP) probe.
- **Proxies.** Gender signals such as clubs or career gaps are not covered by the swapped name.
- **Binary gender.** Treating gender as binary excludes non-binary candidates. The method extends to $k$ groups, but stays categorical.
- **Utility trade-off.** Removing information may cost accuracy, so we report it.
- **Sociotechnical risks.** Fairwashing (a clean dashboard is not fair hiring), over-trust in a linear certificate, and GDPR concerns when collecting protected attributes for auditing. Humans stay in the loop.

## Data

- Resume dataset: 2,484 resumes across 24 categories (`Resume.csv`).
- Job listings dataset (provided by the organisers).
