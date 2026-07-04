# Experimental Strategy for "A Variational Analysis Approach for Bilevel Hyperparameter Optimization with Sparse Regularization"

## Guiding Principle

The paper has two distinct contributions that need empirical support, and they require different types of experiments:

1. The **FBE reformulation** is theoretically exact (preserves the solution set), unlike BE smoothing.
2. The **descent-aligned biactive policy** resolves gradient starvation that standard implicit-differentiation methods suffer.

These are not the same claim. The first is about the *reformulation*, the second is about the *differentiation*. The experiments should make this distinction visible.

### Narrative arc

> Feature-wise regularization is more powerful than scalar, but only if you can actually optimize the vector hyperparameter. Existing methods can't, because they go blind at the regularization boundary. Here's why, here's that it matters in practice, and here's that our method fixes it.

---

## Experiment 1 — Illustrative: why FBE, and why biactive points are dangerous

**One figure, two panels.** No randomness, no seeds, no statistics — both panels are proofs by construction. The reader should be able to verify every number by hand.

---

### Panel A — FBE preserves the solution set; BE smoothing does not

**Purpose:** show concretely that replacing the nonsmooth lower-level cost with the Berkovier-Engelman (BE) smooth surrogate introduces a systematic bias in the minimizer, whereas the FBE reformulation is exact. This motivates the theoretical choice of FBE over smoothing.

**Instance:** A 2D lower-level with diagonal design, so each coordinate decouples and every value can be computed in closed form.

$$\min_{y \in \mathbb{R}^2} \left\{ \frac{1}{2}\|y - d\|^2 + e^{x_1}|y_1| + e^{x_2}|y_2| \right\}$$

Choose $d = (0.8,\, 0.5)$ and hyperparameters such that $e^{x_1} = 0.6$, $e^{x_2} = 0.6$. The true minimizer is:

$$y^* = \bigl(\max(0.8 - 0.6,\, 0),\; \max(0.5 - 0.6,\, 0)\bigr) = (0.2,\; 0)$$

Feature 2 is at the boundary: $d_2 = 0.5 < e^{x_2} = 0.6$, so $y^*_2 = 0$ exactly. This is a biactive coordinate.

**Three objects to plot side by side** in the $(y_1, y_2)$ plane:

1. **True lower-level cost** $\frac{1}{2}\|y - d\|^2 + e^{x_1}|y_1| + e^{x_2}|y_2|$: circular fidelity contours distorted by the ℓ1 diamond. The minimizer sits at the kink $(0.2, 0)$ — on the boundary of the ℓ1 ball, at a non-smooth point of the cost.

2. **BE-smoothed cost** $\frac{1}{2}\|y - d\|^2 + e^{x_1}\sqrt{y_1^2 + \gamma^2} + e^{x_2}\sqrt{y_2^2 + \gamma^2}$: fully smooth everywhere. The minimizer is pulled strictly into the interior — $y^*_{2,\text{BE}} > 0$ for any fixed $\gamma > 0$. Show two values, e.g., $\gamma = 0.1$ and $\gamma = 0.5$, to visualize how the bias grows with $\gamma$.

3. **FBE cost** $\varphi_x^\gamma(y)$: smooth, but its global minimizer coincides exactly with $(0.2, 0)$ by construction. The level sets look different from the true cost, but the minimizer is identical.

**The key visual:** in panel 2, the minimizer has drifted away from the ℓ1 boundary into $y_2 > 0$. In panel 3, the minimizer sits exactly where panel 1 says it should. This is the central claim of the FBE construction, illustrated geometrically.

**Second subplot (optional but recommended):** plot the upper-level objective $\Phi(x_2)$ as a function of $x_2$ alone (fixing $x_1$) for three curves — true bilevel, BE-based bilevel ($\gamma = 0.1$, $\gamma = 0.5$). The true curve has a kink at $e^{x_2} = d_2 = 0.5$ (the biactive transition). The BE curves are smooth but their minimizers are shifted leftward — they prescribe a penalty that is too weak, leaving a spurious nonzero $y^*_2$. This shows that using BE smoothing in the bilevel loop would converge to the wrong hyperparameter.

---

### Panel B — Biactive features cause permanent gradient starvation

**Purpose:** show, in the smallest possible example, that primal-support implicit differentiation assigns an identically zero hypergradient to a validation-relevant feature that happens to sit at the ℓ1 threshold. This is not a numerical imprecision — it is an exact zero that persists across all outer iterations. The descent-aligned oracle resolves it immediately.

**Instance:** $p = 3$ features, diagonal design, all numbers verifiable by hand.

$$A^\text{tr} = A^\text{val} = \sqrt{3}\, I_3, \quad b^\text{tr} = \sqrt{3}\begin{bmatrix}0.4\\1.2\\0\end{bmatrix}, \quad b^\text{val} = \sqrt{3}\begin{bmatrix}0\\1.2\\0\end{bmatrix}, \quad e^{x^0} = \begin{bmatrix}0.35\\1.20\\2.00\end{bmatrix}$$

With $\gamma = 1$, the lower-level simplifies to $F(y) = \frac{1}{2}\|y - d^\text{tr}\|^2$ where $d^\text{tr} = [0.4, 1.2, 0]^\top$, and the solution is soft thresholding:

$$y^*_j = \text{sign}(d^\text{tr}_j)\,\max(|d^\text{tr}_j| - e^{x_j},\, 0)$$

At initialization:
- **Feature 1** (distractor): $y^*_1 = \max(0.4 - 0.35, 0) = 0.05$ — strictly active.
- **Feature 2** (golden): $y^*_2 = \max(1.2 - 1.20, 0) = 0$ — **biactive by construction**: $|d^\text{tr}_2| = e^{x^0_2} = 1.20$ exactly.
- **Feature 3** (noise): $y^*_3 = \max(0 - 2.00, 0) = 0$ — strictly inactive.

With $\gamma = 1$, the proximal argument $w = y^* - \gamma\nabla F(y^*) = d^\text{tr} = [0.4, 1.2, 0]^\top$. Biactivity of feature 2 is confirmed: $|w_2| = 1.2 = e^{x^0_2}$.

**Upper-level loss at initialization:**

$$\Phi(x^0) = \frac{1}{2}\|d^\text{val} - y^*(x^0)\|^2 = \frac{1}{2}\bigl[(0.05)^2 + (1.2)^2 + 0\bigr] \approx 0.721$$

The dominant term is $(1.2 - y^*_2)^2/2 = 0.72$ — feature 2 is the only feature that matters on the validation split, and it is stuck at zero.

**Validation gradient:**

$$z^* = \nabla_y L(x^0, y^*) = y^* - d^\text{val} = \begin{bmatrix}0.05\\ -1.2\\ 0\end{bmatrix}$$

Feature 2 needs to increase toward $d^\text{val}_2 = 1.2$ — $z^*_2 = -1.2$ signals a strong descent direction.

**Null oracle (Sparse-HO):** working set $\mathcal{S} = \{j : y^*_j \neq 0\} = \{1\}$. Feature 2 is not in the strict support.

The adjoint system on $\mathcal{S}$: for the diagonal case with $\gamma = 1$, $H_{11} = 1$, so $p_1 = z^*_1 = 0.05$.

The hypergradient component for feature 2: $h_2 = 0$ — **exactly zero**, not approximately. This is a structural consequence of the support restriction, not a numerical issue. No refinement of the inner solver or outer step size can change this.

$x_2$ receives no update. $y^*_2$ stays at zero. $\Phi$ stays near 0.72 indefinitely.

**DA oracle:** feature 2 is biactive ($|w_2| = e^{x^0_2}$) with $w_2 = 1.2 > 0$, so it belongs to $\mathcal{B}^+$. The descent-alignment condition requires $z^*_j < 0$ for $j \in \mathcal{B}^+$, and $z^*_2 = -1.2 < 0$ — feature 2 enters the working set.

Augmented adjoint on $\mathcal{S} = \{1, 2\}$: $p_1 = 0.05$, $p_2 = -1.2$.

The hypergradient component for feature 2 is now nonzero (positive in this case — increasing $x_2$ keeps the penalty high and hurts the validation, so $h_2 > 0$ and the descent step $x_2 \leftarrow x_2 - \eta h_2$ correctly decreases $x_2$). Once $e^{x_2} < 1.2$, feature 2 enters the model: $y^*_2 = 1.2 - e^{x_2} > 0$, and the validation loss drops.

**What to plot (three subplots, outer iterations on the x-axis):**

1. **$x_2^k$ trajectory:** Sparse-HO flat at $\ln(1.2) \approx 0.18$ throughout; NTRBA monotonically decreasing after the first step.
2. **$y^*_2(x^k)$ trajectory:** Sparse-HO at zero throughout; NTRBA activating at step 1 and growing toward $d^\text{val}_2 = 1.2$.
3. **Validation loss $\Phi(x^k)$:** Sparse-HO plateauing near 0.72; NTRBA descending toward zero as feature 2 is recovered.

Subplot 1 shows *what the method does to the hyperparameter*. Subplot 2 shows *why it matters for the model*. Subplot 3 shows *that it matters for the objective*. Together they make the failure mode and its resolution completely transparent.

---

## Experiment 2 — Synthetic regression: does feature-wise regularization actually help?

This is the experiment the paper implicitly assumes but never cleanly demonstrates. Before showing *how* to optimize vector hyperparameters, you need to show *why* you'd want to.

**Model:** Weighted elastic-net, consistent with the paper's framework:

$$\min_{y} \frac{1}{2n_\text{tr}}\|A^\text{tr}y - b^\text{tr}\|^2 + \frac{\alpha}{2}\|y\|^2 + \sum_j \exp(x_j)|y_j|$$

with α > 0 fixed and x ∈ ℝ^m learned via bilevel. The fixed ridge term is not a tunable hyperparameter — it supplies the strong convexity required by the theory (Assumption on F).

**Setup:** Overparameterized linear regression (n ≪ m). Ground truth is sparse with three feature groups:
- Signal features (should survive, should get small penalty)
- Correlated noise features (should be suppressed, but hard to distinguish from signal on training data)
- Pure noise (should be zeroed out)

**Compare:**
- **Scalar elastic-net CV**: same model with x = λ**1** (one global λ for the ℓ1 part, selected by grid search over 100 values; α fixed identically). This is the apples-to-apples baseline — same regularizer family, different expressivity.
- **Weighted elastic-net (ours)**: full vector x ∈ ℝ^m learned via bilevel.

Using elastic-net rather than pure Lasso for both methods is important for two reasons:
1. It is consistent with the paper's own assumptions (strong convexity of the lower-level objective).
2. It isolates the right failure mode: pure Lasso with correlated features picks one from each correlated group arbitrarily (instability), which would make the scalar baseline look bad for the wrong reason. Elastic-net stabilizes within-group selection, so when the scalar baseline fails it fails because it *lacks feature resolution*, not because of numerical instability.

**Report:** Learned weight profile (as a heatmap over features), support recovery F1, test MSE. Vary n/m ratio and sparsity level.

**The message:** The failure of the scalar model is not about the regularizer family — it is about the expressivity of a single global parameter. Feature-wise regularization learns to suppress the correlated noise while preserving signal, which a scalar cannot do without also suppressing signal. *This* is why vector hyperparameters are worth the effort.

---

## Experiment 3 — Ablation: oracle × optimizer, on degenerate synthetic data

This experiment isolates and attributes the paper's two algorithmic contributions independently. The method proposes two changes over the Sparse-HO baseline: a new hypergradient oracle (the descent-aligned biactive selection policy) and a new outer-level optimizer (the nonsmooth trust-region NTRBA). These operate at different stages of the bilevel procedure — the oracle changes *what gradient information is available*, while the optimizer changes *how that information is used*. Without a controlled ablation, it is impossible to tell which component is responsible for any observed gain, and a reviewer is right to ask.

**Model:** Same weighted elastic-net as Experiments 2 and 4. The same degenerate four-group dataset from Experiment 4 is reused here, since the biactive degeneracy is what makes the oracle distinction meaningful. On non-degenerate data, all four combinations would perform identically — the ablation would have nothing to measure.

### The 2×2 design

Two binary axes are varied independently, yielding four methods:

| | Null biactive policy (baseline oracle) | Descent-aligned policy (DA oracle) |
|---|---|---|
| **NBA** (subgradient) | `NBA-null` | `NBA-DA` |
| **NTRBA** (trust-region) | `NTRBA-null` | `NTRBA-DA` |

- **Null biactive policy:** the adjoint system is solved on the strict primal support $\mathcal{S} = \{j : y^*_j \neq 0\}$ only, exactly reproducing the Sparse-HO oracle behavior. Biactive features receive a zero hypergradient component.

- **Descent-aligned (DA) policy:** the working set is augmented with biactive coordinates that satisfy the descent-alignment condition (Definition in the paper), restoring a nonzero gradient on hidden features.

- **NBA:** projected normalised subgradient method with a fixed step rule. Simple and parameter-light, but sensitive to gradient bias — a biased oracle produces a biased iterate sequence with no mechanism to self-correct.

- **NTRBA:** nonsmooth trust-region method with adaptive radius. The radius contracts near non-smooth ridges and expands in smooth regions. This provides robustness to gradient noise but cannot create gradient information that the oracle does not provide.

### What each cell of the table should reveal

- **`NBA-null` vs. `NBA-DA`:** isolates the effect of the oracle with the simplest outer solver. If the DA policy helps, it should be visible here first and most clearly, since NBA has no adaptive mechanism to compensate for oracle bias.

- **`NTRBA-null` vs. `NTRBA-DA`:** same oracle comparison but with the more sophisticated solver. If NTRBA's adaptive radius can partially compensate for a biased oracle (e.g., by taking smaller steps in bad directions), the gap between null and DA should be smaller here than in the NBA row.

- **`NBA-null` vs. `NTRBA-null`:** isolates the effect of the optimizer when both use the biased oracle. The trust-region's step adaptivity may still improve stationarity even when the gradient direction is wrong — but it cannot improve the objective value beyond what the oracle permits.

- **`NBA-DA` vs. `NTRBA-DA`:** the full combination comparison. Both have the correct oracle; the only difference is the outer solver. This is where the complementarity of the two contributions should be most visible: the DA oracle provides the right direction, and NTRBA uses it more efficiently.

The expected finding, grounded in the theory, is that the oracle policy controls *where* the outer loop converges (validation loss, support quality), while the outer optimizer controls *how efficiently* it gets there (stationarity, gradient norm at convergence). These are genuinely orthogonal: a good oracle with a crude solver converges to a better point than a bad oracle with a sophisticated solver, but more slowly.

### Sweep and seeds

Run across problem sizes m ∈ {250, 500, 1000}, with 5 independent random seeds per size, for a total of 20 runs per cell (4 cells × 5 seeds × ... averaged across seeds). Using multiple problem sizes is important: at small m, the trust-region's step adaptivity may matter even with a biased oracle (because the landscape has fewer flat biactive ridges); at large m, the policy gap should dominate.

All four methods share identical inner solver settings (FISTA to tolerance 10⁻⁸, max 2,000 iterations), identical outer iteration budget (60 steps), and identical initialization. Any difference in outcome is therefore attributable solely to the oracle or the optimizer, not to computational budget or starting point.

### Metrics

Four metrics, measuring different aspects of the solution quality:

1. **Validation loss gap** Δℓ = ℓ_method − ℓ_NTRBA-DA, computed per seed before averaging. Using a per-seed relative gap removes the effect of problem-difficulty variation across seeds — what matters is not the absolute loss but how far each method falls short of the best available combination.

2. **Final hypergradient norm** ‖h_K‖: proximity to a stationary point of the outer objective. This is the stationarity metric. The trust-region methods should dominate here regardless of oracle, because NTRBA is specifically designed to drive this quantity to zero efficiently.

3. **Support recovery F1:** quality of the recovered active set. This should track the oracle policy, not the optimizer, since identifying the correct support requires having a nonzero gradient on the hidden features in the first place.

4. **Wall-clock time per outer iteration:** to confirm that the DA policy and the trust-region overhead do not make the method impractical. The biactive detection (computing the proximal argument and checking the threshold) is O(m); the augmented adjoint system is O(|S|³) which is dominated by the oracle cost already paid. The overhead should be negligible.

### The convergence figure

A two-panel dynamics plot for a single representative instance (m = 500, one seed):

- **(a) Validation loss trajectory** over outer iterations for all four methods. `NBA-null` and `NTRBA-null` should plateau or oscillate; `NBA-DA` and `NTRBA-DA` should descend, with `NTRBA-DA` showing the smoothest monotone behaviour.
- **(b) Trust-region radius** Δ_k for the two NTRBA variants over iterations. The radius should contract sharply when the iterate crosses a biactive ridge (non-smooth boundary) and expand in smoother regions. This reveals the geometry the trust-region is navigating and justifies the adaptive mechanism qualitatively.

### The message

The ablation should deliver two separable conclusions, stated explicitly in the paper:

> The descent-aligned oracle is the primary driver of generalization quality: without it, neither NBA nor NTRBA can escape the gradient-starvation regime, regardless of step adaptivity. The nonsmooth trust-region is the primary driver of stationarity: it drives the hypergradient norm one to two orders of magnitude lower than the subgradient method, regardless of which oracle is used. The two contributions are complementary — their combination is the only configuration that is simultaneously best on all metrics across all problem sizes.

---

## Experiment 4 — SOTA comparison: gradient starvation on degenerate instances

This is the paper's centerpiece experiment. Its purpose is to show that gradient starvation — the phenomenon whereby primal-support implicit differentiation assigns an identically zero hypergradient to validation-relevant features — is not a pathological edge case but a structural failure that occurs systematically in the presence of feature collinearity.

**Model:** Same weighted elastic-net as Experiment 2, now used in a deliberately degenerate regime.

### The construction

The dataset is built around four feature groups. Each group is designed to expose a specific aspect of the failure mode:

- **Easy features:** strictly active at the inner optimum on both training and validation splits. The inner solver selects these features regardless of initialization; no method struggles here. They serve as a reference — all methods should recover them, and any method that doesn't is broken.

- **Distractor features:** strictly active on training, near-collinear with the hidden group (correlation ρ), but carrying no validation signal. Because the training objective explains the response well through the distractors alone, the inner solver has no incentive to also activate the hidden features. These are the features that "absorb" the training signal and crowd out their hidden counterparts.

- **Hidden features:** biactive at the inner optimum by construction — i.e., $y^*_\text{hid} = 0$ and $|w_j| = \gamma\exp(x_j)$ at initialization, where $w = y - \gamma\nabla F(y)$ is the proximal argument. They carry *only* validation signal. Because of near-collinearity with the distractors on the training split, the inner solver consistently prefers the distractors, leaving the hidden features exactly at the ℓ1 threshold. These are the features the paper's method must recover.

- **Noise features:** uncorrelated with everything, always inactive. They provide a background that tests whether methods over-activate spurious features.

The parameter ρ ∈ {0.90, 0.95, 0.98} controls the distractor–hidden collinearity. Higher ρ makes the degeneracy more severe: the Hessian coupling between distractor and hidden blocks in the reduced adjoint system is proportional to ρ, so the variational correction that NTRBA provides grows with ρ while Sparse-HO's starvation remains total regardless of ρ.

### Why gradient starvation is exact, not approximate

For Sparse-HO (wℓ1), the hypergradient is computed by solving the adjoint system restricted to the primal algebraic support $\mathcal{S} = \{j : y^*_j \neq 0\}$. Since hidden features are biactive ($y^*_\text{hid} = 0$), they are not in $\mathcal{S}$. Consequently:

$$\frac{\partial \Phi}{\partial x_j} = 0 \quad \forall j \in \text{hidden}$$

This is not a numerical approximation or a finite-difference artefact — it is an exact zero that follows directly from the support restriction. No amount of inner-solver accuracy or outer-step refinement can produce a nonzero hidden gradient under this adjoint construction. The starvation is permanent throughout the outer loop.

For NTRBA-wℓ1, the descent-aligned biactive selection policy detects hidden features as biactive (within a specified tolerance $\delta_\text{abs}$) and adds them to the working set $\mathcal{S}$ when the corresponding validation gradient $z^*_j = \nabla_y \mathcal{L}(\bar{x}, \bar{y})_j$ has the descent-consistent sign. The augmented adjoint system then has a nonzero right-hand-side component for the hidden block, and the Hessian coupling

$$H_{\text{hid},\text{dist}} = \frac{\gamma\rho}{n_\text{tr}} (A^\text{tr}_\text{hid})^\top A^\text{tr}_\text{dist}$$

propagates a nonzero gradient to the hidden hyperparameters immediately at the first outer iteration. The magnitude of this correction grows with ρ, which is why the method's advantage is most visible at high correlation.

### Baselines

Three methods are compared:

- **Sparse-HO (scalar ℓ1):** a single global penalty $x \in \mathbb{R}$ optimized by implicit differentiation. Per-feature targeting is structurally impossible. At large m, this method can accidentally achieve high hidden recall by collapsing the global penalty close to zero, activating nearly all features — but this conflates signal and noise and yields the largest validation gap of all methods. It is included to show that brute-force activation is not the right strategy.

- **Sparse-HO (wℓ1):** per-feature penalties $x \in \mathbb{R}^m$, optimized via the standard support-restricted adjoint (gradient descent outer loop). This is the closest existing prior work. It has the right parameterization but the wrong differentiation. The starvation is exact and permanent for hidden features.

- **NTRBA-wℓ1 (ours):** same parameterization as above, differentiated via the generalized support adjoint with descent-aligned biactive selection, and updated with the nonsmooth trust-region outer solver.

### Metrics

Four metrics are reported, in order of proximity to the underlying mechanism:

1. **Hidden-feature gradient norm at iteration 0** $\|\nabla_{x_\text{hid}}\Phi\|_{k=0}$: the most direct measurement. For Sparse-HO(wℓ1) this should be near zero for all ρ and all problem sizes. For NTRBA it should be large and grow with ρ (Hessian coupling amplifies the signal at higher correlation). This metric measures the failure mode itself, not a downstream consequence.

2. **Hidden recall at final iterate:** fraction of hidden features correctly identified as active by the end of the outer loop. Sparse-HO(wℓ1) should show near-zero recall because hidden features never receive a nonzero gradient and are never released from the threshold. NTRBA should improve monotonically.

3. **Support recovery F1:** precision-recall balance over the full feature set. This penalizes methods that achieve recall by over-activating (like scalar Sparse-HO at large m).

4. **Validation gap Φ − Φ*:** difference from a reference solution obtained by a long NTRBA run. This is the summary optimization quality metric.

### The convergence figure

A four-panel convergence plot over outer iterations (for a fixed representative instance, e.g., (n,m) = (200, 300), ρ = 0.98):

- **(a) Validation loss gap** Φ − Φ*: should show NTRBA descending, Sparse-HO(wℓ1) plateauing early.
- **(b) Hidden-feature gradient magnitude** ‖∇_{x_hid}Φ‖: the key panel. Sparse-HO(wℓ1) should be a flat line at near-zero throughout. NTRBA should show a large initial signal that drives descent.
- **(c) Mean hidden penalty** (mean of exp(x_j) for j ∈ hidden): should decrease monotonically for NTRBA (features being released from the threshold) and remain constant for Sparse-HO(wℓ1) (no gradient → no update).
- **(d) Support recovery F1:** should improve for NTRBA as hidden features enter the model; flat for Sparse-HO(wℓ1).

Panel (b) is the proof panel. If the hidden gradient curve for Sparse-HO(wℓ1) is flat at zero and the curve for NTRBA is nonzero and drives the improvements in panels (c) and (d), then the experiment has shown what it claims.

### Sweep and reporting

The full table sweeps over problem sizes (n, m) ∈ {(100, 150), (200, 300), (500, 750)} and correlations ρ ∈ {0.90, 0.95, 0.98}, repeated over 5 seeds. This gives 45 rows (3 methods × 3 sizes × 3 correlations), reported as mean ± std. The primary observation to highlight is that $\|\nabla_{x_\text{hid}}\Phi\|_{k=0}$ for Sparse-HO(wℓ1) is at least one order of magnitude below NTRBA across all 9 configurations, and that this gap in gradient signal translates consistently into a gap in validation quality and support recovery.

---

## Experiment 5 — Scalability of the oracle

**Purpose:** show that the variational correction — augmenting the working set with biactive coordinates — does not destroy the computational tractability of the support-reduced oracle. Feature-wise regularization is most useful when $m$ is large; if the hypergradient oracle scales as $O(m^3)$, the method is dead on arrival for the problems it is designed to solve. This experiment establishes that it does not.

**What is being timed:** a single oracle call — one evaluation of a hypergradient element $h \in \partial\Phi(x)$ at a fixed $(x, y^*(x))$ pair. The outer optimization loop is not running; only the oracle computation is measured. This isolates the cost of the differentiation step from the cost of the inner solver and the outer iterations.

### The three systems being compared

1. **Full dense system:** solve the $m \times m$ adjoint linear system $H^\top p = z^*$ without any support reduction. This is the naive baseline — theoretically correct but scales as $O(m^3)$ in both time and $O(m^2)$ in memory. It becomes the practical bottleneck beyond $m \approx 10^3$.

2. **Support-reduced oracle (null policy):** solve the adjoint system restricted to the strict primal support $\mathcal{S} = \{j : y^*_j \neq 0\}$. With true sparsity density $\rho_s$ (fraction of nonzero coordinates), the working set has $|\mathcal{S}| = \rho_s m$ entries, and the system scales as $O((\rho_s m)^3)$. For $\rho_s = 5\%$, this is a factor of $0.05^3 \approx 1.25 \times 10^{-4}$ reduction in flops relative to the full system.

3. **Support-reduced oracle (DA policy):** same as above, but the working set is augmented with the biactive coordinates selected by the descent-aligned policy, $\mathcal{S} \cup M_{\mathcal{B}^+} \cup M_{\mathcal{B}^-}$. The overhead relative to (2) depends on how many biactive coordinates are added. In the worst case, all $m$ coordinates are biactive and the DA policy degenerates to the full dense system. In practice, biactive coordinates are a small fraction of the total, so the overhead should be negligible.

### Setup

Generate a random weighted elastic-net instance at each dimension level: draw $A \in \mathbb{R}^{n \times m}$ with Gaussian entries ($n = 0.1m$ to keep the problem overparameterized), set the ground truth $y^\dagger$ with $\rho_s m$ nonzero entries of unit magnitude, and compute $b = Ay^\dagger + \epsilon$. Solve the lower-level problem to tolerance $10^{-8}$ to obtain $y^*(x)$, then time a single oracle call at this solution.

Sweep $m \in \{10^2, 10^{2.5}, 10^3, 10^{3.5}, 10^4, 10^{4.5}, 10^5\}$ (roughly 7 points on a log scale), with fixed sparsity density $\rho_s = 5\%$. Repeat each timing 10 times and report the median (wall-clock time, in seconds).

### What the figure should show

A log-log plot of wall-clock time vs. $m$, with three curves:

- **Full dense:** slope 3 on the log-log plot (confirming $O(m^3)$), crossing the 1-second threshold somewhere around $m \approx 2{,}000$–$5{,}000$.
- **Support-reduced null:** slope 3 on the log-log plot but shifted downward by $\log_{10}(\rho_s^3) \approx -3.8$ — i.e., roughly 4 orders of magnitude faster at every dimension. Should remain below 1 second up to $m \approx 10^5$.
- **Support-reduced DA:** overlapping with the null curve up to the point where biactive coordinates add meaningful overhead, then slightly above it. The key visual is that the two support-reduced curves are nearly indistinguishable — the variational correction is essentially free.

Draw reference lines with slope 1, 2, and 3 in the background to make the scaling exponent readable directly from the figure.

### Secondary experiment: sparsity sensitivity

Run the support-reduced DA oracle at fixed $m = 10^4$ while sweeping the true sparsity density $\rho_s \in \{1\%, 2\%, 5\%, 10\%, 20\%, 50\%\}$. Plot oracle time vs. $\rho_s$ on a log-log scale. The expected slope is 3 (since $|\mathcal{S}| \propto \rho_s m$). This shows that the method's tractability depends on sparsity, not dimension — a meaningful message for practitioners who know their data is sparse.

### The message

The support-reduced oracle scales as $O(|\mathcal{S}|^3)$, not $O(m^3)$. At $\rho_s = 5\%$ and $m = 10^5$, the full dense system would require $10^{15}$ flops; the support-reduced oracle requires approximately $(5{,}000)^3 = 1.25 \times 10^{11}$ — a factor of $8{,}000$ reduction. The descent-aligned variational correction adds a negligible number of biactive coordinates to $\mathcal{S}$, and the overhead relative to the null oracle is invisible on the log-log plot. Correctness and efficiency are not in tension.

---

## Experiment 6 — Real-world validation

**Purpose:** confirm that the theoretical and synthetic gains from Experiments 1–4 translate into competitive end-to-end performance on standard benchmarks, and that the method is not prohibitively slower than the baselines at the scales where feature-wise regularization is practically relevant ($m \sim 10^4$–$10^6$).

This experiment is deliberately less controlled than the synthetic ones. There is no designed degeneracy, no known ground truth for support recovery (in the fully real-world setting), and no guarantee that biactive failure will be present. The only question is: does the method hold its own?

### Setting 1 — Semi-synthetic benchmark (RCV1)

**Purpose of this sub-experiment:** bridge the gap between synthetic experiments (where everything is known) and the fully real-world setting (where nothing is known). It uses a real feature matrix — preserving realistic correlation structure and dimensionality — but injects controlled signal so that support recovery can still be measured.

**Dataset construction:** start from the RCV1 binary text-classification corpus (Lewis et al., 2004). Select the top-$K = 500$ background features by variance (these form the realistic covariate structure). Then inject:
- $n_\text{easy} = 10$ always-active signal features with known nonzero weights.
- $P = 20$ distractor/hidden feature pairs drawn from the RCV1 corpus itself (i.e., real correlated text features, not synthetic), where the hidden features are initialized at the ℓ1 biactive boundary.

Labels are drawn from a logistic model with known signal strength and label noise $\sigma = 0.20$:

$$\text{logit}_i = \mathbf{a}_i^\top w^* + \sigma\,\varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0,1), \quad y_i = \text{sign}(\text{logit}_i)$$

**Why this is harder than the synthetic Experiment 4:** the distractor/hidden pairs are real correlated text features, not constructed to be collinear at a specific $\rho$. The correlation structure is whatever it is in the data — more realistic, less controlled.

**Metrics:** hidden-feature recall, support F1, held-out log-loss, all as a function of outer iteration. Sweep $\rho_\text{seeds} = 5$ independent train/val/test splits ($60\%/20\%/20\%$), run $n_\text{outer} = 60$ outer iterations per method.

**Baselines:** Sparse-HO (scalar) and Sparse-HO (wℓ1), same as Experiment 4.

**The message:** gradient starvation is not a synthetic artifact — it occurs on real correlated text features, and the descent-aligned oracle resolves it in a realistic setting.

### Setting 2 — Fully real-world classification benchmarks

**Datasets:** three standard LIBSVM binary text-classification corpora of increasing scale:

| Dataset | Features ($m$) | Samples | Notes |
|---|---|---|---|
| rcv1 | 47,236 | ~20,000 | Standard NLP benchmark |
| real-sim | 20,958 | ~72,000 | Mixed real/simulated news |
| news20 | 1,355,191 | ~19,000 | Very high-dimensional, sparse |

news20 is the stress test: $m > 10^6$ with very high natural sparsity. Any method that does not exploit support reduction will be impractical here.

**Split:** random train/val/test ($60\%/20\%/20\%$), repeated over $n_\text{seeds} = 3$ independent splits. Report mean ± std.

**Baselines:** three methods representing increasing levels of sophistication:

1. **Scalar-CV:** a single global penalty $C$ selected by 5-fold cross-validation (`LogisticRegressionCV`). Zero hyperparameter optimization cost beyond the CV grid. This is the practical default — the method must justify its added complexity.

2. **Sparse-HO (wℓ1):** per-feature penalties learned by implicit differentiation with gradient descent outer loop (Bertrand et al., 2022). Same expressivity as our method, different oracle.

3. **NTRBA-wℓ1 (ours):** same parameterization as (2), differentiated via the generalized support adjoint with descent-aligned biactive selection, updated with the nonsmooth trust-region.

**Metrics and what each reveals:**

- **Test-set F1:** the headline performance metric. The expectation is that NTRBA-wℓ1 matches or exceeds Sparse-HO (wℓ1), which in turn matches or exceeds Scalar-CV. The gain over Scalar-CV demonstrates the value of feature-wise regularization; the gain over Sparse-HO demonstrates the value of the correct oracle. If neither gain is visible on these datasets, the paper should acknowledge it honestly — the real-world experiments are not where the theoretical story is made, but they should not actively contradict it.

- **Model sparsity** (percentage of active features at convergence): feature-wise regularization should produce sparser models than scalar CV (which tends to either activate too many or too few features with a global threshold). NTRBA should achieve similar or higher sparsity than Sparse-HO with better F1 — recovering the right sparse structure.

- **Wall-clock time per outer iteration and total runtime:** the critical practical metric for news20. The support-reduced oracle (Experiment 5) established the scaling in isolation; here the cost is measured end-to-end including the inner solver. NTRBA should be within a small constant factor of Sparse-HO — both use support reduction. The comparison against Scalar-CV is less important here (CV with grid search is a very different computation).

**What this experiment cannot show:** support recovery (no ground truth), the failure mode of gradient starvation in isolation (the biactive structure of real data is unknown and uncontrolled), or the relative contribution of oracle vs. optimizer (that is Experiment 3's job). The real-world experiment's role is purely to establish practical viability.

**The message:** at all three scales, NTRBA-wℓ1 is competitive with Sparse-HO (wℓ1) in predictive performance, produces sparser models than scalar CV, and runs within a practical time budget. The overhead of the nonsmooth trust-region and the variational oracle is not prohibitive even at $m > 10^6$.

---

## What NOT to do

- **No grid search comparison for the regression task.** The bilevel vs. grid-search comparison conflates algorithmic efficiency with model class expressivity. It obscures both. Compare against Sparse-HO for the algorithmic story; compare against scalar CV for the expressivity story.
- **No experiments on non-degenerate data.** If the data is non-degenerate, all methods work. Every experiment should have a designed degeneracy that reveals the failure mode the paper resolves.
- **No scatter of metrics across many tables without a clear headline.** Each experiment should have one primary metric that tells the story, with supporting metrics as secondary evidence.

---

## Mapping: claims to experiments

The experiments are not a standard ML benchmark exercise. They are closer to **unit tests of theoretical claims**:

| Claim | Primary experiment |
|---|---|
| FBE preserves solution set; BE does not | Exp 1, Panel A |
| Biactive failure mode exists and is constructible | Exp 1, Panel B |
| Feature-wise regularization outperforms scalar | Exp 2 |
| Oracle drives generalization; optimizer drives stationarity | Exp 3 (ablation) |
| Sparse-HO starves hidden-feature gradients | Exp 4 |
| Support-reduced oracle is computationally tractable | Exp 5 |
| Method works at scale in practice | Exp 6 |

Every row should have a clear positive result. If any row has an ambiguous or weak result, either the claim needs to be weakened in the theory section or the experiment needs to be redesigned.

---

## Implementation Order

The key principle is to build from the innermost component outward, and to validate each layer before depending on it in the next. Shared components that multiple experiments reuse should be built once, correctly, before any experiment that needs them.

### Phase 1 — Build the engine (no experiments yet)

These three components are prerequisites for everything else. Nothing runs without them.

1. **Lower-level solver** (FISTA for weighted elastic-net). This is the foundation. Every experiment calls it at every outer iteration. Get it right, test it against known solutions (e.g., standard Lasso with sklearn), and make sure convergence to tolerance 10⁻⁸ is reliable.

2. **Null oracle** (primal support adjoint, reproducing Sparse-HO behavior). Simpler than the DA oracle — it only requires solving a linear system on the strict support. Build this first because it is the baseline compared against in Experiments 3 and 4. If the DA oracle is built first, there is nothing to compare it against.

3. **DA oracle** (descent-aligned biactive selection). Builds directly on the null oracle — it augments the working set before solving the same adjoint system. The incremental complexity over the null oracle is small: biactive detection + the sign-based selection rule. Having the null oracle already implemented makes it easy to verify that the two oracles agree on non-biactive instances.

### Phase 2 — Build the outer solvers

4. **NBA** (projected normalised subgradient). Simpler step rule, implement first.

5. **NTRBA** (nonsmooth trust-region). More complex. Implement second, and verify on the same instances where NBA already gives a known result — if NTRBA gives a worse answer than NBA on a convex instance, something is wrong.

### Phase 3 — Validate components (Experiments 1 and 5)

These two experiments only need the components already built and are the cheapest to run. Do them before investing in any large-scale sweep.

6. **Experiment 1, Panel A** (FBE vs. BE smoothing). Pure math, 1D or 2D instance. Verifiable by hand. If the level sets look wrong, the FBE implementation is broken.

7. **Experiment 1, Panel B** (micro-scale counterexample). Three features, diagonal design. The hypergradient values can be verified analytically. If the null oracle produces a zero on feature 2 and the DA oracle produces a nonzero, the oracle implementation is correct. This is the unit test for the entire oracle infrastructure — run it before touching anything else at scale.

8. **Experiment 5** (oracle scalability). Only needs the oracle, no outer loop. It is a timing experiment — sweep m and record wall-clock time for one oracle call. Quick to implement, and it gives an early result to show.

### Phase 4 — Synthetic experiments (Experiments 2, 3, 4)

Now the full bilevel pipeline is exercised. The order within this phase matters.

9. **Experiment 2** (feature-wise vs. scalar). Use non-degenerate data — no biactive features by design. This validates the full pipeline (solver + DA oracle + NBA/NTRBA) on a problem where the answer is roughly known in advance (the learned weight profile should assign small penalties to signal features and large penalties to noise). If something is wrong with the outer loop, it will show here without the complication of degeneracy.

10. **Degenerate data generator** (four-group construction). Build this as shared infrastructure before Experiments 3 and 4, which both use it. Getting the initialization right — hidden features placed exactly at the ℓ1 boundary — is delicate. Verify that the null oracle produces a zero gradient on hidden features before running anything.

11. **Experiment 3** (ablation). Run all four combinations on the degenerate data. This is an internal comparison — no external baselines, no external code. If NTRBA-DA is not the best combination here, something is wrong with the implementation before comparing against anyone else.

12. **Experiment 4** (SOTA comparison vs. Sparse-HO). Comes last among the synthetic experiments because it requires integrating Sparse-HO as an external baseline. By this point the proposed methods have been validated internally in Experiment 3, so any differences observed against Sparse-HO are attributable to the method, not to bugs.

### Phase 5 — Real-world (Experiment 6)

13. **Experiment 6** last. It adds dataset loading complexity and removes ground-truth access for support recovery. Everything should be debugged on synthetic data before running on real data, where unexpected failures are much harder to diagnose.

### Summary

```
Phase 1 │ Lower-level solver → Null oracle → DA oracle
Phase 2 │ NBA → NTRBA
Phase 3 │ Exp 1 (sanity checks) → Exp 5 (scalability)
Phase 4 │ Exp 2 (non-degenerate) → data generator → Exp 3 (ablation) → Exp 4 (SOTA)
Phase 5 │ Exp 6 (real-world)
```

The guiding rule at every step: **each experiment is a test of the components built in the previous phase**. If an experiment fails, the failure points to exactly which layer to investigate.
