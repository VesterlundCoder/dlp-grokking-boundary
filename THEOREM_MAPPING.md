# THEOREM_MAPPING.md — Mapping Takhanov et al. to Our Experiments

## Reference

Takhanov, R., Tezekbayev, M., Pak, A., Bolatov, A., Kadyrsizova, Z., Assylbekov, Z. (2024).
"Intractability of Learning the Discrete Logarithm with Gradient-Based Methods."
Proceedings of Machine Learning Research, vol. 222.

## Theorem Summary

### Core Result

The theorem establishes that gradient-based methods cannot efficiently learn
the parity bit of the discrete logarithm in cyclic groups of large prime order.
Specifically, the gradient of the loss with respect to the model parameters
concentrates (becomes exponentially small) as the group order grows.

### Key Assumptions

| # | Assumption | Details |
|---|-----------|---------|
| A1 | Group structure | Cyclic group of prime order q |
| A2 | Concept class | Parity bit: f(g, h) = bit_i(x) where g^x = h |
| A3 | Data distribution | Uniform over group elements (or specific distribution) |
| A4 | Loss function | Bounded, differentiable (e.g., cross-entropy) |
| A5 | Model class | Neural networks with bounded parameters |
| A6 | Optimization | Gradient-based (SGD, Adam, etc.) |

### Bounded Quantity

The theorem bounds a gradient concentration quantity. The exact form depends
on the specific theorem statement, but the key quantity is:

**Gradient signal:** ||E_{(g,h)~D}[∇_θ L(θ; g, h)]||

For prime-order groups, this quantity decays exponentially in log(q),
meaning the expected gradient direction becomes increasingly uninformative
as the group grows.

### Asymptotic Dependence

The bound is of the form:

||E[∇L]|| ≤ C · exp(-α · log(q)) = C · q^(-α)

for some constants C, α > 0 depending on the model class and loss function.

This means the number of gradient steps required to learn scales as
q^(α) — i.e., polynomially in q, which is exponential in the bit length b = log₂(q).

## Mapping to Our Experiments

### Which Experiments Satisfy Which Assumptions

| Experiment | A1 (prime q) | A2 (parity bit) | A3 (uniform) | A4 (CE loss) | A5 (NN) | A6 (gradient) |
|-----------|-------------|----------------|-------------|-------------|---------|--------------|
| Stage 0 (MLP, natural dist.) | ✓ (q prime) | ✓ (low_bit=0) | ✗ (natural) | ✓ | ✓ (MLP) | ✓ (Adam) |
| Stage 0b (MLP, uniform) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Stage 1 (Transformer, 2k ep) | ✓ | ✓ | ✗ | ✓ | ✓ (Transformer) | ✓ (Adam) |
| Stage 2 (Transformer, 100k ep) | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ (AdamW+WD) |
| Stage 3 (2×2, 200k ep) | ✓ | varies | ✗ | ✓ | ✓ | ✓ (AdamW+WD) |
| **New: Matched isomorphic** | ✓ | ✗ (full log) | ✓ (by design) | ✓ | ✓ | ✓ |

### Critical Distinctions

1. **Stage 0/0b are the closest to the theorem setup.** They use MLPs (simple
   model class), parity bit targets, and Stage 0b uses uniform distribution.
   The 16-20 bit transition we observe is a finite-scale empirical counterpart
   of the asymptotic hardness prediction.

2. **Stages 1-2 are NOT a test of the theorem.** They use Transformers (different
   model class), non-uniform distribution, and — critically — they use the
   SAME task (DLP parity bit) with different optimization. Stage 2 is NOT
   modular addition (see AUDIT_REPORT.md).

3. **Stage 3 uses full-log targets** for some runs, which is a different concept
   class than the theorem's parity bit. The "hidden" vs "visible" base distinction
   is also not in the theorem.

4. **The new matched isomorphic experiment** tests full-log DLP (not parity bit),
   so it does not directly test the theorem. However, it tests the broader
   question of representation-dependent learnability, which is related but distinct.

### What We Can and Cannot Claim

**CAN claim:**
- "Our MLP experiments (Stages 0, 0b) show a finite-scale transition consistent
  with the gradient-concentration hardness mechanism predicted by Takhanov et al."
- "The transition occurs between b=16 and b=20 for our MLP architecture and
  training budget."

**CANNOT claim:**
- "The theorem predicts a threshold at b=16-20." (The theorem is asymptotic.)
- "Prime-order DLP is not grokkable." (Paper 1 grokked q=113 prime-order subgroup.)
- "Our experiments violate or confirm the theorem." (Finite-scale, different setup.)

### Empirical Gradient Concentration (Planned)

To connect to the theorem more directly, we plan to measure:

1. **||E[g]||** — Mean gradient norm (the theorem's bounded quantity)
2. **E[||g||²]** — Second moment of gradient
3. **Var(g)** — Gradient variance
4. **GSNR = ||E[g]||² / E[||g - E[g]||²]** — Gradient signal-to-noise ratio
5. **Pairwise gradient cosine similarity**
6. **Gradient effective rank**

These will be measured at:
- Initialization
- Early fitting
- Memorization
- Pre-grokking (if applicable)
- During transition
- Post-generalization

If the theorem's mechanism is at work, we expect GSNR to decay approximately
exponentially with bit length: log(GSNR) = α - β·b.

### Important Caveats

- The theorem's exact gradient quantity may not match our empirical measurement.
  We measure gradients of the training loss, while the theorem may bound a
  different quantity (e.g., population gradient, or gradient with respect to
  a specific concept class indicator).
- Our experiments use finite q (7-26 bits), while the theorem is asymptotic.
  Finite-scale transitions are consistent with but not proof of the theorem.
- The theorem assumes prime-order groups. Our matched isomorphic experiment
  uses prime q=113, satisfying this assumption, but tests full-log DLP rather
  than parity bit.
