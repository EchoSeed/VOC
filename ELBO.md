# Formal Proof: Variational Inference and the Evidence Lower Bound (ELBO)

## Problem Setup

Given:
- Observed data **x**
- Latent variables **z**
- True posterior p(z|x) (intractable)
- Approximate posterior q(z|x) (variational distribution)
- Prior p(z)
- Likelihood p(x|z)

**Objective:** Maximize the log marginal likelihood log p(x) by finding the best approximation q(z|x) to the true posterior p(z|x).

---

## Main Theorem

The log marginal likelihood can be decomposed as:

```
log p(x) = 𝔼_q[log p(x,z) - log q(z|x)] + KL(q(z|x) || p(z|x))
         = ELBO(q) + KL(q(z|x) || p(z|x))
```

Where:
- **ELBO(q)** = 𝔼_q[log p(x,z) - log q(z|x)] is the Evidence Lower Bound
- **KL(q(z|x) || p(z|x))** ≥ 0 is the KL divergence (always non-negative)

---

## Proof

### Step 1: Start with the KL divergence definition

```
KL(q(z|x) || p(z|x)) = 𝔼_q[log q(z|x) - log p(z|x)]
```

### Step 2: Apply Bayes' rule to the true posterior

```
p(z|x) = p(x,z) / p(x)

log p(z|x) = log p(x,z) - log p(x)
```

### Step 3: Substitute into the KL divergence

```
KL(q(z|x) || p(z|x)) = 𝔼_q[log q(z|x) - log p(x,z) + log p(x)]
```

### Step 4: Separate the expectation

```
KL(q(z|x) || p(z|x)) = 𝔼_q[log q(z|x) - log p(x,z)] + 𝔼_q[log p(x)]
```

### Step 5: Note that log p(x) doesn't depend on z

```
KL(q(z|x) || p(z|x)) = 𝔼_q[log q(z|x) - log p(x,z)] + log p(x)
```

### Step 6: Rearrange to isolate log p(x)

```
log p(x) = 𝔼_q[log p(x,z) - log q(z|x)] + KL(q(z|x) || p(z|x))
```

### Step 7: Define the ELBO

Let:
```
ELBO(q) := 𝔼_q[log p(x,z) - log q(z|x)]
```

Then:
```
log p(x) = ELBO(q) + KL(q(z|x) || p(z|x))
```

### Step 8: Since KL ≥ 0, we have:

```
log p(x) ≥ ELBO(q)
```

This proves the ELBO is a lower bound on the log evidence. **QED**

---

## Alternative ELBO Formulation

The ELBO can be rewritten as:

```
ELBO(q) = 𝔼_q[log p(x,z) - log q(z|x)]
        = 𝔼_q[log p(x|z) + log p(z) - log q(z|x)]
        = 𝔼_q[log p(x|z)] - KL(q(z|x) || p(z))
```

### Proof of equivalence:

```
ELBO(q) = 𝔼_q[log p(x|z) + log p(z) - log q(z|x)]
        = 𝔼_q[log p(x|z)] + 𝔼_q[log p(z) - log q(z|x)]
        = 𝔼_q[log p(x|z)] - 𝔼_q[log q(z|x) - log p(z)]
        = 𝔼_q[log p(x|z)] - KL(q(z|x) || p(z))
```

---

## Interpretation

**Reconstruction term:** 𝔼_q[log p(x|z)]
- How well can we reconstruct x from sampled latent z?
- Maximizing this improves generation quality

**Regularization term:** -KL(q(z|x) || p(z))
- How close is our approximate posterior to the prior?
- Minimizing this KL divergence keeps the latent space structured

---

## Optimization Strategy

To maximize log p(x), we equivalently maximize ELBO(q):

```
max_q ELBO(q) = max_q [𝔼_q[log p(x|z)] - KL(q(z|x) || p(z))]
```

Since KL(q || p(z|x)) ≥ 0, when ELBO is maximized:
- q(z|x) approaches p(z|x)
- log p(x) is effectively maximized

---

## Application to VAEs

In Variational Autoencoders:

1. **Encoder:** Neural network parametrizes q_φ(z|x)
2. **Decoder:** Neural network parametrizes p_θ(x|z)
3. **Prior:** Typically p(z) = 𝒩(0, I)

**Loss function:**
```
ℒ(θ, φ; x) = -ELBO(q_φ)
           = -𝔼_q_φ[log p_θ(x|z)] + KL(q_φ(z|x) || p(z))
```

Minimizing this loss = Maximizing the ELBO = Maximizing log p(x)

---

## Key Properties

1. **Non-negativity of KL:** KL(q || p) ≥ 0, with equality iff q = p almost everywhere

2. **Tightness:** ELBO = log p(x) when q(z|x) = p(z|x) exactly

3. **Tractability:** ELBO involves only q, p(x|z), and p(z) - all tractable

4. **Gradients:** Can use reparametrization trick to compute ∇_φ ELBO

---

## Conclusion

This decomposition shows that:
- Maximizing ELBO is equivalent to minimizing KL(q || p(z|x))
- The gap between ELBO and log p(x) measures approximation quality
- VAEs optimize a principled lower bound on the true objective

