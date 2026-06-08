# Matlab_PPO — Tutorial

> 中文版见 [TUTORIAL_zh.md](TUTORIAL_zh.md)

This document explains how the library works: the PPO algorithm, the specific implementation
details it adopts, the zero-dependency code architecture, the environment models, and how the
implementation was verified.

## Contents

1. [Design philosophy](#1-design-philosophy)
2. [The PPO algorithm](#2-the-ppo-algorithm)
3. [Implementation details](#3-implementation-details)
4. [Code architecture](#4-code-architecture)
5. [Environment models](#5-environment-models)
6. [Hyperparameter guide](#6-hyperparameter-guide)
7. [Custom environments](#7-custom-environments)
8. [How it was verified](#8-how-it-was-verified)
9. [Troubleshooting](#9-troubleshooting)

## 1. Design philosophy

The goal is a PPO implementation that is **correct, modern, and runs everywhere** without paid
toolboxes. Three decisions follow from that:

- **No Deep Learning Toolbox.** The multilayer perceptron, its backpropagation, and the Adam
  optimizer are written with plain matrix algebra. The exact same `.m` files run in MATLAB and in
  free GNU Octave.
- **Column-major data.** Observations and activations are stored as `(features × batch)`. A linear
  layer is `z = W*h + b` with `W` shaped `(out × in)`. This matches MATLAB's column-major memory
  and keeps the backprop formulas clean.
- **Explicit, testable math.** Because the gradients are hand-written, every one is checked by
  finite differences and against an independent NumPy reference (see §8).

## 2. The PPO algorithm

PPO is a policy-gradient method that improves a stochastic policy `π_θ(a|s)` while preventing
destructively large updates via a clipped objective.

### 2.1 Clipped surrogate objective

With the probability ratio `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` and advantage estimate `Â_t`:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\big[\min(r_t \hat{A}_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\,\hat{A}_t)\big]$$

We minimize `−L^CLIP`. The clip range `ε` defaults to 0.2. The gradient flows through the
unclipped term exactly when it is the smaller of the two (`s1 ≤ s2`); when the clipped term is
selected the policy receives no gradient — this is the trust-region effect.

### 2.2 Generalized Advantage Estimation

Advantages use GAE(λ), which trades bias against variance:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat{A}_t = \sum_{l\ge 0} (\gamma\lambda)^l \delta_{t+l}$$

computed by the backward recursion `Â_t = δ_t + γλ(1−done_t) Â_{t+1}`. Returns are `Â_t + V(s_t)`.
Defaults: `γ = 0.99`, `λ = 0.95`. See `utils/gae.m`.

### 2.3 Termination vs. truncation (a detail that matters)

Control tasks usually end for two different reasons: the episode **terminates** (the pole falls)
or it is **truncated** by a time limit. Bootstrapping is different in each case: at a true
terminal `V(s_{t+1}) = 0`, but at a time-limit truncation the episode would have continued, so we
bootstrap `V(s_{t+1})`. Ignoring this loses 20–40% on time-limited tasks.

This library handles it as follows: on a truncated step it adds `γ·V(next_obs)` to the reward and
marks `done = 1`, so the GAE recursion can use a single `done` mask uniformly.

### 2.4 Advantage normalization, entropy, value loss

- **Advantage normalization is per minibatch**, not over the whole batch:
  `Â ← (Â − mean(Â)) / (std(Â) + 1e-8)`.
- **Entropy bonus** `−c_ent·H(π)` encourages exploration (small/zero for continuous control,
  0.01 for CartPole).
- **Value loss** is plain MSE `mean((V − R)²)`. Value-function clipping is supported but **off by
  default** — there is no evidence it helps.

### 2.5 KL early stopping

After each epoch the mean approximate KL is checked with the low-variance **k3 estimator**
`E[(r − 1) − log r]`, and the update stops early if it exceeds `1.5·target_kl` (default `target_kl
= 0.02`). This is an adaptive trust region layered on top of clipping. We also log `clipfrac` and
the value function's **explained variance**.

## 3. Implementation details

These are the choices that separate a robust PPO from a fragile one. Each is cheap and high-impact.

| Area | Choice |
|---|---|
| Optimizer | Adam, `eps = 1e-5` (not 1e-8) |
| Learning rates | separate, **critic LR ≥ actor LR** (3e-4 / 1e-3); stronger value optimization |
| LR schedule | linear anneal to 0 |
| Grad clipping | global L2 norm 0.5 |
| Advantage norm | per minibatch |
| Init | orthogonal; hidden gain √2, **policy head 0.01**, value head 1.0 |
| Policy std | **state-independent** learnable log-std vector, init 0, clamp [−20, 2] |
| Obs norm | running mean/std, clip ±10 |
| Reward scaling | divide by running std of discounted returns, clip ±10 (optional) |
| KL stop | k3 estimator, `1.5·target_kl` |
| Architecture | width over depth; **optional LayerNorm** on hidden layers |

The less common choices adopted here are: running observation normalization on by default,
stronger critic optimization, the k3 KL estimator, and **optional pre-activation LayerNorm** in the
hidden layers — a lightweight architectural trick that improves stability for wider networks while
staying a few lines of hand-written forward/backward code. More complex variants were deliberately
left out as too involved for marginal gains on these control tasks.

## 4. Code architecture

### 4.1 `MLP` (core/MLP.m)

A feedforward network with explicit forward and manual backprop. Hidden layers are `tanh` with
optional **pre-activation LayerNorm**; the output layer is linear.

- `[y, cache] = forward(x)` returns the output and a cache of intermediate values.
- `[grad, dx] = backward(cache, dy)` returns gradients for every parameter (a struct mirroring the
  learnables) plus the input gradient.

Weights use orthogonal initialization (`utils/orthoInit.m`, via QR), with the output gain set per
role (0.01 for a policy head so initial actions are near zero; 1.0 for a value head).

### 4.2 Policies

- **Continuous** — a diagonal Gaussian. The MLP outputs the mean; the log standard deviation is a
  separate **state-independent** learnable vector (`PPO.logStd`). Log-probability and entropy use
  `utils/gaussianLogProb.m`. The analytic gradients w.r.t. the mean and log-std are derived in
  closed form and verified by finite differences.
- **Discrete** — a categorical distribution. The MLP outputs logits; `utils/softmaxCol.m` and
  `utils/sampleCategorical.m` handle the probabilities and sampling.

### 4.3 `Adam` (core/Adam.m)

A generic Adam that operates on any struct of parameter arrays (so the same optimizer updates the
MLP weights, the log-std vector, and the critic). Moment buffers mirror the parameter struct.

### 4.4 `RunningMeanStd` (utils/RunningMeanStd.m)

Online mean/variance via a numerically-stable batch update. Used to normalize observations every
step (and optionally to scale rewards by the running return std). Statistics are saved with the
model so deployment matches training.

### 4.5 `PPO` (core/PPO.m)

The agent ties it together:

1. **`collectRollout`** — run the policy for `nSteps`, normalizing observations and updating their
   statistics, storing transitions, and applying the truncation reward-bootstrap.
2. **`computeGAE`** — advantages and returns via `utils/gae.m`.
3. **`update`** — `nEpochs` passes over shuffled minibatches: per-minibatch advantage
   normalization, the clipped policy loss, value loss, entropy bonus, global-norm gradient
   clipping, Adam steps with annealed learning rates, and KL early stopping.

`train` loops this and logs return / losses / entropy / approx-KL / clipfrac / explained-variance.

## 5. Environment models

All environments subclass `Environment`. Discrete `step` takes a 1-based action index; continuous
`step` takes an `(actDim × 1)` vector. Time-limited tasks set `info.truncated = true` on the
final step.

### 5.1 CartPole (discrete)

State `[x, ẋ, θ, θ̇]`; action push left/right; reward +1 per step. Terminates when `|θ| > 12°` or
`|x| > 2.4`, truncates at 500 steps. Standard semi-implicit-Euler dynamics.

### 5.2 Pendulum swing-up (continuous)

The canonical continuous-control benchmark and the single-agent replacement for the old
multi-agent double pendulum. Observation `[cosθ, sinθ, θ̇]` (so the angle has no wraparound
discontinuity), action a torque in `[−1, 1]·maxTorque`, reward
`−(θ_wrapped² + 0.1·θ̇² + 0.001·u²)`. Time-limited (200 steps), no terminal state.

### 5.3 DC motor (continuous)

Angular-position regulation. Electrical/mechanical model
`θ̇ = ω`, `ω̇ = (Ki − bω)/J`, `İ = (V − Ri − Kω)/L`, integrated with Euler. Observation
`[angle_error, ω/ω_max, I/I_max]`, action a normalized voltage in `[−1, 1]`, reward penalizes
angle error, speed, and control effort.

### 5.4 AC induction motor — FOC (continuous, advanced)

Field-oriented control of a three-phase induction motor. The agent outputs the d/q-axis voltages
`[V_d, V_q]` while a speed-loop PI sets the q-current reference and the load torque steps over
time. The rotor-flux, stator-current, and mechanical equations are integrated at `dt = 1 ms`;
speed and currents are clamped to keep the explicit-Euler integration stable. Six-dimensional
observation; reward combines speed tracking, d/q current tracking, and control effort. This is the
hardest task and the best showcase for observation/reward normalization and LayerNorm.

## 6. Hyperparameter guide

`rlConfig(name)` returns a tuned preset; override any field afterwards.

- **clip ε** (0.1–0.3): smaller is more conservative; ACMotor uses 0.1.
- **γ** (0.95–0.99) / **λ** (0.9–0.97): horizon and bias/variance.
- **actorLR / criticLR**: keep the critic LR ≥ actor LR. Anneal with `lrAnneal`.
- **nSteps / nEpochs / miniBatch**: bigger rollouts and more epochs are more sample-efficient but
  cost compute; reduce epochs if `approx_kl` blows past `targetKL` early.
- **entCoef**: 0.01 for CartPole, ~0 for the continuous tasks (raise slightly if a policy collapses).
- **normObs** (keep on) / **normReward** (on for shaped continuous rewards).
- **hidden / useLayerNorm**: `[64 64]` for the simple tasks; `[256 256]` + LayerNorm for ACMotor.

Diagnostics to watch: rising **return**; **explained variance** climbing toward 1 (the critic is
learning); **approx_kl** staying near `targetKL`; **clipfrac** roughly 0.05–0.3.

## 7. Custom environments

```matlab
classdef MyEnv < Environment
    properties
        state
    end
    methods
        function obj = MyEnv()
            obj.obsDim = 4; obj.actDim = 2; obj.isDiscrete = false;
        end
        function obs = reset(obj)
            obj.state = zeros(4, 1);
            obs = obj.state;
        end
        function [obs, reward, done, info] = step(obj, action)
            a = max(min(action(:), 1), -1);
            obs = obj.state; reward = 0; done = false;
            info = struct('truncated', false);
        end
    end
end
```

Clamp continuous actions to `[−1, 1]`, then advance `obj.state` and compute the reward inside
`step`. Then `agent = PPO(MyEnv(), rlConfig());` and `agent.train();`. For discrete environments
set `isDiscrete = true`, make `actDim` the number of actions, and have `step` accept a 1-based
index.

## 8. How it was verified

Because the gradients are hand-written, correctness was established at three levels:

1. **Finite-difference gradient checks** of the MLP (with and without LayerNorm), the Gaussian and
   categorical policy gradients, the clipped surrogate, the value loss (incl. clipping), and the
   entropy term — all match analytic gradients to ~1e-10.
2. **NumPy oracle cross-check** — GAE, log-probabilities, ratios, entropies, and every loss term
   are reproduced in MATLAB to machine precision against an independent NumPy implementation.
3. **End-to-end training** — CartPole reaches the maximum return of 500; Pendulum, DC-motor, and
   AC-motor all learn. Save/load round-trips reproduce deterministic actions exactly.

## 9. Troubleshooting

- **Not learning / return flat** — confirm `normObs` is on; lower the learning rate; check that
  `explained_variance` is rising (if it stays ≤ 0 the critic is not fitting — raise `criticLR` or
  `nEpochs`).
- **`approx_kl` large, training unstable** — lower `actorLR`, lower `clip`, fewer `nEpochs`, or set
  a tighter `targetKL`.
- **Continuous policy collapses to a corner** — raise `entCoef` slightly or `logStdInit`.
- **NaNs on a custom env** — clamp states/actions; very stiff dynamics need smaller `dt` or
  clamping (see ACMotor).
