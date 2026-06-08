# 🤖 Matlab_PPO — Zero-Dependency PPO for Control

[![Matlab_PPO](https://img.shields.io/badge/Matlab__PPO-v2.0.0-blueviolet)](https://github.com/AIResearcherHZ/Matlab_PPO)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2019b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Octave](https://img.shields.io/badge/GNU%20Octave-6%2B-orange.svg)](https://octave.org/)
[![No Toolbox](https://img.shields.io/badge/toolboxes-none%20required-success.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, modern implementation of **Proximal Policy Optimization (PPO)** for control problems,
written in **pure MATLAB** with **zero toolbox dependencies** — the same code runs unmodified in
**GNU Octave**. Networks, automatic differentiation, and the optimizer are all implemented from
scratch with plain matrix operations, so you do not need the Deep Learning Toolbox.

> 📚 Algorithm details, the math, and every implementation choice are in
> [TUTORIAL.md](TUTORIAL.md). 中文文档见 [README_zh.md](README_zh.md) 和 [TUTORIAL_zh.md](TUTORIAL_zh.md)。

## ✨ Why this exists

- **Runs anywhere** — base MATLAB or free Octave, no paid toolboxes, no GPU required.
- **Correct by construction** — every gradient is finite-difference checked and cross-validated
  against a NumPy reference to machine precision; CartPole trains to the maximum return of 500.
- **Strong defaults** — a well-tested set of PPO techniques rather than a bare-bones baseline (see below).
- **Readable** — a small, well-factored codebase you can actually follow end to end.

## 🚀 Quick start

```matlab
setup

agent = PPO(CartPoleEnv(), rlConfig('CartPole'));
agent.train();
res = agent.evaluate(20);
fprintf('mean return %.1f\n', res.meanReturn);
```

That is the entire API surface: `PPO(env, cfg)`, `train`, `evaluate`, `predict`, `save`, `load`.

## 🧠 What's inside

| Technique | What it does |
|---|---|
| **GAE(λ)** + correct **truncation vs termination** | bootstraps timeouts, zeroes true terminals |
| **Per-minibatch advantage normalization** | reward-scale invariance |
| **From-scratch Adam** (`eps=1e-5`) + **separate, stronger critic LR** | robust optimization |
| **Linear LR annealing** + **global gradient-norm clipping** | stable late-stage training |
| **KL early-stopping** (k3 estimator) | adaptive trust region on top of clipping |
| **Orthogonal init** (policy-head gain 0.01) | the single most impactful init detail |
| **Running observation normalization** (+ optional reward scaling) | biggest sample-efficiency lever for control |
| **State-independent log-std** Gaussian policy | a standard, stable parameterization |
| **Optional LayerNorm** hidden layers | lightweight stability for wider networks |

Full details: [TUTORIAL.md](TUTORIAL.md).

## 🎮 Environments

| Env | Action space | Task |
|---|---|---|
| `CartPoleEnv` | discrete (2) | balance the pole — classic control |
| `PendulumEnv` | continuous (1) | swing-up and hold (single-agent) |
| `DCMotorEnv` | continuous (1) | drive the rotor to a target angle |
| `ACMotorEnv` | continuous (2) | AC induction motor field-oriented control (advanced) |

## 📈 Reference results

Using the bundled presets (`rlConfig(name)`), trained and evaluated under Octave:

| Env | Random | Trained (20-episode eval) |
|---|---|---|
| CartPole | ~20 | **500.0 / 500** (solved) |
| Pendulum | ~ −1300 | ~ −300 and improving with more iterations |
| DCMotor  | ~ −800 | ~ −400 |

CartPole reaches the maximum return; the continuous tasks improve monotonically. Increase
`cfg.numIterations` / `cfg.nSteps` for stronger continuous-control policies.

## 📦 Usage

### Run the examples

```matlab
quickstart
train_cartpole
test_cartpole
```

`train_*` saves a model under `logs/`; the matching `test_*` loads it and evaluates (set
`doRender = true` to watch). The same `train_*` / `test_*` pairs exist for `pendulum`, `dcmotor`,
and `acmotor`.

### Configure

Start from a preset and override any field:

```matlab
cfg = rlConfig('Pendulum');
cfg.numIterations = 200;
cfg.hidden = [128 128];
cfg.useLayerNorm = true;
cfg.normReward = true;
agent = PPO(PendulumEnv(), cfg);
agent.train();
```

Key fields: `gamma, lambda, clip, entCoef, vfCoef, maxGradNorm, actorLR, criticLR, lrAnneal,
nSteps, nEpochs, miniBatch, targetKL, hidden, useLayerNorm, logStdInit, normObs, normReward,
numIterations, seed`. See `config/rlConfig.m` for documentation and defaults.

### Save / load and deploy

```matlab
agent.save('model.mat');
agent.load('model.mat');
action = agent.predict(obs, true);
```

`predict(obs, true)` returns a deterministic action for a raw observation.

### Add your own environment

Subclass `Environment`, set `obsDim` / `actDim` / `isDiscrete`, and implement `reset` and `step`:

```matlab
classdef MyEnv < Environment
    methods
        function obj = MyEnv()
            obj.obsDim = 3; obj.actDim = 1; obj.isDiscrete = false;
        end
        function obs = reset(obj)
            obs = zeros(obj.obsDim, 1);
        end
        function [obs, reward, done, info] = step(obj, action)
            obs = zeros(obj.obsDim, 1); reward = 0; done = false;
            info = struct('truncated', false);
        end
    end
end
```

`reset` returns an `(obsDim × 1)` observation. In `step`, a continuous `action` is `(actDim × 1)`
and a discrete one is a 1-based index; set `info.truncated = true` on a time-limit step.

## 📁 Layout

```
Matlab_PPO/
├── core/          PPO.m, MLP.m, Adam.m
├── environments/  Environment.m + CartPole / Pendulum / DCMotor / ACMotor
├── config/        rlConfig.m (per-environment presets)
├── utils/         gae, normalization, Gaussian/softmax math, logger, init
├── examples/      quickstart + train_/test_ scripts
└── setup.m        adds the source folders to the path
```

## 📋 Requirements

- MATLAB **R2019b+** *or* GNU **Octave 6+**. Nothing else — no Deep Learning / Parallel toolboxes.

## 📖 Citation

```bibtex
@misc{matlab_ppo,
  author = {Haozheng Xie},
  title  = {Matlab_PPO: A Zero-Dependency PPO Framework for Control Systems},
  year   = {2026},
  url    = {https://github.com/AIResearcherHZ/Matlab_PPO}
}
```

## 📄 License

MIT — see [LICENSE](LICENSE).
