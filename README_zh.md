# 🤖 Matlab_PPO — 零依赖的控制系统 PPO 框架

[![Matlab_PPO](https://img.shields.io/badge/Matlab__PPO-v2.0.0-blueviolet)](https://github.com/AIResearcherHZ/Matlab_PPO)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2019b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Octave](https://img.shields.io/badge/GNU%20Octave-6%2B-orange.svg)](https://octave.org/)
[![No Toolbox](https://img.shields.io/badge/toolboxes-%E6%97%A0%E9%9C%80-success.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个面向控制问题的、**简洁现代**的**近端策略优化（PPO）** 实现：用**纯 MATLAB** 编写,
**零工具箱依赖**，同一份代码可以原封不动地运行在**免费的 GNU Octave** 上。神经网络、自动微分和
优化器都用基础矩阵运算从零实现，因此**无需 Deep Learning Toolbox**。

> 📚 算法原理、数学推导与每一处实现取舍见 [TUTORIAL_zh.md](TUTORIAL_zh.md)。
> English docs: [README.md](README.md) / [TUTORIAL.md](TUTORIAL.md)。

## ✨ 设计初衷

- **到处都能跑** —— 基础版 MATLAB 或免费 Octave 均可，无需付费工具箱、无需 GPU。
- **正确性有保障** —— 每个梯度都经过有限差分校验，并与 NumPy 参考实现逐元素对比到机器精度；
  CartPole 训练可达到满分回报 500。
- **默认强力** —— 采用一套经过验证的 PPO 技术，而非简陋的基线（见下表）。
- **代码可读** —— 小而清晰，能从头到尾读懂的代码库。

## 🚀 快速开始

```matlab
setup

agent = PPO(CartPoleEnv(), rlConfig('CartPole'));
agent.train();
res = agent.evaluate(20);
fprintf('平均回报 %.1f\n', res.meanReturn);
```

整套 API 就这些：`PPO(env, cfg)`、`train`、`evaluate`、`predict`、`save`、`load`。

## 🧠 核心技术

| 技术 | 作用 |
|---|---|
| **GAE(λ)** + 正确区分**截断与终止** | 超时步自举价值、真终止置零 |
| **按小批量归一化优势** | 对奖励尺度不敏感 |
| **从零实现的 Adam**（`eps=1e-5`）+ **更强的 critic 学习率** | 稳健优化 |
| **学习率线性退火** + **全局梯度范数裁剪** | 后期训练稳定 |
| **KL 提前停止**（k3 估计量） | 在裁剪之上再加一层自适应信赖域 |
| **正交初始化**（策略头增益 0.01） | 影响最大的初始化细节 |
| **运行时观测归一化**（可选奖励缩放） | 连续控制中收益最大的一项 |
| **状态无关 log-std** 高斯策略 | 标准且稳定的参数化 |
| **可选 LayerNorm** 隐藏层 | 面向更宽网络的轻量稳定性技巧 |

完整说明见 [TUTORIAL_zh.md](TUTORIAL_zh.md)。

## 🎮 环境

| 环境 | 动作空间 | 任务 |
|---|---|---|
| `CartPoleEnv` | 离散（2） | 平衡杆 —— 经典控制 |
| `PendulumEnv` | 连续（1） | 摆起并稳定（单智能体） |
| `DCMotorEnv` | 连续（1） | 驱动转子到达目标角度 |
| `ACMotorEnv` | 连续（2） | 交流感应电机磁场定向控制（进阶） |

## 📈 参考结果

使用内置预设（`rlConfig(name)`），在 Octave 下训练并评估：

| 环境 | 随机策略 | 训练后（20 回合评估） |
|---|---|---|
| CartPole | ~20 | **500.0 / 500**（已解决） |
| Pendulum | ~ −1300 | ~ −300，随迭代继续提升 |
| DCMotor  | ~ −800 | ~ −400 |

CartPole 达到满分；连续控制任务单调提升。增大 `cfg.numIterations` / `cfg.nSteps` 可获得更强的
连续控制策略。

## 📦 使用

### 运行示例

```matlab
quickstart
train_cartpole
test_cartpole
```

`train_*` 会把模型保存到 `logs/`；对应的 `test_*` 加载并评估（设 `doRender = true` 可观看）。
`pendulum`、`dcmotor`、`acmotor` 也有同样的 `train_*` / `test_*` 脚本对。

### 配置

从预设开始，覆盖任意字段：

```matlab
cfg = rlConfig('Pendulum');
cfg.numIterations = 200;
cfg.hidden = [128 128];
cfg.useLayerNorm = true;
cfg.normReward = true;
agent = PPO(PendulumEnv(), cfg);
agent.train();
```

主要字段：`gamma, lambda, clip, entCoef, vfCoef, maxGradNorm, actorLR, criticLR, lrAnneal,
nSteps, nEpochs, miniBatch, targetKL, hidden, useLayerNorm, logStdInit, normObs, normReward,
numIterations, seed`。说明与默认值见 `config/rlConfig.m`。

### 保存 / 加载与部署

```matlab
agent.save('model.mat');
agent.load('model.mat');
action = agent.predict(obs, true);
```

`predict(obs, true)` 对原始观测返回确定性动作。

### 添加自定义环境

继承 `Environment`，设置 `obsDim` / `actDim` / `isDiscrete`，并实现 `reset` 与 `step`：

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

`reset` 返回 `(obsDim × 1)` 的观测。`step` 中，连续 `action` 为 `(actDim × 1)`，离散为 1 起始的
索引；到达时间上限时设 `info.truncated = true`。

## 📁 目录结构

```
Matlab_PPO/
├── core/          PPO.m, MLP.m, Adam.m
├── environments/  Environment.m + CartPole / Pendulum / DCMotor / ACMotor
├── config/        rlConfig.m（各环境预设）
├── utils/         GAE、归一化、高斯/softmax 数学、日志、初始化
├── examples/      quickstart 与 train_/test_ 脚本
└── setup.m        将源码目录加入路径
```

## 📋 环境要求

- MATLAB **R2019b+** 或 GNU **Octave 6+**。除此之外**别无所需** —— 无需 Deep Learning / Parallel 工具箱。

## 📖 引用

```bibtex
@misc{matlab_ppo,
  author = {Haozheng Xie},
  title  = {Matlab_PPO: A Zero-Dependency PPO Framework for Control Systems},
  year   = {2026},
  url    = {https://github.com/AIResearcherHZ/Matlab_PPO}
}
```

## 📄 许可证

MIT —— 详见 [LICENSE](LICENSE)。
