# 🤖 Matlab PPO 强化学习框架

[![Matlab_PPO](https://img.shields.io/badge/Matlab_PPO-v1.0.0-blueviolet)](https://github.com/AIResearcherHZ/Matlab_PPO)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2019b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Deep Learning Toolbox](https://img.shields.io/badge/Deep%20Learning%20Toolbox-Required-green.svg)](https://www.mathworks.com/products/deep-learning.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个基于MATLAB的强化学习框架，实现了近端策略优化(PPO)算法及其多智能体扩展(MAPPO)，支持GPU加速和并行计算，适用于控制系统研究和工程应用。

> 📚 有关算法原理、实现细节和高级功能的详细说明，请参阅 [TUTORIAL_zh.md](TUTORIAL_zh.md)

## ✨ 项目理念

本项目基于以下核心理念：

- 🎯 为控制系统研究提供高效且用户友好的强化学习框架
- 🔄 将先进的PPO算法应用于实际工程控制问题
- 🤝 推广多智能体系统在工业控制中的应用
- 📊 提供直观的性能评估和可视化工具

## 🚀 技术栈

### 核心框架
- **MATLAB R2019b+**: 主要开发环境
- **Deep Learning Toolbox**: 神经网络构建
- **Parallel Computing Toolbox**: 并行数据收集
- **GPU Computing**: CUDA加速支持

### 算法实现
- **PPO**: 基于裁剪的策略优化
- **MAPPO**: 多智能体协同学习
- **Actor-Critic**: 双网络架构
- **GAE**: 广义优势估计

### 环境模型
- **经典控制**: 倒立摆系统
- **电机控制**: 直流/交流电机系统
- **多智能体**: 双摆系统

## 🌟 主要特性

### PPO (近端策略优化)
- 基于策略梯度的强化学习算法
- 使用裁剪目标函数限制策略更新幅度
- 支持连续和离散动作空间
- 提供稳定高效的训练过程

### MAPPO (多智能体PPO)
- PPO的多智能体扩展
- 采用"集中训练，分散执行"架构
- 为需要智能体协作的任务提供解决方案

## 🎮 支持的环境

框架包含四个控制场景：

1. **倒立摆**
   - 经典控制问题
   - 通过左右移动小车平衡垂直杆
   - 离散动作空间示例

2. **直流电机控制**
   - 调节电机速度以跟踪目标速度
   - 连续动作空间示例
   - 包含电气和机械系统动态

3. **交流电机FOC控制**
   - 实现交流感应电机的磁场定向控制
   - 高级工业控制示例
   - 涉及复杂坐标变换和动态

4. **双摆系统**
   - 多智能体协同控制示例
   - 展示MAPPO算法优势
   - 需要协调控制的两个连接摆

## 🚀 快速入门

### 安装要求
- MATLAB R2019b 或更高版本
- Deep Learning Toolbox
- Parallel Computing Toolbox (可选，用于并行数据收集)
- GPU 支持 (可选，但推荐用于加速训练)

### 安装步骤

1. **下载项目**
```bash
git clone https://github.com/AIResearcherHZ/Matlab_PPO.git
cd Matlab_PPO
```

### 使用步骤

#### 1. 运行示例
```matlab
% 运行倒立摆训练示例
train_cartpole

% 运行直流电机控制示例
train_dcmotor

% 运行交流电机FOC控制示例
train_acmotor

% 运行双摆系统MAPPO示例
train_doublependulum
```

#### 2. 测试训练结果
```matlab
% 测试倒立摆
test_cartpole

% 测试直流电机控制
test_dcmotor

% 测试交流电机FOC控制
test_acmotor

% 测试双摆系统
test_doublependulum
```

#### 3. 自定义配置

```matlab
% 创建和修改配置
config = PPOConfig();
config.envName = 'CartPoleEnv';  % 环境名称
config.gamma = 0.99;             % 折扣因子
config.epsilon = 0.2;            % 裁剪参数
config.actorLearningRate = 3e-4; % 策略网络学习率
config.useGPU = true;            % 启用GPU加速
config.numIterations = 100;      % 训练迭代次数

% 创建代理并训练
agent = PPOAgent(config);
agent.train(config.numIterations);
```

#### 4. 保存和加载模型

```matlab
% 保存训练好的模型
agent.saveModel('my_trained_model.mat');

% 加载模型
agent.loadModel('my_trained_model.mat');
```

## 📁 目录结构

```
Matlab_PPO/
├── core/               # 核心算法实现
├── environments/       # 环境实现
├── config/             # 配置文件
├── utils/              # 工具函数
├── examples/           # 示例脚本
└── logs/               # 日志和模型保存目录
```

## 📚 文档

- [TUTORIAL.md](TUTORIAL.md) - 详细的算法教程和实现细节
- API文档可通过MATLAB的`help`命令获取，例如：
  ```matlab
  help PPOAgent
  help Environment
  help DCMotorEnv
  ```

## 🔮 未来规划

1. **算法增强**
   - 实现更多先进的PPO变体
   - 添加其他流行的强化学习算法
   - 优化多智能体训练策略

2. **环境扩展**
   - 增加更多工业控制场景
   - 支持自定义环境接口
   - 添加仿真环境可视化

3. **性能优化**
   - 进一步提升GPU利用率
   - 优化并行训练机制
   - 改进数据收集效率

## 🤝 贡献

欢迎通过Issue和Pull Request提交改进建议和贡献代码。我们特别欢迎以下方面的贡献：

- 🐛 Bug修复和问题报告
- ✨ 新功能和改进建议
- 📝 文档完善和翻译
- 🎯 新的应用场景示例

## 📖 引用

如果您在研究中使用了本框架，请引用以下文献：

```bibtex
@misc{matlab_ppo,
  author = {},
  title = {Matlab PPO: A Reinforcement Learning Framework for Control Systems},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/X-Embodied/Matlab_PPO}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

感谢所有为本项目做出贡献的研究者和开发者！
