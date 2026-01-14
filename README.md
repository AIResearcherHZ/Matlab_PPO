# 🤖 Matlab PPO Reinforcement Learning Framework

[![Matlab_PPO](https://img.shields.io/badge/Matlab_PPO-v1.0.0-blueviolet)](https://github.com/AIResearcherHZ/Matlab_PPO)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2019b%2B-blue.svg)](https://www.mathworks.com/products/matlab.html)
[![Deep Learning Toolbox](https://img.shields.io/badge/Deep%20Learning%20Toolbox-Required-green.svg)](https://www.mathworks.com/products/deep-learning.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A MATLAB-based reinforcement learning framework featuring Proximal Policy Optimization (PPO) algorithm and its multi-agent extension (MAPPO), with GPU acceleration and parallel computing support, suitable for control system research and engineering applications.

> 📚 For detailed algorithm principles, implementation details, and advanced features, please refer to [TUTORIAL.md](TUTORIAL.md)

## ✨ Inspiration

This project was inspired by several core principles:

- 🎯 Provide an efficient and user-friendly reinforcement learning framework for control system research
- 🔄 Apply advanced PPO algorithms to practical engineering control problems
- 🤝 Promote multi-agent system applications in industrial control
- 📊 Offer intuitive performance evaluation and visualization tools

## 🚀 Tech Stack

### Core Framework
- **MATLAB R2019b+**: Primary development environment
- **Deep Learning Toolbox**: Neural network construction
- **Parallel Computing Toolbox**: Parallel data collection
- **GPU Computing**: CUDA acceleration support

### Algorithm Implementation
- **PPO**: Clip-based policy optimization
- **MAPPO**: Multi-agent collaborative learning
- **Actor-Critic**: Dual network architecture
- **GAE**: Generalized Advantage Estimation

### Environment Models
- **Classic Control**: Cart-pole system
- **Motor Control**: DC/AC motor systems
- **Multi-agent**: Double pendulum system

## 🌟 Key Features

### PPO (Proximal Policy Optimization)
- Policy gradient-based reinforcement learning algorithm
- Uses clipped objective function to limit policy update magnitude
- Supports continuous and discrete action spaces
- Provides stable and efficient training process

### MAPPO (Multi-Agent PPO)
- Multi-agent extension of PPO
- Adopts "Centralized Training with Decentralized Execution" architecture
- Provides solutions for tasks requiring agent cooperation

## 🎮 Supported Environments

The framework includes four control scenarios:

1. **CartPole**
   - Classic control problem
   - Balance a vertical pole by moving the cart left or right
   - Discrete action space example

2. **DC Motor Control**
   - Adjust motor speed to track target velocity
   - Continuous action space example
   - Includes electrical and mechanical system dynamics

3. **AC Motor FOC Control**
   - Implement Field-Oriented Control for AC induction motors
   - Advanced industrial control example
   - Involves complex coordinate transformations and dynamics

4. **Double Pendulum System**
   - Multi-agent cooperative control example
   - Demonstrates MAPPO algorithm advantages
   - Two connected pendulums requiring coordinated control

## 🚀 Quick Start

### Requirements
- MATLAB R2019b or higher
- Deep Learning Toolbox
- Parallel Computing Toolbox (optional, for parallel data collection)
- GPU support (optional but recommended for training acceleration)

### Installation Steps

1. **Download the Project**
```bash
git clone https://github.com/AIResearcherHZ/Matlab_PPO.git
cd Matlab_PPO
```

### Usage Steps

#### 1. Run Examples
```matlab
% Run CartPole training example
train_cartpole

% Run DC motor control example
train_dcmotor

% Run AC motor FOC control example
train_acmotor

% Run double pendulum MAPPO example
train_doublependulum
```

#### 2. Test Training Results
```matlab
% Test CartPole
test_cartpole

% Test DC motor control
test_dcmotor

% Test AC motor FOC control
test_acmotor

% Test double pendulum system
test_doublependulum
```

#### 3. Custom Configuration

```matlab
% Create and modify configuration
config = PPOConfig();
config.envName = 'CartPoleEnv';  % Environment name
config.gamma = 0.99;             % Discount factor
config.epsilon = 0.2;            % Clipping parameter
config.actorLearningRate = 3e-4; % Policy network learning rate
config.useGPU = true;            % Enable GPU acceleration
config.numIterations = 100;      % Training iterations

% Create agent and train
agent = PPOAgent(config);
agent.train(config.numIterations);
```

#### 4. Save and Load Models

```matlab
% Save trained model
agent.saveModel('my_trained_model.mat');

% Load model
agent.loadModel('my_trained_model.mat');
```

## 📁 Directory Structure

```
Matlab_PPO/
├── core/               # Core algorithm implementation
├── environments/       # Environment implementation
├── config/             # Configuration files
├── utils/              # Utility functions
├── examples/           # Example scripts
└── logs/               # Logs and model save directory
```

## 📚 Documentation

- [TUTORIAL.md](TUTORIAL.md) - Detailed algorithm tutorial and implementation details
- API documentation available through MATLAB's `help` command, e.g.:
  ```matlab
  help PPOAgent
  help Environment
  help DCMotorEnv
  ```

## 🔮 Future Plans

1. **Algorithm Enhancement**
   - Implement more advanced PPO variants
   - Add other popular reinforcement learning algorithms
   - Optimize multi-agent training strategies

2. **Environment Extension**
   - Add more industrial control scenarios
   - Support custom environment interfaces
   - Add simulation environment visualization

3. **Performance Optimization**
   - Further improve GPU utilization
   - Optimize parallel training mechanisms
   - Enhance data collection efficiency

## 🤝 Contributing

We welcome improvements and code contributions through Issues and Pull Requests. We especially welcome contributions in:

- 🐛 Bug fixes and issue reports
- ✨ New features and improvements
- 📝 Documentation improvements and translations
- 🎯 New application scenario examples

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{matlab_ppo,
  author = {Haozheng Xie},
  title = {Matlab PPO: A Reinforcement Learning Framework for Control Systems},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/X-Embodied/Matlab_PPO}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file

## 🙏 Acknowledgments

Thanks to all researchers and developers who have contributed to this project!
