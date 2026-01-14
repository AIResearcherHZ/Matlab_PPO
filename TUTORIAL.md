# Matlab PPO Reinforcement Learning Framework Detailed Tutorial

> For Chinese version, please refer to [TUTORIAL_zh.md](TUTORIAL_zh.md)

This tutorial provides an in-depth explanation of the Matlab PPO reinforcement learning framework, including algorithm principles, implementation details, environment model descriptions and extension guidelines.

## Table of Contents

1. [Algorithm Details](#algorithm-details)
   - [PPO Algorithm Principles](#ppo-algorithm-principles)
   - [MAPPO Algorithm Extension](#mappo-algorithm-extension)
2. [Environment Models](#environment-models)
   - [CartPole](#cartpole)
   - [DC Motor Control](#dc-motor-control)
   - [AC Induction Motor FOC Control](#ac-induction-motor-foc-control)
   - [Double Pendulum System](#double-pendulum-system)
3. [Framework Implementation Details](#framework-implementation-details)
   - [Core Classes](#core-classes)
   - [Network Architecture](#network-architecture)
   - [Training Process](#training-process)
4. [Advanced Application Guide](#advanced-application-guide)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Custom Environment Development](#custom-environment-development)
   - [Multi-Agent System Design](#multi-agent-system-design)
5. [Performance Optimization and Debugging](#performance-optimization-and-debugging)
   - [GPU Acceleration](#gpu-acceleration)
   - [Parallel Computing](#parallel-computing)
   - [Troubleshooting](#troubleshooting)

## Algorithm Details

### PPO Algorithm Principles

PPO (Proximal Policy Optimization) is a policy gradient method that balances sample efficiency and implementation simplicity. It introduces a clipping term during policy updates to limit the difference between old and new policies, preventing excessively large policy updates.

#### Core Mathematical Principles

The PPO objective function is:

$$L^{CLIP}(\theta) = \hat{E}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ represents the probability ratio between old and new policies
- $\hat{A}_t$ is the advantage function estimate
- $\epsilon$ is the clipping parameter, typically set to 0.2

#### Algorithm Steps

1. **Sampling**: Collect a batch of trajectory data using current policy $\pi_{\theta_{old}}$
2. **Advantage Estimation**: Calculate advantage estimates $\hat{A}_t$ for each state-action pair
3. **Policy Update**: Maximize the clipped objective through multiple mini-batch gradient ascent steps
4. **Value Function Update**: Update the value function to better estimate returns

#### Generalized Advantage Estimation (GAE)

PPO typically uses GAE to calculate the advantage function, balancing bias and variance:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + ... + (\gamma\lambda)^{T-t+1}\delta_{T-1}$$

Where:
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the temporal difference error
- $\gamma$ is the discount factor
- $\lambda$ is the GAE parameter, controlling the bias-variance trade-off

### MAPPO Algorithm Extension

MAPPO (Multi-Agent PPO) is an extension of PPO for multi-agent environments. It adopts the CTDE (Centralized Training with Decentralized Execution) framework, allowing the use of global information during training while only using local observations during execution.

#### Core Design

1. **Independent Policy Networks**: Each agent $i$ has its own policy network $\pi_i(a_i|o_i)$, making decisions based on local observations $o_i$
2. **Centralized Value Network**: Uses a centralized critic network $V(s)$ to evaluate global state value
3. **Coordinated Training**: Considers interactions between agents to optimize joint policy returns

#### Differences from Single-Agent PPO

- **Observation-State Separation**: Each agent only receives local observations, but the value network can use global state
- **Credit Assignment**: Needs to solve the credit assignment problem in multi-agent settings, determining each agent's contribution to global returns
- **Collaborative Policy**: Learns cooperative strategies through implicit coordination between agents

#### Application Scenarios

MAPPO is particularly suitable for:
- Naturally distributed control problems (e.g., multi-joint robots)
- Tasks requiring cooperation (e.g., multi-agent collaborative control)
- Tasks requiring specialized agent roles (e.g., controllers with different functions)

## Environment Models

### CartPole

CartPole is a classic control problem and a standard test environment for reinforcement learning beginners.

#### Physical Model

The CartPole system consists of a horizontally movable cart and a rigid pendulum connected to the cart. The system dynamics can be described by the following differential equations:

$$\ddot{x} = \frac{F + m l \sin\theta (\dot{\theta})^2 - m g \cos\theta \sin\theta}{M + m \sin^2\theta}$$

$$\ddot{\theta} = \frac{g \sin\theta - \ddot{x}\cos\theta}{l}$$

Where:
- $x$ is the cart position
- $\theta$ is the pole angle (relative to vertical upward direction)
- $F$ is the force applied to the cart
- $M$ is the cart mass
- $m$ is the pole mass
- $l$ is the pole half-length
- $g$ is the gravitational acceleration

#### Reinforcement Learning Setup

- **State Space**: $[x, \dot{x}, \theta, \dot{\theta}]$ 
- **Action Space**: Discrete actions {move left, move right}
- **Reward**: +1 for each timestep until termination
- **Termination Conditions**: Pole angle exceeds 15 degrees or cart position deviates more than 2.4 units from center

In our implementation, `CartPoleEnv.m` provides complete environment simulation including dynamics update, reward calculation and visualization functions.

### DC Motor Control

DC motor control is a fundamental industrial control problem involving dynamic models of electrical and mechanical systems.

#### Motor Model

The dynamic equations of a DC motor are as follows:

**Electrical Equation**:
$$L\frac{di}{dt} = v - Ri - K_e\omega$$

**Mechanical Equation**:
$$J\frac{d\omega}{dt} = K_ti - B\omega - T_L$$

Where:
- $v$ is the applied voltage
- $i$ is the motor current
- $\omega$ is the motor angular velocity
- $T_L$ is the load torque
- $L$ is the inductance
- $R$ is the resistance
- $K_t$ is the torque constant
- $K_e$ is the back EMF constant
- $J$ is the moment of inertia
- $B$ is the friction coefficient

#### Reinforcement Learning Setup

- **State Space**: $[\omega, i, \omega_{ref}, \omega_{ref} - \omega]$
- **Action Space**: Continuous actions representing applied voltage $v \in [-V_{max}, V_{max}]$
- **Reward**: Combines speed tracking error, control signal magnitude and energy consumption
- **Termination Conditions**: Maximum steps reached or excessive speed error

`DCMotorEnv.m` implements this environment model, including discretized dynamic equation solving, step response testing and load disturbance simulation.

### AC Induction Motor FOC Control

AC induction motor control is a more complex problem. This framework implements learning strategies for AC motor control based on Field-Oriented Control (FOC).

#### FOC Principles

FOC transforms AC motor control into a problem similar to DC motor control through coordinate transformation:

1. **Three-phase to Two-phase Transformation**: Converts three-phase currents/voltages ($i_a$, $i_b$, $i_c$) to two-phase stationary coordinate system ($i_\alpha$, $i_\beta$)
2. **Stationary to Rotating Coordinates**: Transforms stationary coordinates to rotating coordinates ($i_d$, $i_q$) synchronized with rotor magnetic field
3. **Control in d-q Coordinates**: In the rotating coordinate system, $i_d$ controls flux and $i_q$ controls torque

#### AC Motor Model

In the d-q rotating coordinate system, the voltage equations of the induction motor are:

$$v_d = R_si_d + \frac{d\lambda_d}{dt} - \omega_e\lambda_q$$
$$v_q = R_si_q + \frac{d\lambda_q}{dt} + \omega_e\lambda_d$$

Flux linkage equations:
$$\lambda_d = L_di_d + L_mi_{dr}$$
$$\lambda_q = L_qi_q + L_mi_{qr}$$

#### Reinforcement Learning Setup

- **State Space**: $[\omega_r, i_d, i_q, \omega_{ref}, \lambda_d, \lambda_q]$
- **Action Space**: Continuous actions representing d-q axis voltages $v_d$ and $v_q$
- **Reward**: Combines speed tracking error, flux linkage stability, current limits and energy efficiency

### Double Pendulum System

The double pendulum system is a classic multi-agent collaboration problem, suitable for solving with MAPPO algorithm.

#### System Model

The double pendulum system consists of two connected rigid pendulum rods, each controlled by an independent actuator. The system's Lagrangian equation is:

$$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = \tau$$

Where:
- $q = [\theta_1, \theta_2]^T$ is the pendulum angle vector
- $M(q)$ is the mass matrix
- $C(q,\dot{q})$ represents Coriolis and centrifugal terms
- $G(q)$ is the gravity term
- $\tau = [\tau_1, \tau_2]^T$ is the torque applied by actuators

#### Multi-Agent Reinforcement Learning Setup

- **Agent Observations**: Each agent observes its own pendulum state and limited information about adjacent pendulums
- **Action Space**: Each agent controls an actuator torque $\tau_i$
- **Global State**: Complete dynamic state of the entire system
- **Joint Reward**: Shared reward function based on system stability

`DoublePendulumEnv.m` implements this environment and provides multi-agent interfaces to support MAPPO algorithm training.

## Framework Implementation Details

### Core Classes

#### PPOAgent Class

`PPOAgent.m` is the core implementation of single-agent PPO algorithm, with main methods including:

- **collectTrajectories**: Collects agent-environment interaction trajectories
- **computeGAE**: Calculates advantage values using Generalized Advantage Estimation
- **updatePolicy**: Updates policy and value networks based on collected data
- **train**: Executes complete training loop
- **getAction**: Gets action for given observation

Key attributes include policy network, value network and various training parameters.

#### MAPPOAgent Class

`MAPPOAgent.m` extends PPO algorithm to multi-agent scenarios, with main features:

- Manages policy networks (Actors) for multiple agents
- Uses centralized critic network to estimate joint value function
- Coordinates trajectory collection and policy updates for multiple agents

#### ActorNetwork and CriticNetwork

- **ContinuousActorNetwork.m**: Implements policy network for continuous action spaces
- **DiscreteActorNetwork.m**: Implements policy network for discrete action spaces
- **CriticNetwork.m**: Implements state value estimation network

These network classes encapsulate network structure, forward propagation and gradient calculation functions.

### Network Architecture Design

#### Policy Network

The policy network (Actor) typically consists of the following components:

```
Observation Input -> Fully Connected Layer -> ReLU -> Fully Connected Layer -> ReLU -> Output Layer
```

For continuous action spaces, the output layer generates action mean and standard deviation; for discrete action spaces, the output layer generates action probability distribution.

#### Value Network

The value network (Critic) has a similar structure:

```
State Input -> Fully Connected Layer -> ReLU -> Fully Connected Layer -> ReLU -> Scalar Output (State Value)
```

For MAPPO, the value network receives joint observations as input to evaluate global state value.

### Training Process Description

The typical PPO training process is as follows:

1. **Network Initialization**: Create policy and value networks
2. **Training Loop**:
   - Collect trajectory data through environment interaction
   - Calculate returns and advantage estimates
   - Update policy network (multiple mini-batch updates)
   - Update value network
   - Record and visualize training data
3. **Model Saving**: Save trained network parameters to file

The MAPPO training process is similar but requires coordination of data collection and policy updates for multiple agents.

## Advanced Application Guide

### Hyperparameter Tuning

Reinforcement learning algorithms are highly sensitive to hyperparameters. Here are key hyperparameters and tuning recommendations:

- **PPO clipping parameter (epsilon)**: Typically set to 0.1-0.3, controls policy update magnitude
- **Discount factor (gamma)**: Typically set to 0.95-0.99, controls importance of future rewards
- **GAE parameter (lambda)**: Typically set to 0.9-0.99, controls bias-variance tradeoff
- **Learning rate**: For policy and value networks, usually in range 1e-4 to 1e-3
- **Network size**: Hidden layer size and count, depends on task complexity

For complex environments, grid search or Bayesian optimization is recommended to find optimal hyperparameter combinations.

### Custom Environment Development

Creating custom environments is an important way to extend the framework's functionality. Here are the basic steps for developing a new environment:

1. **Inherit Base Class**: The new environment should inherit from the `Environment` base class
2. **Implement Required Methods**:
   - `reset()`: Reset the environment to initial state and return initial observation
   - `step(action)`: Execute action, update environment state and return (next observation, reward, done, info)
   - `render()`: Optional visualization method

#### Custom Environment Example

```matlab
classdef MyCustomEnv < Environment
    properties
        % Environment state variables
        state
        
        % Environment parameters
        param1
        param2
    end
    
    methods
        function obj = MyCustomEnv(config)
            % Initialize environment parameters
            obj.param1 = config.param1Value;
            obj.param2 = config.param2Value;
            
            % Define observation and action space dimensions
            obj.observationDimension = 4;
            obj.continuousAction = true;  % Use continuous action space
            obj.actionDimension = 2;
        end
        
        function observation = reset(obj)
            % Reset environment state
            obj.state = [0; 0; 0; 0];
            
            % Return initial observation
            observation = obj.state;
        end
        
        function [nextObs, reward, done, info] = step(obj, action)
            % Validate action
            action = min(max(action, -1), 1);  % Clip action to [-1,1]
            
            % Update environment state
            % ... Implement state transition equations ...
            
            % Calculate reward
            reward = calculateReward(obj, action);
            
            % Check termination condition
            done = checkTermination(obj);
            
            % Return results
            nextObs = obj.state;
            info = struct();  % Can contain additional information
        end
        
        function render(obj)
            % Implement visualization
            figure(1);
            % ... Draw environment state ...
            drawnow;
        end
        
        function reward = calculateReward(obj, action)
            % Custom reward function
            % ... Calculate reward ...
        end
        
        function done = checkTermination(obj)
            % Check termination conditions
            % ... Determine if episode should end ...
        end
    end
end
```

### Multi-Agent System Design

Designing multi-agent systems requires consideration of the following key points:

1. **Environment Interface**: The multi-agent environment should provide interfaces supporting multiple agents
   - Implement `getNumAgents()` method to return agent count
   - `step(actions)` should receive joint actions from all agents
   - `reset()` should return initial observations for each agent

2. **Observation and State Design**:
   - Clearly distinguish between local observations (visible to each agent) and global state (visible to centralized critic)
   - Define observation function `getObservation(agentIdx)` to get specified agent's observation

3. **Reward Design**:
   - Shared reward: All agents receive same reward to promote cooperation
   - Individual reward: Each agent has its own reward function, which may lead to competition
   - Hybrid reward: Combine shared and individual rewards to balance cooperation and specific task objectives

## Performance Optimization and Debugging

### GPU Acceleration

This framework supports MATLAB's GPU acceleration to significantly improve training speed:

```matlab
% Enable GPU in configuration
config = PPOConfig();
config.useGPU = true;

% Create agent - GPU migration is handled automatically
agent = PPOAgent(config);
```

Using GPU acceleration requires installing compatible CUDA and GPU computing toolboxes. For large networks, GPU acceleration can improve training speed by 5-10x.

### Parallel Computing

This framework utilizes MATLAB's Parallel Computing Toolbox for parallelizing data collection:

```matlab
% The framework handles trajectory collection internally
% For parallel computing, ensure Parallel Computing Toolbox is installed
config = PPOConfig();
config.numTrajectories = 10;  % Number of trajectories to collect

% Create and train agent
agent = PPOAgent(config);
agent.train(config.numIterations);
```

Parallel computing is particularly suitable for trajectory collection as different trajectories have no dependencies. For complex environments and large numbers of trajectories, near-linear speedup can be achieved.

### Troubleshooting

#### 1. Training Not Converging

Possible causes and solutions:
- **Learning rate too high**: Try reducing learning rate
- **Network structure unsuitable**: Increase network capacity or adjust structure
- **Poor reward design**: Redesign more instructive reward function
- **Inaccurate advantage estimation**: Adjust GAE parameter (lambda)

#### 2. Unstable Training Process

Possible causes and solutions:
- **Batch size too small**: Increase batch size to reduce gradient estimation variance
- **Inappropriate clipping parameter**: Adjust epsilon value (0.1-0.3)
- **Too many update steps**: Reduce the number of updates per batch of data

#### 3. Multi-Agent Coordination Issues

Possible causes and solutions:
- **Poor observation space design**: Ensure agents have sufficient information for collaboration
- **Shared reward ratio problem**: Adjust the ratio between shared and individual rewards
- **Insufficient value network capacity**: Increase the capacity of the centralized critic network

#### 4. Performance Analysis Tools

Using MATLAB's built-in performance analysis tools:
```matlab
% Enable code profiling
profile on

% Run the code to be analyzed
agent.train(env, 10);

% View the profiling report
profile viewer
```

This helps identify bottlenecks in the code and optimize critical sections.

## Summary

The Matlab PPO framework provides powerful and flexible infrastructure for solving various control problems, from basic inverted pendulums to complex multi-agent systems. By deeply understanding the algorithm principles, implementation details and environment models, users can fully utilize the framework's capabilities and extend/optimize it according to their needs.

Whether for research or engineering applications, this framework provides the necessary tools and flexibility to address challenges in modern control systems. As reinforcement learning continues to evolve, we will keep updating and improving this framework to maintain its practicality and advancement.
