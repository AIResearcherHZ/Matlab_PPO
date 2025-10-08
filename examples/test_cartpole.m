% test_cartpole.m
% 测试已训练的倒立摆PPO模型

% 添加路径
addpath('../');
addpath('../core');
addpath('../environments');
addpath('../config');
addpath('../utils');

% 模型路径
modelPath = '../logs/cartpole/model_iter_100.mat';

% 检查模型文件是否存在
if ~exist(modelPath, 'file')
    error('模型文件不存在: %s\n请先运行train_cartpole.m进行训练', modelPath);
end

% 加载配置
config = PPOConfig();
config.envName = 'CartPoleEnv';
config.actorLayerSizes = [64, 64];
config.criticLayerSizes = [64, 64];
config.useGPU = true; % 根据需要设置是否使用GPU

% 创建PPO代理
try
    agent = PPOAgent(config);
catch ME
    error('创建PPO代理失败: %s', ME.message);
end

% 加载模型
fprintf('加载模型: %s\n', modelPath);
try
    agent.loadModel(modelPath);
catch ME
    error('加载模型失败: %s', ME.message);
end

% 测试参数
numEpisodes = 10;  % 测试回合数
renderTest = true; % 是否可视化测试过程

% 测试已训练的模型
fprintf('开始测试，共%d回合...\n', numEpisodes);
totalReward = 0;
totalSteps = 0;

% 创建环境
env = feval(config.envName);

for episode = 1:numEpisodes
    % 重置环境
    obs = env.reset();
    episodeReward = 0;
    steps = 0;
    done = false;
    
    while ~done
        % 转换为dlarray并根据需要迁移到GPU
        if agent.useGPU
            dlObs = dlarray(single(obs), 'CB');
            dlObs = gpuArray(dlObs);
        else
            dlObs = dlarray(single(obs), 'CB');
        end
        
        % 使用确定性策略（无探索）
        if isa(agent.actorNet, 'DiscreteActorNetwork')
            action = agent.actorNet.getBestAction(dlObs);
        else
            action = agent.actorNet.getMeanAction(dlObs);
        end
        
        % 转换为CPU并提取数值
        if agent.useGPU
            action = gather(extractdata(action));
        else
            action = extractdata(action);
        end
        
        % 如果是离散动作，转换为索引
        if env.isDiscrete
            [~, actionIdx] = max(action);
            action = actionIdx - 1; % 转为0-索引
        end
        
        % 执行动作
        [obs, reward, done, ~] = env.step(action);
        
        % 更新统计
        episodeReward = episodeReward + reward;
        steps = steps + 1;
        
        % 如果需要渲染
        if renderTest
            env.render();
            pause(0.01); % 控制渲染速度
        end
    end
    
    % 更新总统计
    totalReward = totalReward + episodeReward;
    totalSteps = totalSteps + steps;
    
    fprintf('回合 %d: 奖励 = %.2f, 步数 = %d\n', episode, episodeReward, steps);
end

% 打印测试结果摘要
fprintf('\n测试结果摘要:\n');
fprintf('平均奖励: %.2f\n', totalReward / numEpisodes);
fprintf('平均步数: %.2f\n', totalSteps / numEpisodes);
fprintf('测试完成\n');
