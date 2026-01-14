classdef PPOAgent < handle
    % PPOAgent 基于近端策略优化(PPO)算法的强化学习代理
    %   这个类实现了PPO算法，支持GPU并行化训练
    
    properties
        % 环境相关
        envName         % 环境名称
        env             % 环境对象
        obsSize         % 观察空间大小
        actionSize      % 动作空间大小
        isDiscrete      % 是否为离散动作空间
        
        % 策略网络
        actorNet        % 策略网络（Actor）
        criticNet       % 价值网络（Critic）
        actorOptimizer  % Actor优化器
        criticOptimizer % Critic优化器
        
        % 超参数
        gamma           % 折扣因子
        lambda          % GAE参数
        epsilon         % PPO裁剪参数
        entropyCoef     % 熵正则化系数
        vfCoef          % 价值函数系数
        maxGradNorm     % 梯度裁剪
        
        % 优化器参数
        actorLearningRate   % Actor学习率
        criticLearningRate  % Critic学习率
        momentum            % 动量参数
        
        % 训练参数
        batchSize       % 批次大小
        epochsPerIter   % 每次迭代的训练轮数
        trajectoryLen   % 轨迹长度
        
        % GPU加速
        useGPU          % 是否使用GPU
        gpuDevice       % GPU设备 
        
        % 记录与可视化
        logger          % 日志记录器
    end
    
    methods
        function obj = PPOAgent(config)
            % 构造函数：初始化PPO代理
            %   config - 包含所有配置参数的结构体
            
            % 验证输入
            assert(~isempty(config.envName), 'PPOAgent:环境名称不能为空');
            
            % 设置环境
            obj.envName = config.envName;
            try
                obj.env = feval(config.envName);
            catch ME
                error('PPOAgent:无法创建环境 %s: %s', config.envName, ME.message);
            end
            
            obj.obsSize = obj.env.observationSize;
            obj.actionSize = obj.env.actionSize;
            obj.isDiscrete = obj.env.isDiscrete;
            
            % 设置超参数
            obj.gamma = config.gamma;
            obj.lambda = config.lambda;
            obj.epsilon = config.epsilon;
            obj.entropyCoef = config.entropyCoef;
            obj.vfCoef = config.vfCoef;
            obj.maxGradNorm = config.maxGradNorm;
            
            % 设置训练参数
            obj.batchSize = config.batchSize;
            obj.epochsPerIter = config.epochsPerIter;
            obj.trajectoryLen = config.trajectoryLen;
            
            % 设置GPU加速
            obj.useGPU = config.useGPU;
            if obj.useGPU
                if gpuDeviceCount > 0
                    obj.gpuDevice = gpuDevice(1);
                    fprintf('使用GPU: %s\n', obj.gpuDevice.Name);
                else
                    warning('未检测到GPU, 将使用CPU训练');
                    obj.useGPU = false;
                end
            end
            
            % 初始化策略网络与价值网络
            obj.initNetworks(config);
            
            % 初始化优化器
            obj.initOptimizers(config);
            
            % 初始化日志记录
            obj.logger = Logger(config.logDir, obj.envName);
        end
        
        function initNetworks(obj, config)
            % 初始化策略网络与价值网络
            % 策略网络(Actor)
            if obj.isDiscrete
                obj.actorNet = DiscreteActorNetwork(obj.obsSize, obj.actionSize, config.actorLayerSizes);
            else
                obj.actorNet = ContinuousActorNetwork(obj.obsSize, obj.actionSize, config.actorLayerSizes);
            end
            
            % 价值网络(Critic)
            obj.criticNet = CriticNetwork(obj.obsSize, config.criticLayerSizes);
            
            % 如果使用GPU，则将网络迁移到GPU
            if obj.useGPU
                obj.actorNet.toGPU();
                obj.criticNet.toGPU();
            end
        end
        
        function initOptimizers(obj, config)
            % 初始化优化器状态（由dlupdate.sgdm管理）
            obj.actorOptimizer = [];
            obj.criticOptimizer = [];
            
            % 存储优化器参数
            obj.actorLearningRate = config.actorLearningRate;
            obj.criticLearningRate = config.criticLearningRate;
            obj.momentum = config.momentum;
        end
        
        function train(obj, numIterations)
            % 训练PPO代理
            %   numIterations - 训练迭代次数
            
            % 验证输入
            assert(numIterations > 0 && numIterations == floor(numIterations), ...
                'PPOAgent:迭代次数必须是正整数');
            
            fprintf('开始训练 %s 环境的PPO代理\n', obj.envName);
            
            for iter = 1:numIterations
                fprintf('迭代 %d/%d\n', iter, numIterations);
                
                % 收集轨迹数据
                fprintf('收集轨迹...\n');
                trajectories = obj.collectTrajectories();
                
                % 计算优势函数和回报
                fprintf('计算优势函数和回报...\n');
                trajectories = obj.computeAdvantagesAndReturns(trajectories);
                
                % PPO更新
                fprintf('执行PPO更新...\n');
                metrics = obj.updatePolicy(trajectories);
                
                % 记录本次迭代的性能指标
                obj.logger.logIteration(iter, metrics);
                
                % 每10次迭代保存模型
                if mod(iter, 10) == 0
                    obj.saveModel(fullfile(obj.logger.logDir, ['model_iter_', num2str(iter), '.mat']));
                end
            end
            
            fprintf('训练完成\n');
        end
        
        function trajectories = collectTrajectories(obj)
            % 收集训练轨迹
            % 返回一个轨迹结构体数组
            
            numTrajectories = max(1, ceil(obj.batchSize / obj.trajectoryLen));
            trajectories(numTrajectories) = struct();
            
            for i = 1:numTrajectories
                % 收集单条轨迹
                trajectories(i) = obj.collectSingleTrajectory();
            end
        end
        
        function trajectory = collectSingleTrajectory(obj)
            % 收集单条轨迹
            
            % 初始化轨迹存储
            trajectory = struct();
            trajectory.observations = cell(obj.trajectoryLen, 1);
            trajectory.actions = cell(obj.trajectoryLen, 1);
            trajectory.rewards = zeros(obj.trajectoryLen, 1);
            trajectory.dones = false(obj.trajectoryLen, 1);
            trajectory.values = zeros(obj.trajectoryLen, 1);
            trajectory.logProbs = zeros(obj.trajectoryLen, 1);
            
            % 重置环境
            obs = obj.env.reset();
            
            % 逐步收集数据
            for t = 1:obj.trajectoryLen
                % 转换为dlarray并根据需要迁移到GPU
                if obj.useGPU
                    dlObs = dlarray(single(obs), 'CB');
                    dlObs = gpuArray(dlObs);
                else
                    dlObs = dlarray(single(obs), 'CB');
                end
                
                % 通过策略网络获取动作
                [action, logProb] = obj.actorNet.sampleAction(dlObs);
                
                % 获取价值估计
                value = obj.criticNet.getValue(dlObs);
                
                % 转换为CPU并提取数值
                if obj.useGPU
                    action = gather(extractdata(action));
                    logProb = gather(extractdata(logProb));
                    value = gather(extractdata(value));
                else
                    action = extractdata(action);
                    logProb = extractdata(logProb);
                    value = extractdata(value);
                end
                
                % 在环境中执行动作
                [nextObs, reward, done, ~] = obj.env.step(action);
                
                % 存储当前步骤数据
                trajectory.observations{t} = obs;
                trajectory.actions{t} = action;
                trajectory.rewards(t) = reward;
                trajectory.dones(t) = done;
                trajectory.values(t) = value;
                trajectory.logProbs(t) = logProb;
                
                % 更新观察
                obs = nextObs;
                
                % 如果回合结束，重置环境
                if done && t < obj.trajectoryLen
                    obs = obj.env.reset();
                end
            end
        end
        
        function trajectories = computeAdvantagesAndReturns(obj, trajectories)
            % 计算广义优势估计(GAE)和折扣回报
            
            numTrajectories = length(trajectories);
            
            % 计算GAE和回报
            for i = 1:numTrajectories
                trajectory = trajectories(i);
                T = obj.trajectoryLen;
                
                % 初始化优势函数和回报
                advantages = zeros(T, 1);
                returns = zeros(T, 1);
                
                % 获取最终观察的价值估计
                lastObs = trajectory.observations{T};
                if ~trajectory.dones(T)
                    if obj.useGPU
                        dlObs = dlarray(single(lastObs), 'CB');
                        dlObs = gpuArray(dlObs);
                        lastValue = gather(extractdata(obj.criticNet.getValue(dlObs)));
                    else
                        dlObs = dlarray(single(lastObs), 'CB');
                        lastValue = extractdata(obj.criticNet.getValue(dlObs));
                    end
                else
                    lastValue = 0;
                end
                
                % GAE计算
                gae = 0;
                for t = T:-1:1
                    if t == T
                        nextValue = lastValue;
                        nextNonTerminal = 1 - trajectory.dones(T);
                    else
                        nextValue = trajectory.values(t+1);
                        nextNonTerminal = 1 - trajectory.dones(t);
                    end
                    
                    delta = trajectory.rewards(t) + obj.gamma * nextValue * nextNonTerminal - trajectory.values(t);
                    gae = delta + obj.gamma * obj.lambda * nextNonTerminal * gae;
                    advantages(t) = gae;
                    returns(t) = advantages(t) + trajectory.values(t);
                end
                
                % 存储计算结果
                trajectories(i).advantages = advantages;
                trajectories(i).returns = returns;
            end
        end
        
        function metrics = updatePolicy(obj, trajectories)
            % 使用PPO算法更新策略
            
            % 合并所有轨迹数据
            [observations, actions, oldLogProbs, returns, advantages] = obj.prepareTrainingData(trajectories);
            
            % 标准化优势函数
            advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
            
            % 数据集大小
            datasetSize = size(observations, 2);
            
            % 初始化指标
            actorLosses = [];
            criticLosses = [];
            entropyLosses = [];
            totalLosses = [];
            
            % 多轮训练
            for epoch = 1:obj.epochsPerIter
                % 随机打乱数据
                idx = randperm(datasetSize);
                
                % 分批处理
                for i = 1:obj.batchSize:datasetSize
                    endIdx = min(i + obj.batchSize - 1, datasetSize);
                    batchIdx = idx(i:endIdx);
                    
                    % 提取小批量数据
                    batchObs = observations(:, batchIdx);
                    batchActions = actions(:, batchIdx);
                    batchOldLogProbs = oldLogProbs(batchIdx);
                    batchReturns = returns(batchIdx);
                    batchAdvantages = advantages(batchIdx);
                    
                    % 转换为dlarray并根据需要迁移到GPU
                    if obj.useGPU
                        dlObs = dlarray(batchObs, 'CB');
                        dlObs = gpuArray(dlObs);
                        
                        dlActions = dlarray(batchActions, 'CB');
                        dlActions = gpuArray(dlActions);
                        
                        dlOldLogProbs = dlarray(batchOldLogProbs, 'CB');
                        dlOldLogProbs = gpuArray(dlOldLogProbs);
                        
                        dlReturns = dlarray(batchReturns, 'CB');
                        dlReturns = gpuArray(dlReturns);
                        
                        dlAdvantages = dlarray(batchAdvantages, 'CB');
                        dlAdvantages = gpuArray(dlAdvantages);
                    else
                        dlObs = dlarray(batchObs, 'CB');
                        dlActions = dlarray(batchActions, 'CB');
                        dlOldLogProbs = dlarray(batchOldLogProbs, 'CB');
                        dlReturns = dlarray(batchReturns, 'CB');
                        dlAdvantages = dlarray(batchAdvantages, 'CB');
                    end
                    
                    % 计算梯度并更新策略和价值网络
                    [actorGradients, criticGradients, actorLoss, criticLoss, entropyLoss, totalLoss] = ...
                        dlfeval(@obj.computeGradients, dlObs, dlActions, dlOldLogProbs, dlReturns, dlAdvantages);
                    
                    % 梯度裁剪
                    if obj.maxGradNorm > 0
                        actorGradients = thresholdL2Norm(actorGradients, obj.maxGradNorm);
                        criticGradients = thresholdL2Norm(criticGradients, obj.maxGradNorm);
                    end
                    
                    % 更新Actor网络
                    [obj.actorNet.learnables, obj.actorOptimizer] = sgdmupdate(obj.actorNet.learnables, actorGradients, obj.actorOptimizer, obj.actorLearningRate, obj.momentum);
                    
                    % 更新Critic网络
                    [obj.criticNet.learnables, obj.criticOptimizer] = sgdmupdate(obj.criticNet.learnables, criticGradients, obj.criticOptimizer, obj.criticLearningRate, obj.momentum);
                    
                    % 记录损失值
                    actorLosses(end+1) = actorLoss;
                    criticLosses(end+1) = criticLoss;
                    entropyLosses(end+1) = entropyLoss;
                    totalLosses(end+1) = totalLoss;
                end
            end
            
            % 返回本次更新的指标
            metrics = struct();
            metrics.actorLoss = mean(actorLosses);
            metrics.criticLoss = mean(criticLosses);
            metrics.entropyLoss = mean(entropyLosses);
            metrics.totalLoss = mean(totalLosses);
        end
        
        function [actorGradients, criticGradients, actorLoss, criticLoss, entropyLoss, totalLoss] = ...
                computeGradients(obj, observations, actions, oldLogProbs, returns, advantages)
            % 计算PPO损失和梯度
            
            % 前向传播
            [actorLossValue, criticLossValue, entropyLossValue, totalLossValue] = ...
                obj.computeLosses(observations, actions, oldLogProbs, returns, advantages);
            
            % 计算梯度
            [actorGradients, criticGradients] = dlgradient(totalLossValue, obj.actorNet.learnables, obj.criticNet.learnables);
            
            % 返回标量损失值
            actorLoss = extractdata(actorLossValue);
            criticLoss = extractdata(criticLossValue);
            entropyLoss = extractdata(entropyLossValue);
            totalLoss = extractdata(totalLossValue);
        end
        
        function [actorLoss, criticLoss, entropyLoss, totalLoss] = ...
                computeLosses(obj, observations, actions, oldLogProbs, returns, advantages)
            % 计算PPO损失函数
            
            % 获取当前策略的动作概率和价值
            [logProbs, entropy] = obj.actorNet.evaluateActions(observations, actions);
            values = obj.criticNet.getValue(observations);
            
            % 计算比率
            ratio = exp(logProbs - oldLogProbs);
            
            % 裁剪比率
            clippedRatio = min(max(ratio, 1 - obj.epsilon), 1 + obj.epsilon);
            
            % Actor损失 (策略损失)
            surrogateLoss1 = ratio .* advantages;
            surrogateLoss2 = clippedRatio .* advantages;
            actorLoss = -mean(min(surrogateLoss1, surrogateLoss2));
            
            % Critic损失 (价值损失)
            criticLoss = mean((values - returns).^2);
            
            % 熵损失 (用于鼓励探索)
            entropyLoss = -mean(entropy);
            
            % 总损失
            totalLoss = actorLoss + obj.vfCoef * criticLoss + obj.entropyCoef * entropyLoss;
        end
        
        function [observations, actions, logProbs, returns, advantages] = prepareTrainingData(obj, trajectories)
            % 将轨迹数据转换为训练所需的格式
            
            numTrajectories = length(trajectories);
            totalSteps = numTrajectories * obj.trajectoryLen;
            
            % 预分配存储空间
            observations = zeros(obj.obsSize, totalSteps);
            actions = zeros(obj.actionSize, totalSteps);
            logProbs = zeros(totalSteps, 1);
            returns = zeros(totalSteps, 1);
            advantages = zeros(totalSteps, 1);
            
            % 填充数据
            stepIdx = 1;
            for i = 1:numTrajectories
                for t = 1:obj.trajectoryLen
                    observations(:, stepIdx) = trajectories(i).observations{t};
                    actions(:, stepIdx) = trajectories(i).actions{t};
                    logProbs(stepIdx) = trajectories(i).logProbs(t);
                    returns(stepIdx) = trajectories(i).returns(t);
                    advantages(stepIdx) = trajectories(i).advantages(t);
                    
                    stepIdx = stepIdx + 1;
                end
            end
        end
        
        function saveModel(obj, filePath)
            % 保存模型
            actorParams = obj.actorNet.getParameters();
            criticParams = obj.criticNet.getParameters();
            
            % 如果参数在GPU上，移回CPU
            if obj.useGPU
                actorParams = cellfun(@gather, actorParams, 'UniformOutput', false);
                criticParams = cellfun(@gather, criticParams, 'UniformOutput', false);
            end
            
            % 保存模型参数和配置
            save(filePath, 'actorParams', 'criticParams');
            fprintf('模型已保存到: %s\n', filePath);
        end
        
        function loadModel(obj, filePath)
            % 加载模型
            
            % 验证文件是否存在
            assert(exist(filePath, 'file') == 2, 'PPOAgent:模型文件不存在: %s', filePath);
            
            load(filePath, 'actorParams', 'criticParams');
            
            % 设置模型参数
            if obj.useGPU
                actorParams = cellfun(@gpuArray, actorParams, 'UniformOutput', false);
                criticParams = cellfun(@gpuArray, criticParams, 'UniformOutput', false);
            end
            
            obj.actorNet.setParameters(actorParams);
            obj.criticNet.setParameters(criticParams);
            
            fprintf('模型已加载自: %s\n', filePath);
        end
        
        function result = evaluate(obj, numEpisodes)
            % 评估训练好的代理
            %   numEpisodes - 评估的回合数
            
            % 验证输入
            assert(numEpisodes > 0 && numEpisodes == floor(numEpisodes), ...
                'PPOAgent:评估回合数必须是正整数');
            
            returns = zeros(numEpisodes, 1);
            lengths = zeros(numEpisodes, 1);
            
            for i = 1:numEpisodes
                obs = obj.env.reset();
                done = false;
                episodeReturn = 0;
                episodeLength = 0;
                
                while ~done
                    % 转换为dlarray并根据需要迁移到GPU
                    if obj.useGPU
                        dlObs = dlarray(single(obs), 'CB');
                        dlObs = gpuArray(dlObs);
                    else
                        dlObs = dlarray(single(obs), 'CB');
                    end
                    
                    % 采样动作
                    [action, ~] = obj.actorNet.sampleAction(dlObs);
                    
                    % 转换为CPU并提取数值
                    if obj.useGPU
                        action = gather(extractdata(action));
                    else
                        action = extractdata(action);
                    end
                    
                    % 执行动作
                    [obs, reward, done, ~] = obj.env.step(action);
                    
                    episodeReturn = episodeReturn + reward;
                    episodeLength = episodeLength + 1;
                end
                
                returns(i) = episodeReturn;
                lengths(i) = episodeLength;
            end
            
            % 统计返回值
            result = struct();
            result.meanReturn = mean(returns);
            result.stdReturn = std(returns);
            result.minReturn = min(returns);
            result.maxReturn = max(returns);
            result.meanLength = mean(lengths);
        end
    end
end
