classdef MAPPOAgent < handle
    % MAPPOAgent 多智能体PPO算法的实现
    %   扩展了PPO算法以处理多智能体强化学习问题
    %   支持CTDE（集中训练，分散执行）的训练模式
    
    properties
        % 配置
        config              % 算法配置
        useGPU              % 是否使用GPU
        
        % 智能体信息
        numAgents           % 智能体数量
        actorNets           % 每个智能体的策略网络 (cell数组)
        criticNet           % 中央价值网络 (集中式学习)
        
        % 优化器
        actorOptimizers     % 每个智能体的策略优化器 (cell数组)
        criticOptimizer     % 价值网络优化器
        actorLearningRate   % Actor学习率
        criticLearningRate  % Critic学习率
        momentum            % 动量参数
        
        % 环境
        env                 % 多智能体环境
        
        % 日志
        logger              % 日志工具
    end
    
    methods
        function obj = MAPPOAgent(config)
            % 构造函数
            
            % 验证输入
            assert(~isempty(config.envName), 'MAPPOAgent:环境名称不能为空');
            assert(config.numAgents > 0, 'MAPPOAgent:智能体数量必须大于0');
            
            obj.config = config;
            obj.useGPU = config.useGPU;
            obj.numAgents = config.numAgents;
            
            % 创建环境
            try
                obj.env = feval(config.envName);
            catch ME
                error('MAPPOAgent:无法创建环境 %s: %s', config.envName, ME.message);
            end
            
            % 设置学习率和动量
            obj.actorLearningRate = config.actorLearningRate;
            obj.criticLearningRate = config.criticLearningRate;
            obj.momentum = config.momentum;
            
            % 初始化网络和优化器
            obj.initializeNetworks();
            
            % 创建日志记录器
            if ~isempty(config.logDir) && ~exist(config.logDir, 'dir')
                mkdir(config.logDir);
            end
            obj.logger = Logger(config.logDir);
        end
        
        function initializeNetworks(obj)
            % 初始化策略网络和价值网络
            
            % 为每个智能体创建策略网络（Actor）
            obj.actorNets = cell(obj.numAgents, 1);
            obj.actorOptimizers = cell(obj.numAgents, 1);
            
            for i = 1:obj.numAgents
                % 检查每个智能体的动作空间类型
                if obj.env.isDiscreteAction(i)
                    % 离散动作空间
                    obj.actorNets{i} = DiscreteActorNetwork(...
                        obj.env.observationSize(i), ...
                        obj.env.actionSize(i), ...
                        obj.config.actorLayerSizes, ...
                        obj.useGPU);
                else
                    % 连续动作空间
                    obj.actorNets{i} = ContinuousActorNetwork(...
                        obj.env.observationSize(i), ...
                        obj.env.actionSize(i), ...
                        obj.config.actorLayerSizes, ...
                        obj.useGPU);
                end
                
                % 初始化优化器状态为空（由dlupdate.sgdm管理）
                obj.actorOptimizers{i} = [];
            end
            
            % 创建中央价值网络（Critic）- 接收所有智能体的联合观察
            totalObsSize = sum(obj.env.observationSize);
            obj.criticNet = CriticNetwork(...
                totalObsSize, ...
                obj.config.criticLayerSizes, ...
                obj.useGPU);
            
            % 初始化价值网络优化器状态为空（由dlupdate.sgdm管理）
            obj.criticOptimizer = [];
        end
        
        function trajectories = collectTrajectories(obj, trajectoryLen, numTrajectories)
            % 收集多智能体轨迹样本
            %   trajectoryLen - 单个轨迹的长度
            %   numTrajectories - 需要收集的轨迹数量
            
            % 初始化轨迹数组
            trajectories = struct(...
                'observations', [], ...    % 每个智能体的观察
                'jointObservations', [], ... % 联合观察（所有智能体）
                'actions', [], ...         % 每个智能体的动作
                'logProbs', [], ...        % 每个智能体动作的对数概率
                'rewards', [], ...         % 每个轨迹步骤的奖励
                'dones', [], ...           % 每个轨迹步骤是否结束
                'values', [], ...          % 每个轨迹步骤的价值估计
                'return', 0, ...           % 轨迹的总回报
                'length', 0 ...            % 轨迹的实际长度
            );
            trajectories = repmat(trajectories, numTrajectories, 1);
            
            % 并行收集多条轨迹
            parfor trajectIdx = 1:numTrajectories
                % 重置环境
                [agentObs, jointObs] = obj.env.reset();
                
                % 初始化轨迹数据
                obsForAgents = cell(trajectoryLen, obj.numAgents);
                jointObsForAgents = zeros(trajectoryLen, sum(obj.env.observationSize));
                actionsForAgents = cell(trajectoryLen, obj.numAgents);
                logProbsForAgents = cell(trajectoryLen, obj.numAgents);
                rewards = zeros(trajectoryLen, 1);
                dones = false(trajectoryLen, 1);
                values = zeros(trajectoryLen, 1);
                
                trajectoryReturn = 0;
                trajectoryLength = trajectoryLen;
                
                % 收集轨迹数据
                for t = 1:trajectoryLen
                    % 存储观察
                    for i = 1:obj.numAgents
                        obsForAgents{t, i} = agentObs{i};
                    end
                    jointObsForAgents(t, :) = jointObs;
                    
                    % 获取价值估计
                    if obj.useGPU
                        dlJointObs = dlarray(single(jointObs), 'CB');
                        dlJointObs = gpuArray(dlJointObs);
                    else
                        dlJointObs = dlarray(single(jointObs), 'CB');
                    end
                    value = obj.criticNet.getValue(dlJointObs);
                    if obj.useGPU
                        value = gather(extractdata(value));
                    else
                        value = extractdata(value);
                    end
                    values(t) = value;
                    
                    % 为每个智能体采样动作
                    actions = cell(obj.numAgents, 1);
                    logProbs = cell(obj.numAgents, 1);
                    
                    for i = 1:obj.numAgents
                        if obj.useGPU
                            dlObs = dlarray(single(agentObs{i}), 'CB');
                            dlObs = gpuArray(dlObs);
                        else
                            dlObs = dlarray(single(agentObs{i}), 'CB');
                        end
                        
                        [action, logProb] = obj.actorNets{i}.sampleAction(dlObs);
                        
                        if obj.useGPU
                            action = gather(extractdata(action));
                            logProb = gather(extractdata(logProb));
                        else
                            action = extractdata(action);
                            logProb = extractdata(logProb);
                        end
                        
                        actions{i} = action;
                        logProbs{i} = logProb;
                    end
                    
                    % 存储动作和日志概率
                    for i = 1:obj.numAgents
                        actionsForAgents{t, i} = actions{i};
                        logProbsForAgents{t, i} = logProbs{i};
                    end
                    
                    % 执行动作
                    [nextAgentObs, nextJointObs, reward, done, ~] = obj.env.step(actions);
                    rewards(t) = reward;
                    dones(t) = done;
                    
                    % 累计回报
                    trajectoryReturn = trajectoryReturn + reward;
                    
                    % 更新观察
                    agentObs = nextAgentObs;
                    jointObs = nextJointObs;
                    
                    % 如果环境结束则提前停止
                    if done
                        trajectoryLength = t;
                        break;
                    end
                end
                
                % 保存轨迹数据
                trajectories(trajectIdx).observations = obsForAgents(1:trajectoryLength, :);
                trajectories(trajectIdx).jointObservations = jointObsForAgents(1:trajectoryLength, :);
                trajectories(trajectIdx).actions = actionsForAgents(1:trajectoryLength, :);
                trajectories(trajectIdx).logProbs = logProbsForAgents(1:trajectoryLength, :);
                trajectories(trajectIdx).rewards = rewards(1:trajectoryLength);
                trajectories(trajectIdx).dones = dones(1:trajectoryLength);
                trajectories(trajectIdx).values = values(1:trajectoryLength);
                trajectories(trajectIdx).return = trajectoryReturn;
                trajectories(trajectIdx).length = trajectoryLength;
            end
        end
        
        function trajectories = computeAdvantagesAndReturns(obj, trajectories)
            % 计算每个轨迹的优势函数和目标回报
            
            for i = 1:length(trajectories)
                % 获取轨迹数据
                rewards = trajectories(i).rewards;
                values = trajectories(i).values;
                dones = trajectories(i).dones;
                T = length(rewards);
                
                % 计算GAE (Generalized Advantage Estimation)
                advantages = zeros(T, 1);
                returns = zeros(T, 1);
                lastGAE = 0;
                lastValue = 0;
                
                % 如果轨迹未结束，使用最后的价值估计作为引导值
                if ~dones(end)
                    jointObs = trajectories(i).jointObservations(end, :);
                    if obj.useGPU
                        dlJointObs = dlarray(single(jointObs), 'CB');
                        dlJointObs = gpuArray(dlJointObs);
                    else
                        dlJointObs = dlarray(single(jointObs), 'CB');
                    end
                    
                    lastValue = obj.criticNet.getValue(dlJointObs);
                    if obj.useGPU
                        lastValue = gather(extractdata(lastValue));
                    else
                        lastValue = extractdata(lastValue);
                    end
                end
                
                % 反向计算GAE和目标回报
                for t = T:-1:1
                    if t == T
                        nextValue = lastValue;
                        nextNonTerminal = ~dones(t);
                    else
                        nextValue = values(t+1);
                        nextNonTerminal = ~dones(t);
                    end
                    
                    delta = rewards(t) + obj.config.gamma * nextValue * nextNonTerminal - values(t);
                    lastGAE = delta + obj.config.gamma * obj.config.lambda * nextNonTerminal * lastGAE;
                    advantages(t) = lastGAE;
                    returns(t) = advantages(t) + values(t);
                end
                
                % 归一化优势（可选）
                if length(advantages) > 1
                    advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
                end
                
                % 保存计算结果
                trajectories(i).advantages = advantages;
                trajectories(i).returns = returns;
            end
        end
        
        function [actorLoss, criticLoss, entropy] = updatePolicy(obj, trajectories, epochs, batchSize)
            % 使用收集的轨迹更新策略和价值网络
            
            % 将轨迹数据合并为单个批次
            allObservations = cell(obj.numAgents, 1);
            for i = 1:obj.numAgents
                allObservations{i} = [];
            end
            allJointObservations = [];
            allActions = cell(obj.numAgents, 1);
            for i = 1:obj.numAgents
                allActions{i} = [];
            end
            allLogProbs = cell(obj.numAgents, 1);
            for i = 1:obj.numAgents
                allLogProbs{i} = [];
            end
            allAdvantages = [];
            allReturns = [];
            
            % 合并所有轨迹数据
            for i = 1:length(trajectories)
                traj = trajectories(i);
                for j = 1:obj.numAgents
                    obsData = [];
                    for t = 1:traj.length
                        obsData = [obsData; traj.observations{t, j}];
                    end
                    allObservations{j} = [allObservations{j}; obsData];
                    
                    actionData = [];
                    for t = 1:traj.length
                        actionData = [actionData; traj.actions{t, j}];
                    end
                    allActions{j} = [allActions{j}; actionData];
                    
                    logProbData = [];
                    for t = 1:traj.length
                        logProbData = [logProbData; traj.logProbs{t, j}];
                    end
                    allLogProbs{j} = [allLogProbs{j}; logProbData];
                end
                
                allJointObservations = [allJointObservations; traj.jointObservations];
                allAdvantages = [allAdvantages; traj.advantages];
                allReturns = [allReturns; traj.returns];
            end
            
            % 数据集大小
            datasetSize = size(allJointObservations, 1);
            
            % 记录平均损失
            avgActorLoss = 0;
            avgCriticLoss = 0;
            avgEntropy = 0;
            
            % 多轮训练
            for epoch = 1:epochs
                % 随机打乱数据
                shuffleIdx = randperm(datasetSize);
                
                % 分批次训练
                numBatches = ceil(datasetSize / batchSize);
                
                for batchIdx = 1:numBatches
                    % 获取当前批次索引
                    startIdx = (batchIdx - 1) * batchSize + 1;
                    endIdx = min(batchIdx * batchSize, datasetSize);
                    batchIndices = shuffleIdx(startIdx:endIdx);
                    
                    % 为每个智能体更新策略网络
                    actorLosses = zeros(obj.numAgents, 1);
                    entropyValues = zeros(obj.numAgents, 1);
                    
                    for agentIdx = 1:obj.numAgents
                        % 准备批次数据
                        batchObs = allObservations{agentIdx}(batchIndices, :);
                        batchActions = allActions{agentIdx}(batchIndices, :);
                        batchOldLogProbs = allLogProbs{agentIdx}(batchIndices, :);
                        batchAdvantages = allAdvantages(batchIndices);
                        
                        % 转换为dlarray
                        if obj.useGPU
                            dlObs = dlarray(single(batchObs), 'CB');
                            dlObs = gpuArray(dlObs);
                            
                            if obj.env.isDiscreteAction(agentIdx)
                                % 离散动作不需要转换为dlarray
                                dlActions = batchActions;
                            else
                                dlActions = dlarray(single(batchActions), 'CB');
                                dlActions = gpuArray(dlActions);
                            end
                            
                            dlAdvantages = dlarray(single(batchAdvantages), 'CB');
                            dlAdvantages = gpuArray(dlAdvantages);
                            
                            dlOldLogProbs = dlarray(single(batchOldLogProbs), 'CB');
                            dlOldLogProbs = gpuArray(dlOldLogProbs);
                        else
                            dlObs = dlarray(single(batchObs), 'CB');
                            
                            if obj.env.isDiscreteAction(agentIdx)
                                % 离散动作不需要转换为dlarray
                                dlActions = batchActions;
                            else
                                dlActions = dlarray(single(batchActions), 'CB');
                            end
                            
                            dlAdvantages = dlarray(single(batchAdvantages), 'CB');
                            dlOldLogProbs = dlarray(single(batchOldLogProbs), 'CB');
                        end
                        
                        % 定义损失函数和梯度计算
                        [gradients, loss, entropy] = dlfeval(@obj.actorLossGradients, ...
                            obj.actorNets{agentIdx}, dlObs, dlActions, dlOldLogProbs, ...
                            dlAdvantages, obj.config.epsilon, obj.config.entropyCoef);
                        
                        % 应用梯度裁剪
                        gradients = dlupdate.clipgradients(gradients, obj.config.maxGradNorm);
                        
                        % 更新网络权重
                        [obj.actorNets{agentIdx}.learnables, obj.actorOptimizers{agentIdx}] = ...
                            dlupdate.sgdm(obj.actorNets{agentIdx}.learnables, gradients, ...
                            obj.actorOptimizers{agentIdx}, obj.actorLearningRate, obj.momentum);
                        
                        % 记录损失
                        if obj.useGPU
                            actorLosses(agentIdx) = gather(extractdata(loss));
                            entropyValues(agentIdx) = gather(extractdata(entropy));
                        else
                            actorLosses(agentIdx) = extractdata(loss);
                            entropyValues(agentIdx) = extractdata(entropy);
                        end
                    end
                    
                    % 更新中央价值网络
                    batchJointObs = allJointObservations(batchIndices, :);
                    batchReturns = allReturns(batchIndices);
                    
                    % 转换为dlarray
                    if obj.useGPU
                        dlJointObs = dlarray(single(batchJointObs), 'CB');
                        dlJointObs = gpuArray(dlJointObs);
                        
                        dlReturns = dlarray(single(batchReturns), 'CB');
                        dlReturns = gpuArray(dlReturns);
                    else
                        dlJointObs = dlarray(single(batchJointObs), 'CB');
                        dlReturns = dlarray(single(batchReturns), 'CB');
                    end
                    
                    % 计算价值网络损失和梯度
                    [gradients, criticLossValue] = dlfeval(@obj.criticLossGradients, ...
                        obj.criticNet, dlJointObs, dlReturns, obj.config.vfCoef);
                    
                    % 应用梯度裁剪
                    gradients = dlupdate.clipgradients(gradients, obj.config.maxGradNorm);
                    
                    % 更新网络权重
                    [obj.criticNet.learnables, obj.criticOptimizer] = ...
                        dlupdate.sgdm(obj.criticNet.learnables, gradients, ...
                        obj.criticOptimizer, obj.criticLearningRate, obj.momentum);
                    
                    % 记录价值网络损失
                    if obj.useGPU
                        criticLossValue = gather(extractdata(criticLossValue));
                    else
                        criticLossValue = extractdata(criticLossValue);
                    end
                    
                    % 累加损失
                    avgActorLoss = avgActorLoss + mean(actorLosses);
                    avgCriticLoss = avgCriticLoss + criticLossValue;
                    avgEntropy = avgEntropy + mean(entropyValues);
                end
            end
            
            % 计算平均损失
            totalBatches = epochs * numBatches;
            actorLoss = avgActorLoss / totalBatches;
            criticLoss = avgCriticLoss / totalBatches;
            entropy = avgEntropy / totalBatches;
        end
        
        function [gradients, loss, entropy] = actorLossGradients(obj, actorNet, obs, actions, oldLogProbs, advantages, epsilon, entropyCoef)
            % 计算策略网络的损失和梯度
            
            % 前向传播
            [logProbs, entValue] = actorNet.evaluateActions(obs, actions);
            
            % 计算比率
            ratios = exp(logProbs - oldLogProbs);
            
            % 裁剪比率
            clippedRatios = min(max(ratios, 1-epsilon), 1+epsilon);
            
            % 计算目标函数（取最小值）
            surrogate1 = ratios .* advantages;
            surrogate2 = clippedRatios .* advantages;
            surrogateLoss = -min(surrogate1, surrogate2);
            
            % 熵正则化
            entropyLoss = -entropyCoef * entValue;
            
            % 总损失
            loss = mean(surrogateLoss + entropyLoss);
            entropy = entValue;
            
            % 计算梯度
            gradients = dlgradient(loss, actorNet.learnables);
        end
        
        function [gradients, loss] = criticLossGradients(obj, criticNet, obs, returns, vfCoef)
            % 计算价值网络的损失和梯度
            
            % 前向传播
            values = criticNet.getValue(obs);
            
            % 均方误差损失
            valueLoss = mean((values - returns).^2);
            
            % 总损失
            loss = vfCoef * valueLoss;
            
            % 计算梯度
            gradients = dlgradient(loss, criticNet.learnables);
        end
        
        function train(obj, numIterations)
            % 训练多智能体PPO
            
            % 验证输入
            assert(numIterations > 0 && numIterations == floor(numIterations), ...
                'MAPPOAgent:迭代次数必须是正整数');
            
            for iter = 1:numIterations
                % 收集轨迹
                trajectories = obj.collectTrajectories(...
                    obj.config.trajectoryLen, obj.config.numTrajectories);
                
                % 计算优势和回报
                trajectories = obj.computeAdvantagesAndReturns(trajectories);
                
                % 更新策略和价值网络
                [actorLoss, criticLoss, entropy] = obj.updatePolicy(...
                    trajectories, obj.config.epochsPerIter, obj.config.batchSize);
                
                % 计算平均回报
                returns = [trajectories.return];
                meanReturn = mean(returns);
                stdReturn = std(returns);
                
                % 计算平均轨迹长度
                lengths = [trajectories.length];
                meanLength = mean(lengths);
                
                % 记录训练信息
                fprintf('迭代 %d/%d:\n', iter, numIterations);
                fprintf('  平均回报: %.2f ± %.2f\n', meanReturn, stdReturn);
                fprintf('  最小回报: %.2f\n', min(returns));
                fprintf('  最大回报: %.2f\n', max(returns));
                fprintf('  平均长度: %.2f\n', meanLength);
                fprintf('  Actor损失: %.4f\n', actorLoss);
                fprintf('  Critic损失: %.4f\n', criticLoss);
                fprintf('  熵: %.4f\n', entropy);
                
                % 记录到日志
                obj.logger.logTrainingMetrics(iter, meanReturn, meanLength, actorLoss, criticLoss, entropy);
                
                % 评估当前策略
                if obj.config.evalFreq > 0 && mod(iter, obj.config.evalFreq) == 0
                    evalResult = obj.evaluate(obj.config.numEvalEpisodes);
                    
                    fprintf('评估结果:\n');
                    fprintf('  平均回报: %.2f ± %.2f\n', evalResult.meanReturn, evalResult.stdReturn);
                    fprintf('  最小回报: %.2f\n', evalResult.minReturn);
                    fprintf('  最大回报: %.2f\n', evalResult.maxReturn);
                    fprintf('  平均长度: %.2f\n', evalResult.meanLength);
                    
                    % 记录评估指标
                    obj.logger.logEvaluationMetrics(iter, evalResult.meanReturn, ...
                        evalResult.stdReturn, evalResult.minReturn, evalResult.maxReturn);
                end
                
                % 保存模型
                if obj.config.saveModelFreq > 0 && mod(iter, obj.config.saveModelFreq) == 0
                    obj.saveModel(fullfile(obj.config.logDir, sprintf('model_iter_%d.mat', iter)));
                end
            end
            
            % 训练完成，保存最终模型
            obj.saveModel(fullfile(obj.config.logDir, 'model_final.mat'));
            
            % 绘制训练曲线
            obj.logger.plotTrainingCurves();
        end
        
        function result = evaluate(obj, numEpisodes, render)
            % 评估当前策略
            
            % 验证输入
            if nargin < 3
                render = false;
            end
            assert(numEpisodes > 0 && numEpisodes == floor(numEpisodes), ...
                'MAPPOAgent:评估回合数必须是正整数');
            
            returns = zeros(numEpisodes, 1);
            lengths = zeros(numEpisodes, 1);
            
            for ep = 1:numEpisodes
                [agentObs, ~] = obj.env.reset();
                episodeReturn = 0;
                episodeLength = 0;
                done = false;
                
                while ~done
                    % 为每个智能体选择动作
                    actions = cell(obj.numAgents, 1);
                    
                    for i = 1:obj.numAgents
                        if obj.useGPU
                            dlObs = dlarray(single(agentObs{i}), 'CB');
                            dlObs = gpuArray(dlObs);
                        else
                            dlObs = dlarray(single(agentObs{i}), 'CB');
                        end
                        
                        % 使用确定性策略（均值）
                        action = obj.actorNets{i}.getMeanAction(dlObs);
                        
                        if obj.useGPU
                            action = gather(extractdata(action));
                        else
                            action = extractdata(action);
                        end
                        
                        actions{i} = action;
                    end
                    
                    % 执行动作
                    [agentObs, ~, reward, done, ~] = obj.env.step(actions);
                    episodeReturn = episodeReturn + reward;
                    episodeLength = episodeLength + 1;
                    
                    % 如果需要渲染
                    if render
                        obj.env.render();
                        pause(0.01);
                    end
                end
                
                returns(ep) = episodeReturn;
                lengths(ep) = episodeLength;
            end
            
            % 计算统计信息
            result.meanReturn = mean(returns);
            result.stdReturn = std(returns);
            result.minReturn = min(returns);
            result.maxReturn = max(returns);
            result.meanLength = mean(lengths);
        end
        
        function saveModel(obj, filepath)
            % 保存模型权重和配置
            
            % 提取所有智能体的参数
            actorParams = cell(obj.numAgents, 1);
            for i = 1:obj.numAgents
                actorParams{i} = obj.actorNets{i}.learnables;
            end
            
            criticParams = obj.criticNet.learnables;
            config = obj.config;
            
            % 保存到.mat文件
            save(filepath, 'actorParams', 'criticParams', 'config');
            fprintf('模型已保存到 %s\n', filepath);
        end
        
        function loadModel(obj, filepath)
            % 加载模型权重
            
            % 验证文件是否存在
            assert(exist(filepath, 'file') == 2, 'MAPPOAgent:模型文件不存在: %s', filepath);
            
            % 加载.mat文件
            load(filepath, 'actorParams', 'criticParams');
            
            % 设置智能体参数
            for i = 1:obj.numAgents
                obj.actorNets{i}.learnables = actorParams{i};
            end
            
            % 设置价值网络参数
            obj.criticNet.learnables = criticParams;
            
            fprintf('模型已从 %s 加载\n', filepath);
        end
    end
end
