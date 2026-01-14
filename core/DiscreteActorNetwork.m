classdef DiscreteActorNetwork < handle
    % DiscreteActorNetwork 用于离散动作空间的策略网络
    %   实现了具有Categorical分布输出的策略网络
    
    properties
        layerSizes      % 隐藏层大小
        learnables      % 可学习参数
        useGPU          % 是否使用GPU
    end
    
    methods
        function obj = DiscreteActorNetwork(inputSize, outputSize, layerSizes)
            % 构造函数：初始化离散动作空间的Actor网络
            %   inputSize - 输入维度（观察空间大小）
            %   outputSize - 输出维度（动作空间大小，即动作数量）
            %   layerSizes - 隐藏层大小的数组
            
            obj.layerSizes = layerSizes;
            obj.useGPU = false;
            
            % 初始化网络参数
            obj.learnables = struct();
            
            % 初始化第一个隐藏层
            layerIdx = 1;
            obj.learnables.fc1w = dlarray(initializeGlorot(layerSizes(1), inputSize), 'CB');
            obj.learnables.fc1b = dlarray(zeros(layerSizes(1), 1), 'CB');
            
            % 初始化中间隐藏层
            for i = 2:length(layerSizes)
                obj.learnables.(sprintf('fc%dw', i)) = dlarray(initializeGlorot(layerSizes(i), layerSizes(i-1)), 'CB');
                obj.learnables.(sprintf('fc%db', i)) = dlarray(zeros(layerSizes(i), 1), 'CB');
            end
            
            % 初始化输出层（logits）
            obj.learnables.outw = dlarray(initializeGlorot(outputSize, layerSizes(end)), 'CB');
            obj.learnables.outb = dlarray(zeros(outputSize, 1), 'CB');
        end
        
        function [action, logProb, probs] = sampleAction(obj, observation)
            % 采样一个动作并计算其对数概率
            %   observation - 当前的观察状态
            %   action - 采样的动作（one-hot编码）
            %   logProb - 动作的对数概率
            %   probs - 各动作的概率
            
            % 前向传播，获取动作概率
            logits = obj.forward(observation);
            probs = softmax(logits);
            
            % 从类别分布中采样
            probsData = extractdata(probs);
            cumProbs = cumsum(probsData, 1);
            
            % 生成随机数
            if obj.useGPU
                r = rand(1, size(probs, 2), 'gpuArray');
            else
                r = rand(1, size(probs, 2));
            end
            
            % 矢量化采样：找到每个样本对应的动作索引
            actionIndices = sum(cumProbs < r, 1) + 1;
            
            % 创建one-hot编码的动作
            actionSize = size(probs, 1);
            batchSize = size(probs, 2);
            
            if obj.useGPU
                action = dlarray(gpuArray(zeros(actionSize, batchSize, 'single')), 'CB');
            else
                action = dlarray(zeros(actionSize, batchSize, 'single'), 'CB');
            end
            
            % 使用线性索引设置one-hot编码
            linearIndices = sub2ind([actionSize, batchSize], actionIndices, 1:batchSize);
            actionData = extractdata(action);
            actionData(linearIndices) = 1;
            action = dlarray(actionData, 'CB');
            
            % 计算对数概率
            logProb = log(sum(probs .* action, 1) + 1e-10);
        end
        
        function [logProb, entropy] = evaluateActions(obj, observation, action)
            % 评估动作的对数概率和熵
            %   observation - 观察状态
            %   action - 执行的动作（one-hot编码）
            %   logProb - 动作的对数概率
            %   entropy - 策略的熵
            
            % 前向传播，获取动作概率
            logits = obj.forward(observation);
            probs = softmax(logits);
            
            % 计算对数概率
            logProb = log(sum(probs .* action, 1) + 1e-10);
            
            % 计算熵
            entropy = -sum(probs .* log(probs + 1e-10), 1);
        end
        
        function logits = forward(obj, observation)
            % 前向传播，计算动作分布参数
            %   observation - 观察状态
            %   logits - 未经softmax的输出
            
            % 第一个隐藏层
            x = fullyconnect(observation, obj.learnables.fc1w, obj.learnables.fc1b);
            x = relu(x);
            
            % 中间隐藏层
            for i = 2:length(obj.layerSizes)
                x = fullyconnect(x, obj.learnables.(sprintf('fc%dw', i)), obj.learnables.(sprintf('fc%db', i)));
                x = relu(x);
            end
            
            % 输出层
            logits = fullyconnect(x, obj.learnables.outw, obj.learnables.outb);
        end
        
        function action = getBestAction(obj, observation)
            % 获取确定性动作（最高概率）
            %   observation - 观察状态
            %   action - 确定性动作（one-hot编码）
            
            % 前向传播，获取动作概率
            logits = obj.forward(observation);
            probs = softmax(logits);
            
            % 选择最高概率的动作
            [~, idx] = max(extractdata(probs), [], 1);
            
            % 创建one-hot动作（矢量化）
            actionSize = size(probs, 1);
            batchSize = size(probs, 2);
            action = zeros(size(probs), 'like', probs);
            
            % 使用线性索引设置one-hot编码
            linearIndices = sub2ind([actionSize, batchSize], idx, 1:batchSize);
            actionData = extractdata(action);
            actionData(linearIndices) = 1;
            action = dlarray(actionData, 'CB');
        end
        
        function action = getMeanAction(obj, observation)
            % 获取确定性动作（与getBestAction相同，用于兼容连续动作网络接口）
            action = obj.getBestAction(observation);
        end
        
        function toGPU(obj)
            % 将网络参数转移到GPU
            obj.useGPU = true;
            
            % 将所有参数移至GPU
            fnames = fieldnames(obj.learnables);
            for i = 1:length(fnames)
                obj.learnables.(fnames{i}) = gpuArray(obj.learnables.(fnames{i}));
            end
        end
        
        function toCPU(obj)
            % 将网络参数转移回CPU
            obj.useGPU = false;
            
            % 将所有参数移回CPU
            fnames = fieldnames(obj.learnables);
            for i = 1:length(fnames)
                obj.learnables.(fnames{i}) = gather(obj.learnables.(fnames{i}));
            end
        end
        
        function params = getParameters(obj)
            % 获取网络参数
            params = struct2cell(obj.learnables);
        end
        
        function setParameters(obj, params)
            % 设置网络参数
            fnames = fieldnames(obj.learnables);
            for i = 1:length(fnames)
                obj.learnables.(fnames{i}) = params{i};
            end
        end
    end
end

function W = initializeGlorot(numOut, numIn)
    % Glorot/Xavier初始化
    stddev = sqrt(2 / (numIn + numOut));
    W = stddev * randn(numOut, numIn);
end
