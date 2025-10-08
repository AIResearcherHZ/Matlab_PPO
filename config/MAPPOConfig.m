classdef MAPPOConfig < handle
    % MAPPOConfig 多智能体PPO的配置类
    %   管理多智能体PPO算法的各种参数和配置
    
    properties
        % 环境配置
        envName             % 环境名称
        numAgents           % 智能体数量
        
        % 网络配置
        actorLayerSizes     % 策略网络隐藏层大小
        criticLayerSizes    % 中央价值网络隐藏层大小
        
        % 算法超参数
        gamma               % 折扣因子
        lambda              % GAE参数
        epsilon             % PPO裁剪参数
        entropyCoef         % 熵正则化系数
        vfCoef              % 价值函数系数
        maxGradNorm         % 梯度裁剪阈值
        
        % 优化器配置
        actorLearningRate   % 策略网络学习率
        criticLearningRate  % 价值网络学习率
        momentum            % 动量
        
        % 训练配置
        numIterations       % 训练迭代次数
        batchSize           % 每次更新的批次大小
        epochsPerIter       % 每次迭代的训练轮数
        trajectoryLen       % 轨迹长度
        numTrajectories     % 每次迭代收集的轨迹数量
        
        % 硬件配置
        useGPU              % 是否使用GPU
        
        % 日志配置
        logDir              % 日志保存目录
        evalFreq            % 评估频率（迭代次数）
        numEvalEpisodes     % 评估时的回合数
        saveModelFreq       % 保存模型频率（迭代次数）
    end
    
    methods
        function obj = MAPPOConfig()
            % 构造函数：使用默认值初始化
            
            % 环境配置
            obj.envName = '';
            obj.numAgents = 2;
            
            % 网络配置
            obj.actorLayerSizes = [64, 64];
            obj.criticLayerSizes = [128, 128];
            
            % 算法超参数
            obj.gamma = 0.99;
            obj.lambda = 0.95;
            obj.epsilon = 0.2;
            obj.entropyCoef = 0.01;
            obj.vfCoef = 0.5;
            obj.maxGradNorm = 0.5;
            
            % 优化器配置
            obj.actorLearningRate = 3e-4;
            obj.criticLearningRate = 1e-3;
            obj.momentum = 0.9;
            
            % 训练配置
            obj.numIterations = 100;
            obj.batchSize = 64;
            obj.epochsPerIter = 4;
            obj.trajectoryLen = 200;
            obj.numTrajectories = 10;
            
            % 硬件配置
            obj.useGPU = true;
            
            % 日志配置
            obj.logDir = 'logs/mappo';
            obj.evalFreq = 10;
            obj.numEvalEpisodes = 5;
            obj.saveModelFreq = 20;
        end
        
        function params = toStruct(obj)
            % 将配置转换为结构体，用于保存
            params = struct();
            
            props = properties(obj);
            for i = 1:length(props)
                propName = props{i};
                params.(propName) = obj.(propName);
            end
        end
        
        function save(obj, filepath)
            % 保存配置到文件
            config = obj.toStruct();
            save(filepath, '-struct', 'config');
            fprintf('配置已保存到: %s\n', filepath);
        end
        
        function obj = loadFromFile(obj, filepath)
            % 从文件加载配置
            assert(exist(filepath, 'file') == 2, 'MAPPOConfig:配置文件不存在: %s', filepath);
            
            % 加载配置
            config = load(filepath);
            
            % 更新对象属性
            props = fieldnames(config);
            for i = 1:length(props)
                propName = props{i};
                if isprop(obj, propName)
                    obj.(propName) = config.(propName);
                end
            end
            
            fprintf('配置已从 %s 加载\n', filepath);
        end
    end
end
