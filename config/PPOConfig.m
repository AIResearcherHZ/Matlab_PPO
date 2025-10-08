classdef PPOConfig < handle
    % PPOConfig PPO算法配置类
    %   用于管理和加载PPO算法的所有配置参数
    
    properties
        % 环境配置
        envName                 % 环境名称
        
        % 网络配置
        actorLayerSizes         % Actor网络隐藏层大小
        criticLayerSizes        % Critic网络隐藏层大小
        
        % 算法超参数
        gamma                   % 折扣因子
        lambda                  % GAE参数
        epsilon                 % PPO裁剪参数
        entropyCoef             % 熵正则化系数
        vfCoef                  % 价值函数系数
        maxGradNorm             % 梯度裁剪
        
        % 优化器配置
        actorLearningRate       % Actor学习率
        criticLearningRate      % Critic学习率
        momentum                % 动量
        
        % 训练配置
        numIterations           % 训练迭代次数
        batchSize               % 批次大小
        epochsPerIter           % 每次迭代的训练轮数
        trajectoryLen           % 轨迹长度
        numTrajectories         % 每次迭代收集的轨迹数量
        
        % 硬件配置
        useGPU                  % 是否使用GPU
        
        % 日志配置
        logDir                  % 日志保存目录
        evalFreq                % 评估频率 (迭代次数)
        numEvalEpisodes         % 评估时的回合数
        saveModelFreq           % 保存模型频率 (迭代次数)
    end
    
    methods
        function obj = PPOConfig()
            % 构造函数：设置默认配置
            
            % 环境配置
            obj.envName = 'CartPoleEnv';
            
            % 网络配置
            obj.actorLayerSizes = [64, 64];
            obj.criticLayerSizes = [64, 64];
            
            % 算法超参数
            obj.gamma = 0.99;
            obj.lambda = 0.95;
            obj.epsilon = 0.2;
            obj.entropyCoef = 0.01;
            obj.vfCoef = 0.5;
            obj.maxGradNorm = 0.5;
            
            % 优化器配置
            obj.actorLearningRate = 3e-4;
            obj.criticLearningRate = 3e-4;
            obj.momentum = 0.9;
            
            % 训练配置
            obj.numIterations = 100;
            obj.batchSize = 128;
            obj.epochsPerIter = 4;
            obj.trajectoryLen = 200;
            obj.numTrajectories = 10;
            
            % 硬件配置
            obj.useGPU = true;
            
            % 日志配置
            obj.logDir = 'logs';
            obj.evalFreq = 10;
            obj.numEvalEpisodes = 10;
            obj.saveModelFreq = 20;
        end
        
        function config = loadFromFile(obj, filePath)
            % 从文件加载配置
            %   filePath - 配置文件路径
            %   返回填充了配置的对象
            
            % 检查文件是否存在
            assert(exist(filePath, 'file') == 2, 'PPOConfig:配置文件不存在: %s', filePath);
            
            % 加载配置
            configData = load(filePath);
            
            % 获取所有属性名
            propNames = properties(obj);
            
            % 遍历所有属性并设置值
            for i = 1:length(propNames)
                propName = propNames{i};
                
                % 如果配置文件中有对应的字段，则设置该属性
                if isfield(configData, propName)
                    obj.(propName) = configData.(propName);
                end
            end
            
            config = obj;
        end
        
        function saveToFile(obj, filePath)
            % 将配置保存到文件
            %   filePath - 保存路径
            
            % 确保目录存在
            [dirPath, ~, ~] = fileparts(filePath);
            if ~isempty(dirPath) && ~exist(dirPath, 'dir')
                mkdir(dirPath);
            end
            
            % 将对象转换为结构体
            configStruct = obj.toStruct();
            
            % 保存到文件
            save(filePath, '-struct', 'configStruct');
            fprintf('配置已保存到: %s\n', filePath);
        end
        
        function configStruct = toStruct(obj)
            % 将配置对象转换为结构体
            
            configStruct = struct();
            propNames = properties(obj);
            
            for i = 1:length(propNames)
                propName = propNames{i};
                configStruct.(propName) = obj.(propName);
            end
        end
        
        function varargout = subsref(obj, s)
            % 重载下标引用操作，支持点符号和括号引用
            
            switch s(1).type
                case '.'
                    % 获取属性值
                    if length(s) == 1
                        % 直接访问属性
                        varargout{1} = obj.(s(1).subs);
                    else
                        % 级联访问
                        [varargout{1:nargout}] = subsref(obj.(s(1).subs), s(2:end));
                    end
                case '()'
                    % 括号引用，调用父类的处理
                    [varargout{1:nargout}] = builtin('subsref', obj, s);
                case '{}'
                    % 花括号引用，调用父类的处理
                    [varargout{1:nargout}] = builtin('subsref', obj, s);
            end
        end
        
        function obj = subsasgn(obj, s, val)
            % 重载下标赋值操作，支持点符号和括号赋值
            
            switch s(1).type
                case '.'
                    % 设置属性值
                    if length(s) == 1
                        % 直接设置属性
                        obj.(s(1).subs) = val;
                    else
                        % 级联设置
                        obj.(s(1).subs) = subsasgn(obj.(s(1).subs), s(2:end), val);
                    end
                case '()'
                    % 括号赋值，调用父类的处理
                    obj = builtin('subsasgn', obj, s, val);
                case '{}'
                    % 花括号赋值，调用父类的处理
                    obj = builtin('subsasgn', obj, s, val);
            end
        end
        
        function disp(obj)
            % 显示对象信息
            
            fprintf('PPO配置:\n');
            propNames = properties(obj);
            
            for i = 1:length(propNames)
                propName = propNames{i};
                propValue = obj.(propName);
                
                % 根据属性值类型格式化输出
                if ischar(propValue)
                    fprintf('  %s: %s\n', propName, propValue);
                elseif isnumeric(propValue) && isscalar(propValue)
                    fprintf('  %s: %.6g\n', propName, propValue);
                elseif isnumeric(propValue) && ~isscalar(propValue)
                    fprintf('  %s: [', propName);
                    fprintf('%.6g ', propValue);
                    fprintf(']\n');
                elseif islogical(propValue)
                    if propValue
                        fprintf('  %s: true\n', propName);
                    else
                        fprintf('  %s: false\n', propName);
                    end
                else
                    fprintf('  %s: %s\n', propName, class(propValue));
                end
            end
        end
    end
end
