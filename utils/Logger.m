classdef Logger < handle
    % Logger 训练日志记录器
    %   用于记录和可视化训练过程中的各种指标
    
    properties
        logDir          % 日志保存目录
        envName         % 环境名称
        trainStats      % 训练统计数据
        evalStats       % 评估统计数据
        saveFigs        % 是否保存图表
    end
    
    methods
        function obj = Logger(logDir, envName)
            % 构造函数：初始化日志记录器
            %   logDir - 日志保存目录
            %   envName - 环境名称（可选）
            
            % 验证输入
            if nargin < 2
                envName = 'unknown';
            end
            
            % 设置日志目录和环境名称
            obj.logDir = logDir;
            obj.envName = envName;
            
            % 创建日志目录（如果不存在）
            if ~exist(logDir, 'dir')
                mkdir(logDir);
                fprintf('创建日志目录: %s\n', logDir);
            end
            
            % 初始化统计数据
            obj.trainStats = struct();
            obj.trainStats.iterations = [];
            obj.trainStats.actorLoss = [];
            obj.trainStats.criticLoss = [];
            obj.trainStats.entropyLoss = [];
            obj.trainStats.totalLoss = [];
            
            obj.evalStats = struct();
            obj.evalStats.iterations = [];
            obj.evalStats.returns = [];
            obj.evalStats.lengths = [];
            
            % 默认保存图表
            obj.saveFigs = true;
        end
        
        function logIteration(obj, iteration, metrics)
            % 记录训练迭代的指标
            %   iteration - 当前迭代次数
            %   metrics - 包含各种指标的结构体
            
            % 验证输入
            assert(isstruct(metrics), 'Logger:指标必须是结构体');
            assert(isfield(metrics, 'actorLoss'), 'Logger:缺少actorLoss字段');
            assert(isfield(metrics, 'criticLoss'), 'Logger:缺少criticLoss字段');
            
            % 添加到训练统计数据
            obj.trainStats.iterations(end+1) = iteration;
            obj.trainStats.actorLoss(end+1) = metrics.actorLoss;
            obj.trainStats.criticLoss(end+1) = metrics.criticLoss;
            obj.trainStats.entropyLoss(end+1) = metrics.entropyLoss;
            obj.trainStats.totalLoss(end+1) = metrics.totalLoss;
            
            % 打印当前迭代的指标
            fprintf('迭代 %d: Actor损失 = %.4f, Critic损失 = %.4f, 熵损失 = %.4f, 总损失 = %.4f\n', ...
                iteration, metrics.actorLoss, metrics.criticLoss, metrics.entropyLoss, metrics.totalLoss);
            
            % 每10次迭代绘制并保存训练曲线
            if mod(iteration, 10) == 0
                obj.plotTrainingCurves();
            end
        end
        
        function logEvaluation(obj, iteration, evalResult)
            % 记录评估结果
            %   iteration - 当前迭代次数
            %   evalResult - 评估结果
            
            % 验证输入
            assert(isstruct(evalResult), 'Logger:评估结果必须是结构体');
            assert(isfield(evalResult, 'meanReturn'), 'Logger:缺少meanReturn字段');
            
            % 添加到评估统计数据
            obj.evalStats.iterations(end+1) = iteration;
            obj.evalStats.returns(end+1) = evalResult.meanReturn;
            obj.evalStats.lengths(end+1) = evalResult.meanLength;
            
            % 打印评估结果
            fprintf('评估 (迭代 %d): 平均回报 = %.2f ± %.2f, 最小 = %.2f, 最大 = %.2f, 平均长度 = %.2f\n', ...
                iteration, evalResult.meanReturn, evalResult.stdReturn, ...
                evalResult.minReturn, evalResult.maxReturn, evalResult.meanLength);
            
            % 绘制并保存评估曲线
            obj.plotEvaluationCurves();
        end
        
        function plotTrainingCurves(obj)
            % 绘制训练曲线
            
            % 创建图形
            figure('Name', ['训练曲线 - ', obj.envName], 'Position', [100, 100, 1200, 800]);
            
            % 绘制Actor损失
            subplot(2, 2, 1);
            plot(obj.trainStats.iterations, obj.trainStats.actorLoss, 'b-', 'LineWidth', 1.5);
            title('Actor损失');
            xlabel('迭代次数');
            ylabel('损失值');
            grid on;
            
            % 绘制Critic损失
            subplot(2, 2, 2);
            plot(obj.trainStats.iterations, obj.trainStats.criticLoss, 'r-', 'LineWidth', 1.5);
            title('Critic损失');
            xlabel('迭代次数');
            ylabel('损失值');
            grid on;
            
            % 绘制熵损失
            subplot(2, 2, 3);
            plot(obj.trainStats.iterations, obj.trainStats.entropyLoss, 'g-', 'LineWidth', 1.5);
            title('熵损失');
            xlabel('迭代次数');
            ylabel('损失值');
            grid on;
            
            % 绘制总损失
            subplot(2, 2, 4);
            plot(obj.trainStats.iterations, obj.trainStats.totalLoss, 'm-', 'LineWidth', 1.5);
            title('总损失');
            xlabel('迭代次数');
            ylabel('损失值');
            grid on;
            
            % 调整图形布局
            sgtitle(['训练曲线 - ', obj.envName], 'FontSize', 16);
            
            % 保存图形
            if obj.saveFigs
                saveas(gcf, fullfile(obj.logDir, 'training_curves.png'));
            end
        end
        
        function plotEvaluationCurves(obj)
            % 绘制评估曲线
            
            % 如果没有评估数据，直接返回
            if isempty(obj.evalStats.iterations)
                return;
            end
            
            % 创建图形
            figure('Name', ['评估曲线 - ', obj.envName], 'Position', [100, 100, 1000, 500]);
            
            % 绘制平均回报
            subplot(1, 2, 1);
            plot(obj.evalStats.iterations, obj.evalStats.returns, 'b-o', 'LineWidth', 1.5);
            title('评估平均回报');
            xlabel('迭代次数');
            ylabel('平均回报');
            grid on;
            
            % 绘制平均长度
            subplot(1, 2, 2);
            plot(obj.evalStats.iterations, obj.evalStats.lengths, 'r-o', 'LineWidth', 1.5);
            title('评估平均长度');
            xlabel('迭代次数');
            ylabel('平均长度');
            grid on;
            
            % 调整图形布局
            sgtitle(['评估曲线 - ', obj.envName], 'FontSize', 16);
            
            % 保存图形
            if obj.saveFigs
                saveas(gcf, fullfile(obj.logDir, 'evaluation_curves.png'));
            end
        end
        
        function saveTrainingData(obj)
            % 保存训练数据
            trainData = obj.trainStats;
            evalData = obj.evalStats;
            save(fullfile(obj.logDir, 'training_data.mat'), 'trainData', 'evalData');
            fprintf('训练数据已保存到: %s\n', fullfile(obj.logDir, 'training_data.mat'));
        end
    end
end
