classdef Logger < handle
    % Logger 训练日志记录器
    %   用于记录训练过程中的各种指标和性能数据
    
    properties
        logDir          % 日志目录
        envName         % 环境名称
        logFile         % 日志文件路径
        metricsHistory  % 指标历史记录
    end
    
    methods
        function obj = Logger(logDir, envName)
            % 构造函数
            %   logDir - 日志目录
            %   envName - 环境名称
            
            obj.logDir = logDir;
            obj.envName = envName;
            
            % 创建日志目录
            if ~exist(obj.logDir, 'dir')
                mkdir(obj.logDir);
            end
            
            % 设置日志文件路径
            timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
            obj.logFile = fullfile(obj.logDir, sprintf('%s_training_%s.log', obj.envName, timestamp));
            
            % 初始化指标历史
            obj.metricsHistory = struct();
            obj.metricsHistory.iteration = [];
            obj.metricsHistory.actorLoss = [];
            obj.metricsHistory.criticLoss = [];
            obj.metricsHistory.entropyLoss = [];
            obj.metricsHistory.totalLoss = [];
            obj.metricsHistory.meanReturn = [];
            obj.metricsHistory.timestamp = [];
            
            % 写入日志头部信息
            obj.writeLog(sprintf('开始训练 %s 环境', obj.envName));
            obj.writeLog(sprintf('日志目录: %s', obj.logDir));
            obj.writeLog(repmat('=', 1, 50));
        end
        
        function logIteration(obj, iteration, metrics)
            % 记录单次迭代的指标
            %   iteration - 迭代次数
            
            % 更新历史记录
            obj.metricsHistory.iteration(end+1) = iteration;
            obj.metricsHistory.actorLoss(end+1) = metrics.actorLoss;
            obj.metricsHistory.criticLoss(end+1) = metrics.criticLoss;
            obj.metricsHistory.entropyLoss(end+1) = metrics.entropyLoss;
            obj.metricsHistory.totalLoss(end+1) = metrics.totalLoss;
            obj.metricsHistory.timestamp(end+1) = now;
            
            % 如果有回报信息，也记录下来
            if isfield(metrics, 'meanReturn')
                obj.metricsHistory.meanReturn(end+1) = metrics.meanReturn;
            else
                obj.metricsHistory.meanReturn(end+1) = NaN;
            end
            
            % 写入日志
            logMessage = sprintf('迭代 %d: Actor损失=%.4f, Critic损失=%.4f, 熵损失=%.4f, 总损失=%.4f', ...
                iteration, metrics.actorLoss, metrics.criticLoss, metrics.entropyLoss, metrics.totalLoss);
            
            if isfield(metrics, 'meanReturn') && ~isnan(metrics.meanReturn)
                logMessage = sprintf('%s, 平均回报=%.2f', logMessage, metrics.meanReturn);
            end
            
            obj.writeLog(logMessage);
            
            % 控制台输出
            fprintf('%s\n', logMessage);
        end
        
        function writeLog(obj, message)
            % 写入日志文件
            %   message - 日志消息
            
            timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
            logEntry = sprintf('[%s] %s\n', timestamp, message);
            
            % 写入文件
            fid = fopen(obj.logFile, 'a');
            if fid ~= -1
                fprintf(fid, '%s', logEntry);
                fclose(fid);
            else
                warning('无法写入日志文件: %s', obj.logFile);
            end
        end
        
        function plotMetrics(obj)
            % 绘制训练指标图表
            
            if isempty(obj.metricsHistory.iteration)
                warning('没有可绘制的指标数据');
                return;
            end
            
            figure('Name', sprintf('%s 训练指标', obj.envName), 'Position', [100, 100, 1200, 800]);
            
            % 损失函数图
            subplot(2, 2, 1);
            plot(obj.metricsHistory.iteration, obj.metricsHistory.actorLoss, 'b-', 'LineWidth', 2);
            hold on;
            plot(obj.metricsHistory.iteration, obj.metricsHistory.criticLoss, 'r-', 'LineWidth', 2);
            plot(obj.metricsHistory.iteration, obj.metricsHistory.entropyLoss, 'g-', 'LineWidth', 2);
            xlabel('迭代次数');
            ylabel('损失值');
            title('训练损失');
            legend('Actor损失', 'Critic损失', '熵损失', 'Location', 'best');
            grid on;
            
            % 总损失图
            subplot(2, 2, 2);
            plot(obj.metricsHistory.iteration, obj.metricsHistory.totalLoss, 'k-', 'LineWidth', 2);
            xlabel('迭代次数');
            ylabel('总损失');
            title('总损失');
            grid on;
            
            % 平均回报图（如果有数据）
            if any(~isnan(obj.metricsHistory.meanReturn))
                subplot(2, 2, 3);
                validIdx = ~isnan(obj.metricsHistory.meanReturn);
                plot(obj.metricsHistory.iteration(validIdx), obj.metricsHistory.meanReturn(validIdx), 'm-', 'LineWidth', 2);
                xlabel('迭代次数');
                ylabel('平均回报');
                title('平均回报');
                grid on;
            end
            
            % 保存图表
            savePath = fullfile(obj.logDir, sprintf('%s_training_metrics.png', obj.envName));
            saveas(gcf, savePath);
            obj.writeLog(sprintf('训练指标图表已保存到: %s', savePath));
        end
        
        function saveMetrics(obj)
            % 保存指标数据到文件
            
            metricsFile = fullfile(obj.logDir, sprintf('%s_metrics.mat', obj.envName));
            metricsHistory = obj.metricsHistory; %#ok<NASGU>
            save(metricsFile, 'metricsHistory');
            
            obj.writeLog(sprintf('指标数据已保存到: %s', metricsFile));
        end
    end
end