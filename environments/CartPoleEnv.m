classdef CartPoleEnv < Environment
    % CartPoleEnv 倒立摆环境
    %   基于经典控制问题的倒立摆环境实现
    
    properties
        % 环境规格
        observationSize = 4     % 状态空间维度：[x, x_dot, theta, theta_dot]
        actionSize = 1          % 动作空间维度：左右力
        isDiscrete = true       % 是否为离散动作空间
        
        % 物理参数
        gravity = 9.8           % 重力加速度 (m/s^2)
        massCart = 1.0          % 小车质量 (kg)
        massPole = 0.1          % 杆质量 (kg)
        totalMass               % 总质量 (kg)
        length = 0.5            % 杆长的一半 (m)
        poleMassLength          % 杆质量长度
        forceMag = 10.0         % 施加在小车上的力大小 (N)
        tau = 0.02              % 时间步长 (s)
        
        % 界限
        xThreshold = 2.4        % 小车位置阈值
        thetaThreshold = 12     % 杆角度阈值 (度)
        
        % 当前状态
        state                   % [x, x_dot, theta, theta_dot]
        
        % 回合信息
        steps = 0               % 当前回合步数
        maxSteps = 500          % 最大步数
        
        % 可视化
        renderFig               % 图形句柄
        renderAx                % 坐标轴句柄
        cartWidth = 0.5         % 小车宽度
        cartHeight = 0.3        % 小车高度
        poleWidth = 0.1         % 杆宽度
    end
    
    methods
        function obj = CartPoleEnv()
            % 构造函数：初始化倒立摆环境
            
            % 计算物理参数
            obj.totalMass = obj.massCart + obj.massPole;
            obj.poleMassLength = obj.massPole * obj.length;
            
            % 初始化状态
            obj.state = zeros(4, 1);
            
            % 随机种子
            rng('shuffle');
        end
        
        function observation = reset(obj)
            % 重置环境到初始状态
            %   返回初始观察
            
            % 随机初始化状态，±0.05范围内的小扰动
            obj.state = 0.1 * rand(4, 1) - 0.05;
            obj.steps = 0;
            
            % 返回观察
            observation = obj.state;
        end
        
        function [nextObs, reward, done, info] = step(obj, action)
            % 执行动作并返回新的状态
            %   action - 要执行的动作 (0: 左推，1: 右推)
            %   返回值：
            %     nextObs - 新的观察
            %     reward - 获得的奖励
            %     done - 是否回合结束
            %     info - 附加信息
            
            % 如果是标量，直接使用；如果是one-hot向量，提取索引
            if isscalar(action)
                actionIdx = action;
            else
                % 处理one-hot编码的动作
                [~, actionIdx] = max(action);
                actionIdx = actionIdx - 1; % 转换为0或1
            end
            
            % 验证动作
            assert(actionIdx == 0 || actionIdx == 1, 'CartPoleEnv:动作必须是0或1');
            
            % 将动作转换为力
            force = (actionIdx * 2 - 1) * obj.forceMag;  % 0 -> -10, 1 -> 10
            
            % 解包状态
            x = obj.state(1);
            xDot = obj.state(2);
            theta = obj.state(3);
            thetaDot = obj.state(4);
            
            % 角度需要从弧度转换为度进行检查
            thetaDeg = rad2deg(theta);
            
            % 检查是否超出界限
            done = abs(x) > obj.xThreshold || ...
                   abs(thetaDeg) > obj.thetaThreshold || ...
                   obj.steps >= obj.maxSteps;
            
            % 如果回合未结束，计算下一个状态
            if ~done
                % 物理计算
                cosTheta = cos(theta);
                sinTheta = sin(theta);
                
                % 计算加速度
                temp = (force + obj.poleMassLength * thetaDot^2 * sinTheta) / obj.totalMass;
                thetaAcc = (obj.gravity * sinTheta - cosTheta * temp) / ...
                          (obj.length * (4.0/3.0 - obj.massPole * cosTheta^2 / obj.totalMass));
                xAcc = temp - obj.poleMassLength * thetaAcc * cosTheta / obj.totalMass;
                
                % 欧拉积分更新
                x = x + obj.tau * xDot;
                xDot = xDot + obj.tau * xAcc;
                theta = theta + obj.tau * thetaDot;
                thetaDot = thetaDot + obj.tau * thetaAcc;
                
                % 更新状态
                obj.state = [x; xDot; theta; thetaDot];
                obj.steps = obj.steps + 1;
            end
            
            % 设置奖励
            if done && obj.steps < obj.maxSteps
                % 如果因为超出界限而结束（不是因为达到最大步数），给予惩罚
                reward = 0;
            else
                % 存活奖励
                reward = 1.0;
            end
            
            % 返回结果
            nextObs = obj.state;
            info = struct('steps', obj.steps);
        end
        
        function render(obj)
            % 渲染当前环境状态
            
            % 如果没有图形，创建一个
            if isempty(obj.renderFig) || ~isvalid(obj.renderFig)
                obj.renderFig = figure('Name', '倒立摆', 'Position', [100, 100, 800, 400]);
                obj.renderAx = axes('XLim', [-obj.xThreshold - 1, obj.xThreshold + 1], ...
                                   'YLim', [-1, 2]);
                title('倒立摆');
                xlabel('位置');
                ylabel('高度');
                hold(obj.renderAx, 'on');
                grid(obj.renderAx, 'on');
            end
            
            % 清除当前轴
            cla(obj.renderAx);
            
            % 获取当前状态
            x = obj.state(1);
            theta = obj.state(3);
            
            % 计算杆的端点
            poleX = [x, x + 2 * obj.length * sin(theta)];
            poleY = [0, 2 * obj.length * cos(theta)];
            
            % 绘制小车
            cartX = [x - obj.cartWidth/2, x + obj.cartWidth/2, x + obj.cartWidth/2, x - obj.cartWidth/2];
            cartY = [0, 0, obj.cartHeight, obj.cartHeight];
            fill(obj.renderAx, cartX, cartY, 'b');
            
            % 绘制杆
            line(obj.renderAx, poleX, poleY, 'Color', 'r', 'LineWidth', 3);
            
            % 绘制地面
            line(obj.renderAx, [-obj.xThreshold - 1, obj.xThreshold + 1], [0, 0], 'Color', 'k', 'LineWidth', 2);
            
            % 更新图形
            drawnow;
        end
    end
end
