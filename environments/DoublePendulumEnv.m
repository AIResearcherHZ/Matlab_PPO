classdef DoublePendulumEnv < handle
    % DoublePendulumEnv 双倒立摆环境（多智能体）
    %   一个需要多智能体协作的控制问题
    %   两个智能体各自控制一个摆杆，需要协同工作将摆杆保持在倒立位置
    %   注意：此环境为多智能体环境，不继承单智能体Environment基类
    
    properties
        % 环境参数
        g = 9.81            % 重力加速度 (m/s^2)
        l1 = 1.0            % 第一摆杆长度 (m)
        l2 = 1.0            % 第二摆杆长度 (m)
        m1 = 1.0            % 第一摆杆质量 (kg)
        m2 = 1.0            % 第二摆杆质量 (kg)
        maxTorque = 10.0    % 最大扭矩 (N·m)
        dt = 0.05           % 时间步长 (s)
        maxSteps = 200      % 最大步数
        
        % 状态变量
        theta1              % 第一摆杆角度
        theta2              % 第二摆杆角度
        dtheta1             % 第一摆杆角速度
        dtheta2             % 第二摆杆角速度
        
        % 目标状态
        targetTheta1 = pi   % 目标角度（倒立位置）
        targetTheta2 = pi   % 目标角度（倒立位置）
        
        % 环境状态
        stepCount           % 当前步数
        renderFig           % 绘图句柄
        renderAx            % 坐标轴句柄
        pendulum1Line       % 第一摆杆线
        pendulum2Line       % 第二摆杆线
        
        % 多智能体环境属性
        numAgents = 2       % 智能体数量
    end
    
    properties (Dependent)
        % 观察空间和动作空间维度（依赖属性）
        observationSize     % 每个智能体的观察空间大小
        actionSize          % 每个智能体的动作空间大小
    end
    
    methods
        function obj = DoublePendulumEnv()
            % 构造函数
        end
        
        function size = get.observationSize(obj)
            % 返回每个智能体的观察空间维度
            size = [4, 4];
        end
        
        function size = get.actionSize(obj)
            % 返回每个智能体的动作空间维度
            size = [1, 1];
        end
        
        function [agentObs, jointObs] = reset(obj)
            % 重置环境到初始状态
            
            % 随机初始位置（接近但不完全在倒立位置）
            obj.theta1 = pi + (rand - 0.5) * 0.6;  % 近似倒立位置
            obj.theta2 = pi + (rand - 0.5) * 0.6;  % 近似倒立位置
            obj.dtheta1 = (rand - 0.5) * 0.2;      % 小的随机初始速度
            obj.dtheta2 = (rand - 0.5) * 0.2;      % 小的随机初始速度
            
            % 重置步数计数器
            obj.stepCount = 0;
            
            % 返回观察
            agentObs = obj.getAgentObservations();
            jointObs = obj.getJointObservation();
        end
        
        function [nextAgentObs, nextJointObs, reward, done, info] = step(obj, actions)
            % 执行动作并返回下一个状态、奖励和完成标志
            
            % 限制动作在有效范围内
            torque1 = min(max(actions{1}, -obj.maxTorque), obj.maxTorque);
            torque2 = min(max(actions{2}, -obj.maxTorque), obj.maxTorque);
            
            % 使用动力学方程更新状态
            [obj.theta1, obj.theta2, obj.dtheta1, obj.dtheta2] = ...
                obj.dynamics(obj.theta1, obj.theta2, obj.dtheta1, obj.dtheta2, torque1, torque2);
            
            % 归一化角度到 [-pi, pi)
            obj.theta1 = wrapToPi(obj.theta1);
            obj.theta2 = wrapToPi(obj.theta2);
            
            % 计算奖励
            reward = obj.calculateReward();
            
            % 更新步数
            obj.stepCount = obj.stepCount + 1;
            
            % 检查是否完成
            done = obj.stepCount >= obj.maxSteps;
            
            % 获取观察
            nextAgentObs = obj.getAgentObservations();
            nextJointObs = obj.getJointObservation();
            
            % 准备额外信息
            info = struct();
            info.theta1 = obj.theta1;
            info.theta2 = obj.theta2;
            info.dtheta1 = obj.dtheta1;
            info.dtheta2 = obj.dtheta2;
        end
        
        function [theta1_next, theta2_next, dtheta1_next, dtheta2_next] = dynamics(obj, theta1, theta2, dtheta1, dtheta2, torque1, torque2)
            % 双倒立摆的动力学方程
            % 使用欧拉法进行积分
            
            % 参数简写
            g = obj.g;
            m1 = obj.m1;
            m2 = obj.m2;
            l1 = obj.l1;
            l2 = obj.l2;
            dt = obj.dt;
            
            % 计算动力学
            % 简化的双摆动力学方程（实际实现中应该使用完整的拉格朗日方程）
            
            % 计算辅助变量
            delta = theta2 - theta1;
            sdelta = sin(delta);
            cdelta = cos(delta);
            
            % 计算分母项
            den1 = (m1 + m2) * l1 - m2 * l1 * cdelta * cdelta;
            den2 = (l2 / l1) * den1;
            
            % 计算角加速度
            ddtheta1 = ((m2 * l2 * dtheta2^2 * sdelta) - (m2 * g * sin(theta2) * cdelta) + ...
                        (m2 * l1 * dtheta1^2 * sdelta * cdelta) + ...
                        (torque1 + (m1 + m2) * g * sin(theta1))) / den1;
            
            ddtheta2 = ((-l1 / l2) * dtheta1^2 * sdelta - (g / l2) * sin(theta2) + ...
                        (torque2 / (m2 * l2)) - (cdelta * ddtheta1)) / (1 - cdelta^2 / (den2));
            
            % 使用欧拉法更新
            dtheta1_next = dtheta1 + ddtheta1 * dt;
            dtheta2_next = dtheta2 + ddtheta2 * dt;
            theta1_next = theta1 + dtheta1_next * dt;
            theta2_next = theta2 + dtheta2_next * dt;
        end
        
        function reward = calculateReward(obj)
            % 计算奖励值
            
            % 角度偏差，使用余弦相似度来衡量接近目标的程度
            angle_error1 = 1 - cos(obj.theta1 - obj.targetTheta1);
            angle_error2 = 1 - cos(obj.theta2 - obj.targetTheta2);
            
            % 速度惩罚
            velocity_penalty1 = 0.1 * obj.dtheta1^2;
            velocity_penalty2 = 0.1 * obj.dtheta2^2;
            
            % 总奖励
            reward = -(angle_error1 + angle_error2 + velocity_penalty1 + velocity_penalty2);
        end
        
        function agentObservations = getAgentObservations(obj)
            % 获取每个智能体的观察
            
            % 智能体1的观察：自己的角度和角速度，以及对方的状态
            obs1 = [
                sin(obj.theta1);
                cos(obj.theta1);
                obj.dtheta1 / 10.0;  % 归一化
                sin(obj.theta2 - obj.theta1)  % 相对角度信息
            ];
            
            % 智能体2的观察：自己的角度和角速度，以及对方的状态
            obs2 = [
                sin(obj.theta2);
                cos(obj.theta2);
                obj.dtheta2 / 10.0;  % 归一化
                sin(obj.theta2 - obj.theta1)  % 相对角度信息
            ];
            
            % 包装为cell数组
            agentObservations = {obs1, obs2};
        end
        
        function jointObs = getJointObservation(obj)
            % 获取联合观察（用于中央评论家）
            
            jointObs = [
                sin(obj.theta1);
                cos(obj.theta1);
                obj.dtheta1 / 10.0;
                sin(obj.theta2);
                cos(obj.theta2);
                obj.dtheta2 / 10.0;
                sin(obj.theta2 - obj.theta1);
                cos(obj.theta2 - obj.theta1)
            ];
        end
        
        function render(obj)
            % 可视化双倒立摆系统
            
            % 第一次调用时创建图形
            if isempty(obj.renderFig) || ~isvalid(obj.renderFig)
                obj.renderFig = figure('Name', '双倒立摆', 'NumberTitle', 'off');
                obj.renderAx = axes('XLim', [-2.5, 2.5], 'YLim', [-2.5, 2.5]);
                hold(obj.renderAx, 'on');
                grid(obj.renderAx, 'on');
                axis(obj.renderAx, 'equal');
                xlabel(obj.renderAx, 'X');
                ylabel(obj.renderAx, 'Y');
                title(obj.renderAx, '双倒立摆系统');
                
                % 创建摆杆线条对象
                obj.pendulum1Line = line(obj.renderAx, [0, 0], [0, 0], 'LineWidth', 3, 'Color', 'blue');
                obj.pendulum2Line = line(obj.renderAx, [0, 0], [0, 0], 'LineWidth', 3, 'Color', 'red');
            end
            
            % 计算摆杆端点坐标
            x0 = 0;
            y0 = 0;
            x1 = x0 + obj.l1 * sin(obj.theta1);
            y1 = y0 - obj.l1 * cos(obj.theta1);
            x2 = x1 + obj.l2 * sin(obj.theta2);
            y2 = y1 - obj.l2 * cos(obj.theta2);
            
            % 更新线条位置
            set(obj.pendulum1Line, 'XData', [x0, x1], 'YData', [y0, y1]);
            set(obj.pendulum2Line, 'XData', [x1, x2], 'YData', [y1, y2]);
            
            % 更新标题显示当前步数
            title(obj.renderAx, sprintf('双倒立摆系统 - 步数: %d', obj.stepCount));
            
            % 刷新图形
            drawnow;
        end
        
        function result = isDiscreteAction(obj, agentIdx)
            % 判断动作空间是否离散
            result = false;
        end
        
        function close(obj)
            % 关闭环境和释放资源
            if ~isempty(obj.renderFig) && isvalid(obj.renderFig)
                close(obj.renderFig);
            end
        end
    end
end
