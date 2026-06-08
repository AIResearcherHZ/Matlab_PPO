classdef CartPoleEnv < Environment

    properties
        gravity = 9.8
        massCart = 1.0
        massPole = 0.1
        length = 0.5
        forceMag = 10.0
        tau = 0.02
        xThreshold = 2.4
        thetaThreshold = 0.20944
        maxSteps = 500
        state
        steps = 0
    end

    methods
        function obj = CartPoleEnv()
            obj.obsDim = 4;
            obj.actDim = 2;
            obj.isDiscrete = true;
            obj.state = zeros(4, 1);
        end

        function obs = reset(obj)
            obj.state = 0.1 * rand(4, 1) - 0.05;
            obj.steps = 0;
            obs = obj.state;
        end

        function [obs, reward, done, info] = step(obj, action)
            force = (action == 2) * obj.forceMag - (action == 1) * obj.forceMag;
            x = obj.state(1); xDot = obj.state(2);
            theta = obj.state(3); thetaDot = obj.state(4);

            totalMass = obj.massCart + obj.massPole;
            poleMassLength = obj.massPole * obj.length;
            cosT = cos(theta); sinT = sin(theta);
            temp = (force + poleMassLength * thetaDot ^ 2 * sinT) / totalMass;
            thetaAcc = (obj.gravity * sinT - cosT * temp) / ...
                (obj.length * (4.0 / 3.0 - obj.massPole * cosT ^ 2 / totalMass));
            xAcc = temp - poleMassLength * thetaAcc * cosT / totalMass;

            x = x + obj.tau * xDot;          xDot = xDot + obj.tau * xAcc;
            theta = theta + obj.tau * thetaDot; thetaDot = thetaDot + obj.tau * thetaAcc;
            obj.state = [x; xDot; theta; thetaDot];
            obj.steps = obj.steps + 1;

            terminated = abs(x) > obj.xThreshold || abs(theta) > obj.thetaThreshold;
            truncated = obj.steps >= obj.maxSteps;
            done = terminated || truncated;
            reward = 1.0;
            obs = obj.state;
            info = struct('truncated', truncated && ~terminated);
        end

        function render(obj)
            persistent fig ax
            if isempty(fig) || ~ishandle(fig)
                fig = figure('Name', 'CartPole');
                ax = axes('XLim', [-obj.xThreshold - 1, obj.xThreshold + 1], 'YLim', [-1, 2]);
                hold(ax, 'on'); grid(ax, 'on');
            end
            cla(ax);
            x = obj.state(1); theta = obj.state(3);
            line(ax, [x - 0.25, x + 0.25], [0, 0], 'Color', 'b', 'LineWidth', 6);
            line(ax, [x, x + 2 * obj.length * sin(theta)], ...
                     [0, 2 * obj.length * cos(theta)], 'Color', 'r', 'LineWidth', 3);
            drawnow;
        end
    end
end
