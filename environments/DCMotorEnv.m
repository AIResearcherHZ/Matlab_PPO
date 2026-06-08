classdef DCMotorEnv < Environment

    properties
        J = 0.01
        b = 0.1
        K = 0.01
        R = 1.0
        L = 0.5
        dt = 0.01
        maxVoltage = 24.0
        maxCurrent = 10.0
        maxSpeed = 50.0
        maxSteps = 500
        targetAngle
        state
        steps = 0
    end

    methods
        function obj = DCMotorEnv()
            obj.obsDim = 3;
            obj.actDim = 1;
            obj.isDiscrete = false;
            obj.state = zeros(3, 1);
            obj.targetAngle = 2 * pi * rand();
        end

        function obs = reset(obj)
            obj.state = [2 * pi * rand(); 0; 0];
            obj.targetAngle = 2 * pi * rand();
            obj.steps = 0;
            obs = obj.getObs();
        end

        function [obs, reward, done, info] = step(obj, action)
            V = max(min(action(1), 1), -1) * obj.maxVoltage;
            angle = obj.state(1); omega = obj.state(2); i = obj.state(3);

            angle = mod(angle + obj.dt * omega, 2 * pi);
            omega = omega + obj.dt * ((obj.K * i - obj.b * omega) / obj.J);
            i = i + obj.dt * ((V - obj.R * i - obj.K * omega) / obj.L);
            omega = max(min(omega, obj.maxSpeed), -obj.maxSpeed);
            i = max(min(i, obj.maxCurrent), -obj.maxCurrent);
            obj.state = [angle; omega; i];
            obj.steps = obj.steps + 1;

            angleErr = obj.wrapToPi(angle - obj.targetAngle);
            reward = -(angleErr ^ 2) ...
                - 0.01 * (omega / obj.maxSpeed) ^ 2 ...
                - 0.001 * (V / obj.maxVoltage) ^ 2;

            truncated = obj.steps >= obj.maxSteps;
            done = truncated;
            obs = obj.getObs();
            info = struct('truncated', truncated, 'angleError', angleErr);
        end
    end

    methods (Access = private)
        function obs = getObs(obj)
            angleErr = obj.wrapToPi(obj.state(1) - obj.targetAngle);
            obs = [angleErr; obj.state(2) / obj.maxSpeed; obj.state(3) / obj.maxCurrent];
        end

        function y = wrapToPi(~, x)
            y = atan2(sin(x), cos(x));
        end
    end
end
