classdef PendulumEnv < Environment

    properties
        g = 10.0
        m = 1.0
        l = 1.0
        dt = 0.05
        maxTorque = 2.0
        maxSpeed = 8.0
        maxSteps = 200
        th = 0
        thdot = 0
        steps = 0
    end

    methods
        function obj = PendulumEnv()
            obj.obsDim = 3;
            obj.actDim = 1;
            obj.isDiscrete = false;
        end

        function obs = reset(obj)
            obj.th = (rand() * 2 - 1) * pi;
            obj.thdot = (rand() * 2 - 1);
            obj.steps = 0;
            obs = obj.getObs();
        end

        function [obs, reward, done, info] = step(obj, action)
            u = max(min(action(1), 1), -1) * obj.maxTorque;
            thNorm = obj.angleNormalize(obj.th);
            reward = -(thNorm ^ 2 + 0.1 * obj.thdot ^ 2 + 0.001 * u ^ 2);

            newthdot = obj.thdot + (3 * obj.g / (2 * obj.l) * sin(obj.th) ...
                + 3.0 / (obj.m * obj.l ^ 2) * u) * obj.dt;
            newthdot = max(min(newthdot, obj.maxSpeed), -obj.maxSpeed);
            obj.th = obj.th + newthdot * obj.dt;
            obj.thdot = newthdot;
            obj.steps = obj.steps + 1;

            truncated = obj.steps >= obj.maxSteps;
            done = truncated;
            obs = obj.getObs();
            info = struct('truncated', truncated);
        end
    end

    methods (Access = private)
        function obs = getObs(obj)
            obs = [cos(obj.th); sin(obj.th); obj.thdot];
        end

        function a = angleNormalize(~, x)
            a = mod(x + pi, 2 * pi) - pi;
        end
    end
end
