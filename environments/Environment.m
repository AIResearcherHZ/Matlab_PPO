classdef Environment < handle

    properties
        obsDim = 0
        actDim = 0
        isDiscrete = false
    end

    methods
        function obs = reset(obj)
            error('Environment:notImplemented', 'reset() must be overridden');
        end

        function [nextObs, reward, done, info] = step(obj, action)
            error('Environment:notImplemented', 'step() must be overridden');
        end

        function render(~)
        end

        function seed(~, s)
            rand('seed', s); randn('seed', s);
        end
    end
end
