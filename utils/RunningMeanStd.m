classdef RunningMeanStd < handle

    properties
        mean
        var
        count
        eps = 1e-8
        clip = 10
    end

    methods
        function obj = RunningMeanStd(dim, clip)
            obj.mean = zeros(dim, 1);
            obj.var = ones(dim, 1);
            obj.count = 1e-4;
            if nargin >= 2 && ~isempty(clip), obj.clip = clip; end
        end

        function update(obj, x)
            batchMean = mean(x, 2);
            batchVar = var(x, 1, 2);
            batchCount = size(x, 2);
            delta = batchMean - obj.mean;
            tot = obj.count + batchCount;
            obj.mean = obj.mean + delta * (batchCount / tot);
            mA = obj.var * obj.count;
            mB = batchVar * batchCount;
            M2 = mA + mB + (delta .^ 2) * (obj.count * batchCount / tot);
            obj.var = M2 / tot;
            obj.count = tot;
        end

        function y = normalize(obj, x)
            y = (x - obj.mean) ./ sqrt(obj.var + obj.eps);
            if ~isempty(obj.clip) && isfinite(obj.clip)
                y = max(min(y, obj.clip), -obj.clip);
            end
        end

        function s = toStruct(obj)
            s = struct('mean', obj.mean, 'var', obj.var, 'count', obj.count, ...
                       'eps', obj.eps, 'clip', obj.clip);
        end

        function fromStruct(obj, s)
            obj.mean = s.mean; obj.var = s.var; obj.count = s.count;
            obj.eps = s.eps; obj.clip = s.clip;
        end
    end
end
