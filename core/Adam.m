classdef Adam < handle

    properties
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-5
        t = 0
        m = []
        v = []
    end

    methods
        function obj = Adam(beta1, beta2, eps)
            if nargin >= 1 && ~isempty(beta1), obj.beta1 = beta1; end
            if nargin >= 2 && ~isempty(beta2), obj.beta2 = beta2; end
            if nargin >= 3 && ~isempty(eps),   obj.eps   = eps;   end
        end

        function params = step(obj, params, grad, lr)
            f = fieldnames(params);
            if isempty(obj.m)
                obj.m = struct(); obj.v = struct();
                for k = 1:numel(f)
                    obj.m.(f{k}) = zeros(size(params.(f{k})));
                    obj.v.(f{k}) = zeros(size(params.(f{k})));
                end
            end
            obj.t = obj.t + 1;
            bc1 = 1 - obj.beta1 ^ obj.t;
            bc2 = 1 - obj.beta2 ^ obj.t;
            for k = 1:numel(f)
                name = f{k};
                g = grad.(name);
                obj.m.(name) = obj.beta1 * obj.m.(name) + (1 - obj.beta1) * g;
                obj.v.(name) = obj.beta2 * obj.v.(name) + (1 - obj.beta2) * (g .^ 2);
                mhat = obj.m.(name) / bc1;
                vhat = obj.v.(name) / bc2;
                params.(name) = params.(name) - lr * mhat ./ (sqrt(vhat) + obj.eps);
            end
        end
    end
end
