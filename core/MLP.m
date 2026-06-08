classdef MLP < handle

    properties
        sizes
        nLayers
        useLayerNorm
        lnEps = 1e-5
        learn
    end

    methods
        function obj = MLP(sizes, opts)
            if nargin < 2, opts = struct(); end
            hiddenGain = getfielddef(opts, 'hiddenGain', sqrt(2));
            outGain    = getfielddef(opts, 'outGain', 1.0);
            obj.useLayerNorm = getfielddef(opts, 'useLayerNorm', false);

            obj.sizes = sizes(:)';
            obj.nLayers = numel(sizes) - 1;
            obj.learn = struct();
            for i = 1:obj.nLayers
                nin = sizes(i); nout = sizes(i + 1);
                if i < obj.nLayers
                    g = hiddenGain;
                else
                    g = outGain;
                end
                obj.learn.(sprintf('W%d', i)) = orthoInit(nout, nin, g);
                obj.learn.(sprintf('b%d', i)) = zeros(nout, 1);
                if obj.useLayerNorm && i < obj.nLayers
                    obj.learn.(sprintf('lg%d', i)) = ones(nout, 1);
                    obj.learn.(sprintf('lo%d', i)) = zeros(nout, 1);
                end
            end
        end

        function [y, cache] = forward(obj, x)
            L = obj.nLayers;
            h = x;
            cache.x = x;
            cache.yl = cell(L, 1);
            cache.ln = cell(L, 1);
            cache.h  = cell(L, 1);
            for i = 1:L
                W = obj.learn.(sprintf('W%d', i));
                b = obj.learn.(sprintf('b%d', i));
                z = W * h + b;
                if i < L
                    if obj.useLayerNorm
                        g = obj.learn.(sprintf('lg%d', i));
                        o = obj.learn.(sprintf('lo%d', i));
                        [yl, lc] = obj.lnForward(z, g, o);
                        cache.ln{i} = lc;
                        cache.yl{i} = yl;
                        h = tanh(yl);
                    else
                        cache.yl{i} = z;
                        h = tanh(z);
                    end
                else
                    h = z;
                end
                cache.h{i} = h;
            end
            y = h;
        end

        function [grad, dx] = backward(obj, cache, dy)
            L = obj.nLayers;
            grad = struct();
            dh = dy;
            for i = L:-1:1
                if i < L
                    yl = cache.yl{i};
                    dyl = dh .* (1 - tanh(yl) .^ 2);
                    if obj.useLayerNorm
                        [dz, dg, do_] = obj.lnBackward(cache.ln{i}, dyl);
                        grad.(sprintf('lg%d', i)) = dg;
                        grad.(sprintf('lo%d', i)) = do_;
                    else
                        dz = dyl;
                    end
                else
                    dz = dh;
                end
                if i > 1, hprev = cache.h{i - 1}; else, hprev = cache.x; end
                W = obj.learn.(sprintf('W%d', i));
                grad.(sprintf('W%d', i)) = dz * hprev';
                grad.(sprintf('b%d', i)) = sum(dz, 2);
                dh = W' * dz;
            end
            dx = dh;
        end
    end

    methods (Access = private)
        function [y, cache] = lnForward(obj, x, g, beta)
            mu = mean(x, 1);
            v = mean((x - mu) .^ 2, 1);
            istd = 1 ./ sqrt(v + obj.lnEps);
            xhat = (x - mu) .* istd;
            y = g .* xhat + beta;
            cache.xhat = xhat;
            cache.istd = istd;
            cache.g = g;
        end

        function [dx, dg, dbeta] = lnBackward(~, cache, dy)
            xhat = cache.xhat; istd = cache.istd; g = cache.g;
            C = size(xhat, 1);
            dg = sum(dy .* xhat, 2);
            dbeta = sum(dy, 2);
            dxhat = dy .* g;
            dx = (istd / C) .* (C * dxhat ...
                - sum(dxhat, 1) ...
                - xhat .* sum(dxhat .* xhat, 1));
        end
    end
end
