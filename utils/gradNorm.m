function n = gradNorm(varargin)
    s = 0;
    for a = 1:numel(varargin)
        g = varargin{a};
        f = fieldnames(g);
        for k = 1:numel(f)
            x = g.(f{k});
            s = s + sum(x(:) .^ 2);
        end
    end
    n = sqrt(s);
end
