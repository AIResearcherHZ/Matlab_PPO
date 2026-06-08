function ev = explainedVariance(y, ypred)
    y = y(:); ypred = ypred(:);
    vy = var(y, 1);
    if vy < 1e-12
        ev = 0;
    else
        ev = 1 - var(y - ypred, 1) / vy;
    end
end
