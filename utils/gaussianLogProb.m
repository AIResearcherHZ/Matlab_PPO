function lp = gaussianLogProb(mu, logStd, x)
    sd2 = exp(2 * logStd);
    k = size(mu, 1);
    lp = -0.5 * sum(((x - mu) .^ 2) ./ sd2, 1) ...
         - sum(logStd(:)) ...
         - 0.5 * k * log(2 * pi);
end
