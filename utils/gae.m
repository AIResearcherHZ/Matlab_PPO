function [adv, returns] = gae(rewards, values, dones, lastVal, gamma, lambda)
    n = numel(rewards);
    adv = zeros(1, n);
    g = 0;
    for t = n:-1:1
        if t == n
            nextVal = lastVal;
        else
            nextVal = values(t + 1);
        end
        nonTerm = 1 - dones(t);
        delta = rewards(t) + gamma * nextVal * nonTerm - values(t);
        g = delta + gamma * lambda * nonTerm * g;
        adv(t) = g;
    end
    returns = adv + values;
end
