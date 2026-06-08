function p = softmaxCol(logits)
    z = logits - max(logits, [], 1);
    e = exp(z);
    p = e ./ sum(e, 1);
end
