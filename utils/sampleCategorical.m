function idx = sampleCategorical(p)
    c = cumsum(p);
    idx = find(c >= rand() * c(end), 1, 'first');
    if isempty(idx), idx = numel(p); end
end
