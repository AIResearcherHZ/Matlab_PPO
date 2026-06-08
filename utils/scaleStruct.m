function s = scaleStruct(s, c)
    f = fieldnames(s);
    for k = 1:numel(f)
        s.(f{k}) = s.(f{k}) * c;
    end
end
