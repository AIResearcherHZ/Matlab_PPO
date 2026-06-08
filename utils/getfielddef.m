function v = getfielddef(s, name, def)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
    else
        v = def;
    end
end
