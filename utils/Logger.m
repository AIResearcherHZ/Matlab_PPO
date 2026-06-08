classdef Logger < handle

    properties
        path
        fid = -1
        headerWritten = false
        fields = {}
    end

    methods
        function obj = Logger(path)
            obj.path = path;
            obj.fid = fopen(path, 'w');
            if obj.fid < 0
                warning('Logger:open', 'could not open %s', path);
            end
        end

        function log(obj, rec)
            if obj.fid < 0, return; end
            if ~obj.headerWritten
                obj.fields = fieldnames(rec);
                fprintf(obj.fid, '%s\n', strjoin(obj.fields', ','));
                obj.headerWritten = true;
            end
            vals = cell(1, numel(obj.fields));
            for k = 1:numel(obj.fields)
                vals{k} = sprintf('%.6g', rec.(obj.fields{k}));
            end
            fprintf(obj.fid, '%s\n', strjoin(vals, ','));
        end

        function delete(obj)
            if obj.fid >= 0, fclose(obj.fid); obj.fid = -1; end
        end
    end
end
