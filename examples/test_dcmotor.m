root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'core'), fullfile(root, 'utils'), ...
        fullfile(root, 'environments'), fullfile(root, 'config'));

modelPath = fullfile(root, 'logs', 'dcmotor', 'model_final.mat');
if exist(modelPath, 'file') ~= 2
    error('Model not found: %s\nRun train_dcmotor first.', modelPath);
end

agent = PPO(DCMotorEnv(), rlConfig('DCMotor'));
agent.load(modelPath);
res = agent.evaluate(20);
fprintf('DCMotor test: mean %.2f +/- %.2f, min %.2f, max %.2f\n', ...
        res.meanReturn, res.stdReturn, res.minReturn, res.maxReturn);
