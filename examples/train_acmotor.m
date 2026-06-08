root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'core'), fullfile(root, 'utils'), ...
        fullfile(root, 'environments'), fullfile(root, 'config'));

cfg = rlConfig('ACMotor');
cfg.logDir = fullfile(root, 'logs', 'acmotor');

agent = PPO(ACMotorEnv(), cfg);
agent.train();

res = agent.evaluate(20);
fprintf('ACMotor eval over 20 episodes: mean return %.2f +/- %.2f\n', ...
        res.meanReturn, res.stdReturn);
