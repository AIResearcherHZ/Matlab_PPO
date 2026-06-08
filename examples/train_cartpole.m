root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'core'), fullfile(root, 'utils'), ...
        fullfile(root, 'environments'), fullfile(root, 'config'));

cfg = rlConfig('CartPole');
cfg.logDir = fullfile(root, 'logs', 'cartpole');

agent = PPO(CartPoleEnv(), cfg);
agent.train();

res = agent.evaluate(20);
fprintf('CartPole eval over 20 episodes: mean return %.2f +/- %.2f\n', ...
        res.meanReturn, res.stdReturn);
