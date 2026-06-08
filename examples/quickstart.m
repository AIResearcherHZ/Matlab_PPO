root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'core'), fullfile(root, 'utils'), ...
        fullfile(root, 'environments'), fullfile(root, 'config'));

agent = PPO(CartPoleEnv(), rlConfig('CartPole'));
agent.train();

res = agent.evaluate(20);
fprintf('CartPole eval over 20 episodes: mean return %.1f +/- %.1f\n', ...
        res.meanReturn, res.stdReturn);
