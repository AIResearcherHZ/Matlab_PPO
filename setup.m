root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'core'));
addpath(fullfile(root, 'utils'));
addpath(fullfile(root, 'environments'));
addpath(fullfile(root, 'config'));
fprintf('Matlab_PPO ready. Try:  agent = PPO(CartPoleEnv(), rlConfig(''CartPole'')); agent.train();\n');
