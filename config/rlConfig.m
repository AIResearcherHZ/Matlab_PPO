function cfg = rlConfig(name)

    if nargin < 1, name = ''; end

    cfg = struct();
    cfg.gamma = 0.99;        cfg.lambda = 0.95;
    cfg.clip = 0.2;          cfg.entCoef = 0.0;       cfg.vfCoef = 0.5;
    cfg.maxGradNorm = 0.5;
    cfg.actorLR = 3e-4;      cfg.criticLR = 1e-3;     cfg.lrAnneal = true;
    cfg.nSteps = 2048;       cfg.nEpochs = 10;        cfg.miniBatch = 64;
    cfg.targetKL = 0.02;     cfg.clipVF = Inf;
    cfg.hidden = [64 64];    cfg.useLayerNorm = false; cfg.logStdInit = 0.0;
    cfg.normObs = true;      cfg.normReward = false;
    cfg.numIterations = 100; cfg.seed = 0;            cfg.verbose = true;
    cfg.logDir = '';         cfg.saveEvery = 0;

    key = lower(strrep(name, 'Env', ''));
    key = lower(strrep(key, 'env', ''));
    switch key
        case 'cartpole'
            cfg.entCoef = 0.01;
            cfg.nSteps = 2048; cfg.nEpochs = 10; cfg.miniBatch = 64;
            cfg.normReward = false; cfg.numIterations = 60;
        case 'pendulum'
            cfg.entCoef = 0.0; cfg.normReward = true;
            cfg.nSteps = 2048; cfg.nEpochs = 10; cfg.miniBatch = 64;
            cfg.numIterations = 150;
        case 'dcmotor'
            cfg.entCoef = 0.0; cfg.normReward = true;
            cfg.nSteps = 2048; cfg.nEpochs = 10; cfg.miniBatch = 128;
            cfg.numIterations = 120;
        case 'acmotor'
            cfg.hidden = [256 256]; cfg.useLayerNorm = true;
            cfg.clip = 0.1; cfg.entCoef = 0.0; cfg.logStdInit = -0.5;
            cfg.normReward = true; cfg.targetKL = 0.03;
            cfg.nSteps = 4096; cfg.nEpochs = 10; cfg.miniBatch = 256;
            cfg.numIterations = 200;
        otherwise
    end
end
