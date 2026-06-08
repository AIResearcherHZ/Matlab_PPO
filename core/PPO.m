classdef PPO < handle

    properties
        env
        cfg
        isDiscrete
        obsDim
        actDim
        actor
        critic
        logStd
        optActor
        optCritic
        optLogStd
        obsRMS
        retRMS
        retAcc = 0
        history
    end

    methods
        function obj = PPO(env, cfg)
            if nargin < 2, cfg = rlConfig(); end
            obj.env = env;
            obj.cfg = cfg;
            obj.obsDim = env.obsDim;
            obj.isDiscrete = env.isDiscrete;
            obj.actDim = env.actDim;

            if isfield(cfg, 'seed') && ~isempty(cfg.seed)
                rand('seed', cfg.seed); randn('seed', cfg.seed);
            end

            hidden = getfielddef(cfg, 'hidden', [64 64]);
            useLN = getfielddef(cfg, 'useLayerNorm', false);
            actorOut = obj.actDim;
            obj.actor = MLP([obj.obsDim hidden actorOut], ...
                struct('useLayerNorm', useLN, 'outGain', 0.01));
            obj.critic = MLP([obj.obsDim hidden 1], ...
                struct('useLayerNorm', useLN, 'outGain', 1.0));
            obj.optActor = Adam();
            obj.optCritic = Adam();
            if ~obj.isDiscrete
                obj.logStd = getfielddef(cfg, 'logStdInit', 0.0) * ones(obj.actDim, 1);
                obj.optLogStd = Adam();
            end

            if getfielddef(cfg, 'normObs', true)
                obj.obsRMS = RunningMeanStd(obj.obsDim);
            end
            if getfielddef(cfg, 'normReward', false)
                obj.retRMS = RunningMeanStd(1);
            end
            obj.history = struct('iter', {}, 'ret', {}, 'len', {}, 'ploss', {}, ...
                'vloss', {}, 'entropy', {}, 'kl', {}, 'clipfrac', {}, 'ev', {});
        end

        function train(obj)
            cfg = obj.cfg;
            nIters = getfielddef(cfg, 'numIterations', 100);
            verbose = getfielddef(cfg, 'verbose', true);
            logDir = getfielddef(cfg, 'logDir', '');
            saveEvery = getfielddef(cfg, 'saveEvery', 0);
            logger = [];
            if ~isempty(logDir)
                if ~exist(logDir, 'dir'), mkdir(logDir); end
                logger = Logger(fullfile(logDir, 'progress.csv'));
            end
            for it = 1:nIters
                frac = 1 - (it - 1) / nIters;
                roll = obj.collectRollout();
                roll = obj.computeGAE(roll);
                m = obj.update(roll, frac);
                rec = struct('iter', it, 'ret', roll.epRetMean, 'len', roll.epLenMean, ...
                    'ploss', m.ploss, 'vloss', m.vloss, 'entropy', m.entropy, ...
                    'kl', m.kl, 'clipfrac', m.clipfrac, ...
                    'ev', explainedVariance(roll.returns, roll.vals));
                obj.history(end + 1) = rec;
                if verbose
                    fprintf(['iter %3d | return %8.2f | len %6.1f | ploss %+.3f | ' ...
                        'vloss %.3f | ent %+.3f | kl %.4f | clipf %.2f | ev %+.2f\n'], ...
                        it, rec.ret, rec.len, rec.ploss, rec.vloss, rec.entropy, ...
                        rec.kl, rec.clipfrac, rec.ev);
                end
                if ~isempty(logger), logger.log(rec); end
                if saveEvery > 0 && mod(it, saveEvery) == 0 && ~isempty(logDir)
                    obj.save(fullfile(logDir, sprintf('model_%d.mat', it)));
                end
            end
            if ~isempty(logDir), obj.save(fullfile(logDir, 'model_final.mat')); end
        end

        function action = predict(obj, obs, deterministic)
            if nargin < 3, deterministic = true; end
            o = obj.normObsVec(obs(:));
            if obj.isDiscrete
                logits = obj.actor.forward(o);
                if deterministic
                    [~, idx] = max(logits, [], 1);
                else
                    idx = sampleCategorical(softmaxCol(logits));
                end
                action = idx;
            else
                mu = obj.actor.forward(o);
                if deterministic
                    a = mu;
                else
                    a = mu + exp(obj.logStd) .* randn(obj.actDim, 1);
                end
                action = a;
            end
        end

        function result = evaluate(obj, numEpisodes, deterministic)
            if nargin < 2, numEpisodes = 10; end
            if nargin < 3, deterministic = true; end
            rets = zeros(numEpisodes, 1); lens = zeros(numEpisodes, 1);
            for e = 1:numEpisodes
                obs = obj.env.reset(); done = false; R = 0; L = 0;
                while ~done
                    a = obj.predict(obs, deterministic);
                    [obs, r, done, ~] = obj.env.step(a);
                    R = R + r; L = L + 1;
                end
                rets(e) = R; lens(e) = L;
            end
            result = struct('meanReturn', mean(rets), 'stdReturn', std(rets), ...
                'minReturn', min(rets), 'maxReturn', max(rets), 'meanLength', mean(lens));
        end

        function save(obj, path)
            data = struct();
            data.cfg = obj.cfg;
            data.isDiscrete = obj.isDiscrete;
            data.obsDim = obj.obsDim; data.actDim = obj.actDim;
            data.actor = obj.actor.learn;
            data.actorSizes = obj.actor.sizes; data.actorLN = obj.actor.useLayerNorm;
            data.critic = obj.critic.learn;
            data.criticSizes = obj.critic.sizes; data.criticLN = obj.critic.useLayerNorm;
            if ~obj.isDiscrete, data.logStd = obj.logStd; end
            if ~isempty(obj.obsRMS), data.obsRMS = obj.obsRMS.toStruct(); end
            if ~isempty(obj.retRMS), data.retRMS = obj.retRMS.toStruct(); end
            save(path, '-struct', 'data');
            if getfielddef(obj.cfg, 'verbose', true)
                fprintf('saved model -> %s\n', path);
            end
        end

        function load(obj, path)
            data = load(path);
            if isfield(data, 'isDiscrete') && data.isDiscrete ~= obj.isDiscrete
                error('PPO:load', ['saved model action-space type differs from this ' ...
                    'agent/environment (discrete vs continuous)']);
            end
            if isfield(data, 'actorSizes')
                obj.actor = MLP(data.actorSizes, ...
                    struct('useLayerNorm', logical(data.actorLN), 'outGain', 0.01));
            end
            obj.actor.learn = data.actor;
            if isfield(data, 'criticSizes')
                obj.critic = MLP(data.criticSizes, ...
                    struct('useLayerNorm', logical(data.criticLN), 'outGain', 1.0));
            end
            obj.critic.learn = data.critic;
            if ~obj.isDiscrete && isfield(data, 'logStd'), obj.logStd = data.logStd; end
            if isfield(data, 'obsRMS')
                if isempty(obj.obsRMS), obj.obsRMS = RunningMeanStd(obj.obsDim); end
                obj.obsRMS.fromStruct(data.obsRMS);
            end
            if isfield(data, 'retRMS')
                if isempty(obj.retRMS), obj.retRMS = RunningMeanStd(1); end
                obj.retRMS.fromStruct(data.retRMS);
            end
        end
    end

    methods (Access = private)
        function roll = collectRollout(obj)
            cfg = obj.cfg; n = getfielddef(cfg, 'nSteps', 2048);
            gamma = getfielddef(cfg, 'gamma', 0.99);
            doNormR = ~isempty(obj.retRMS);

            obsN = zeros(obj.obsDim, n);
            if obj.isDiscrete, acts = zeros(1, n); else, acts = zeros(obj.actDim, n); end
            logps = zeros(1, n); rews = zeros(1, n);
            vals = zeros(1, n); dones = zeros(1, n);

            epRets = []; epLens = [];
            obs = obj.env.reset(); obj.retAcc = 0;
            curRet = 0; curLen = 0;
            for t = 1:n
                o = obj.observe(obs(:));
                obsN(:, t) = o;
                [a, lp] = obj.sampleAction(o);
                vals(t) = obj.critic.forward(o);
                if obj.isDiscrete, acts(t) = a; else, acts(:, t) = a; end
                logps(t) = lp;

                [obs2, r, done, info] = obj.env.step(a);
                curRet = curRet + r; curLen = curLen + 1;

                rScaled = r;
                if doNormR
                    obj.retAcc = gamma * obj.retAcc + r;
                    obj.retRMS.update(obj.retAcc);
                    rScaled = r / sqrt(obj.retRMS.var + obj.retRMS.eps);
                    rScaled = max(min(rScaled, 10), -10);
                end

                truncated = isfield(info, 'truncated') && info.truncated;
                if done && truncated
                    oNext = obj.normObsVec(obs2(:));
                    rScaled = rScaled + gamma * obj.critic.forward(oNext);
                end
                rews(t) = rScaled;
                dones(t) = double(done);

                if done
                    epRets(end + 1) = curRet; epLens(end + 1) = curLen;
                    curRet = 0; curLen = 0; obj.retAcc = 0;
                    obs = obj.env.reset();
                else
                    obs = obs2;
                end
            end
            if dones(n) > 0
                lastVal = 0;
            else
                lastVal = obj.critic.forward(obj.normObsVec(obs(:)));
            end

            roll = struct();
            roll.obsN = obsN; roll.acts = acts; roll.logps = logps;
            roll.rews = rews; roll.vals = vals; roll.dones = dones;
            roll.lastVal = lastVal;
            if isempty(epRets)
                roll.epRetMean = curRet; roll.epLenMean = curLen;
            else
                roll.epRetMean = mean(epRets); roll.epLenMean = mean(epLens);
            end
        end

        function roll = computeGAE(obj, roll)
            gamma = getfielddef(obj.cfg, 'gamma', 0.99);
            lambda = getfielddef(obj.cfg, 'lambda', 0.95);
            [roll.adv, roll.returns] = gae(roll.rews, roll.vals, roll.dones, ...
                roll.lastVal, gamma, lambda);
        end

        function m = update(obj, roll, frac)
            cfg = obj.cfg;
            nEpochs = getfielddef(cfg, 'nEpochs', 10);
            mb = getfielddef(cfg, 'miniBatch', 64);
            clipE = getfielddef(cfg, 'clip', 0.2);
            entCoef = getfielddef(cfg, 'entCoef', 0.0);
            vfCoef = getfielddef(cfg, 'vfCoef', 0.5);
            maxGN = getfielddef(cfg, 'maxGradNorm', 0.5);
            targetKL = getfielddef(cfg, 'targetKL', Inf);
            clipVF = getfielddef(cfg, 'clipVF', Inf);
            anneal = getfielddef(cfg, 'lrAnneal', true);
            lrA = getfielddef(cfg, 'actorLR', 3e-4);
            lrC = getfielddef(cfg, 'criticLR', 1e-3);
            if anneal, lrA = lrA * frac; lrC = lrC * frac; end

            N = numel(roll.rews);
            obsN = roll.obsN; acts = roll.acts; oldlp = roll.logps;
            adv = roll.adv; returns = roll.returns; oldv = roll.vals;

            ploss = 0; vloss = 0; ent = 0; kl = 0; cf = 0; cnt = 0;
            for ep = 1:nEpochs
                idx = randperm(N);
                epKL = 0; epCnt = 0;
                for s = 1:mb:N
                    b = idx(s:min(s + mb - 1, N));
                    bObs = obsN(:, b);
                    if obj.isDiscrete, bAct = acts(b); else, bAct = acts(:, b); end
                    bOldlp = oldlp(b); bAdv = adv(b);
                    bRet = returns(b); bOldv = oldv(b);
                    Bn = numel(b);

                    bAdv = (bAdv - mean(bAdv)) / (std(bAdv) + 1e-8);

                    [lp, entropy, pcache, extra] = obj.evalActions(bObs, bAct);
                    ratio = exp(lp - bOldlp);
                    s1 = ratio .* bAdv;
                    s2 = max(min(ratio, 1 + clipE), 1 - clipE) .* bAdv;
                    surr = min(s1, s2);
                    polLoss = -mean(surr);
                    mask = double(s1 <= s2);
                    dlp = -(1 / Bn) * (bAdv .* ratio) .* mask;
                    [gActor, gLogStd] = obj.policyBackward(dlp, entropy, entCoef, pcache, extra, Bn);

                    [value, ccache] = obj.critic.forward(bObs);
                    if isfinite(clipVF)
                        vclip = bOldv + max(min(value - bOldv, clipVF), -clipVF);
                        unclipped = (value - bRet) .^ 2;
                        clipped = (vclip - bRet) .^ 2;
                        vfLoss = mean(max(unclipped, clipped));
                        pick = double(unclipped >= clipped);
                        dval = vfCoef * (2 / Bn) * (value - bRet) .* pick;
                    else
                        vfLoss = mean((value - bRet) .^ 2);
                        dval = vfCoef * (2 / Bn) * (value - bRet);
                    end
                    gCritic = obj.critic.backward(ccache, dval);

                    if maxGN > 0
                        an = gradNorm(gActor, struct('logStd', gLogStd));
                        if an > maxGN
                            sc = maxGN / (an + 1e-12);
                            gActor = scaleStruct(gActor, sc); gLogStd = gLogStd * sc;
                        end
                        cn = gradNorm(gCritic);
                        if cn > maxGN, gCritic = scaleStruct(gCritic, maxGN / (cn + 1e-12)); end
                    end
                    obj.actor.learn = obj.optActor.step(obj.actor.learn, gActor, lrA);
                    if ~obj.isDiscrete
                        lsp = obj.optLogStd.step(struct('logStd', obj.logStd), ...
                            struct('logStd', gLogStd), lrA);
                        obj.logStd = max(min(lsp.logStd, 2), -20);
                    end
                    obj.critic.learn = obj.optCritic.step(obj.critic.learn, gCritic, lrC);

                    approxKL = mean((ratio - 1) - log(ratio + 1e-12));
                    ploss = ploss + polLoss; vloss = vloss + vfLoss;
                    ent = ent + mean(entropy); kl = kl + approxKL;
                    cf = cf + mean(double(abs(ratio - 1) > clipE)); cnt = cnt + 1;
                    epKL = epKL + approxKL; epCnt = epCnt + 1;
                end
                if isfinite(targetKL) && (epKL / epCnt) > 1.5 * targetKL
                    break;
                end
            end
            m = struct('ploss', ploss / cnt, 'vloss', vloss / cnt, ...
                'entropy', ent / cnt, 'kl', kl / cnt, 'clipfrac', cf / cnt);
        end

        function [a, lp] = sampleAction(obj, o)
            if obj.isDiscrete
                logits = obj.actor.forward(o);
                p = softmaxCol(logits);
                a = sampleCategorical(p);
                lp = log(p(a) + 1e-12);
            else
                mu = obj.actor.forward(o);
                sd = exp(obj.logStd);
                a = mu + sd .* randn(obj.actDim, 1);
                lp = gaussianLogProb(mu, obj.logStd, a);
            end
        end

        function [lp, entropy, cache, extra] = evalActions(obj, obs, act)
            if obj.isDiscrete
                [logits, cache] = obj.actor.forward(obs);
                p = softmaxCol(logits);
                logpAll = log(p + 1e-12);
                B = size(obs, 2);
                lin = sub2ind(size(p), act, 1:B);
                lp = logpAll(lin);
                entropy = -sum(p .* logpAll, 1);
                extra = struct('p', p, 'logpAll', logpAll, 'act', act);
            else
                [mu, cache] = obj.actor.forward(obs);
                lp = gaussianLogProb(mu, obj.logStd, act);
                entropy = (sum(obj.logStd) + 0.5 * obj.actDim * log(2 * pi * exp(1))) * ...
                    ones(1, size(obs, 2));
                extra = struct('mu', mu, 'act', act);
            end
        end

        function [gActor, gLogStd] = policyBackward(obj, dlp, entropy, entCoef, pcache, extra, B)
            if obj.isDiscrete
                p = extra.p; logpAll = extra.logpAll; act = extra.act;
                onehot = zeros(size(p));
                onehot(sub2ind(size(p), act, 1:B)) = 1;
                dlogits = dlp .* (onehot - p);
                H = entropy;
                dentLogits = p .* (-(logpAll) - H);
                dlogits = dlogits + (-entCoef) * (1 / B) * dentLogits;
                gActor = obj.actor.backward(pcache, dlogits);
                gLogStd = [];
            else
                mu = extra.mu; act = extra.act;
                sd = exp(obj.logStd); inv2 = 1 ./ (sd .^ 2);
                dmu = dlp .* (act - mu) .* inv2;
                dLogStdLp = dlp .* (((act - mu) .^ 2) .* inv2 - 1);
                gLogStd = sum(dLogStdLp, 2) + (-entCoef) * ones(obj.actDim, 1);
                gActor = obj.actor.backward(pcache, dmu);
            end
        end

        function o = observe(obj, obs)
            if ~isempty(obj.obsRMS)
                obj.obsRMS.update(obs);
                o = obj.obsRMS.normalize(obs);
            else
                o = obs;
            end
        end

        function o = normObsVec(obj, obs)
            if ~isempty(obj.obsRMS)
                o = obj.obsRMS.normalize(obs);
            else
                o = obs;
            end
        end
    end
end
