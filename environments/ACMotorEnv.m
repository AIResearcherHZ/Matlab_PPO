classdef ACMotorEnv < Environment

    properties
        Rs = 2.0; Rr = 2.0; Ls = 0.2; Lr = 0.2; Lm = 0.15
        J = 0.02; p = 2; B = 0.005
        dt = 1e-3
        maxVoltage = 400.0; maxCurrent = 15.0; maxSpeed = 157.0
        nominalSpeed = 150; nominalTorque = 10
        maxSteps = 1000
        Kp = 0.5; Ki = 5.0
        idRef = 3.0
        targetSpeed; iqRef = 0; speedIntegral = 0
        speed = 0; id = 0; iq = 0; psiD = 0; psiQ = 0; Te = 0; Tl = 0
        steps = 0
        loadChangeSteps = [200 400 600 800]
        loadProfile
    end

    methods
        function obj = ACMotorEnv()
            obj.obsDim = 6;
            obj.actDim = 2;
            obj.isDiscrete = false;
            obj.loadProfile = obj.nominalTorque * [0.2 0.8 0.4 0.9 0.3];
            obj.targetSpeed = obj.nominalSpeed;
        end

        function obs = reset(obj)
            obj.speed = 0; obj.id = 0; obj.iq = 0;
            obj.psiD = 0; obj.psiQ = 0; obj.Te = 0;
            obj.Tl = obj.loadProfile(1);
            obj.iqRef = 0; obj.speedIntegral = 0; obj.steps = 0;
            obj.targetSpeed = (0.5 + 0.5 * rand()) * obj.nominalSpeed;
            obs = obj.getObs();
        end

        function [obs, reward, done, info] = step(obj, action)
            a = max(min(action(:), 1), -1);
            Vd = a(1) * obj.maxVoltage;
            Vq = a(2) * obj.maxVoltage;
            obj.updateLoad();

            speedErr = obj.targetSpeed - obj.speed;
            obj.speedIntegral = obj.speedIntegral + speedErr * obj.dt;
            obj.iqRef = obj.Kp * speedErr + obj.Ki * obj.speedIntegral;
            obj.iqRef = max(min(obj.iqRef, obj.maxCurrent), -obj.maxCurrent);

            id0 = obj.id; iq0 = obj.iq; pd0 = obj.psiD; pq0 = obj.psiQ; w0 = obj.speed;
            obj.Te = 1.5 * obj.p * obj.Lm / obj.Lr * (iq0 * pd0 - id0 * pq0);
            sigma = 1 - obj.Lm ^ 2 / (obj.Ls * obj.Lr);
            Tr = obj.Lr / obj.Rr;

            wNew = w0 + obj.dt * ((obj.Te - obj.Tl - obj.B * w0) / obj.J);
            pdNew = pd0 + obj.dt * ((-pd0 + obj.Lm * id0) / Tr - obj.p * w0 * pq0);
            pqNew = pq0 + obj.dt * ((-pq0 + obj.Lm * iq0) / Tr + obj.p * w0 * pd0);
            idNew = id0 + obj.dt * ((Vd - obj.Rs * id0 + obj.p * w0 * sigma * obj.Ls * iq0) ...
                / (sigma * obj.Ls));
            iqNew = iq0 + obj.dt * ((Vq - obj.Rs * iq0 - obj.p * w0 ...
                * (sigma * obj.Ls * id0 + obj.Lm / obj.Lr * pd0)) / (sigma * obj.Ls));
            idNew = max(min(idNew, obj.maxCurrent), -obj.maxCurrent);
            iqNew = max(min(iqNew, obj.maxCurrent), -obj.maxCurrent);
            wNew = max(min(wNew, obj.maxSpeed), -obj.maxSpeed);

            obj.id = idNew; obj.iq = iqNew; obj.psiD = pdNew; obj.psiQ = pqNew;
            obj.speed = wNew; obj.steps = obj.steps + 1;

            reward = -0.5 * (speedErr / obj.maxSpeed) ^ 2 ...
                - 0.3 * ((obj.idRef - idNew) / obj.maxCurrent) ^ 2 ...
                - 0.3 * ((obj.iqRef - iqNew) / obj.maxCurrent) ^ 2 ...
                - 0.05 * ((Vd / obj.maxVoltage) ^ 2 + (Vq / obj.maxVoltage) ^ 2);

            truncated = obj.steps >= obj.maxSteps;
            done = truncated;
            obs = obj.getObs();
            info = struct('truncated', truncated, 'speed', obj.speed, 'Te', obj.Te);
        end
    end

    methods (Access = private)
        function updateLoad(obj)
            k = find(obj.loadChangeSteps == obj.steps, 1);
            if ~isempty(k), obj.Tl = obj.loadProfile(k + 1); end
        end

        function obs = getObs(obj)
            speedErr = obj.targetSpeed - obj.speed;
            obs = [speedErr / obj.maxSpeed; ...
                   obj.id / obj.maxCurrent; ...
                   obj.iq / obj.maxCurrent; ...
                   (obj.idRef - obj.id) / obj.maxCurrent; ...
                   (obj.iqRef - obj.iq) / obj.maxCurrent; ...
                   obj.Tl / obj.nominalTorque];
        end
    end
end
