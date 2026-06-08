function W = orthoInit(numOut, numIn, gain)
    if nargin < 3, gain = 1.0; end
    rows = numOut; cols = numIn; flip = false;
    if rows < cols
        tmp = rows; rows = cols; cols = tmp; flip = true;
    end
    A = randn(rows, cols);
    [Q, R] = qr(A, 0);
    d = sign(diag(R)); d(d == 0) = 1;
    Q = Q .* d';
    if flip
        W = gain * Q';
    else
        W = gain * Q;
    end
end
