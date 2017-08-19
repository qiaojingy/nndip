function x = ADMM(I)

% load image
%I = im2double(imread('birds_gray.png'));

% blur kernel
c = fspecial('gaussian', [9 9], 3);

% convolution kernel for Dx and Dy
dy = [0 0 0; 0 -1 0; 0 1 0];
dx = [0 0 0; 0 -1 1; 0 0 0];

% functions converting a point spread function (convolution kernel) to the
% corresponding optical transfer function (Fouier multiplier)
p2o = @(x) psf2otf(x, size(I));

% precompute OTFs 
cFT     = p2o(c);
cTFT    = conj(p2o(c));
dxFT    = p2o(dx);
dxTFT   = conj(p2o(dx));
dyFT    = p2o(dy);
dyTFT   = conj(p2o(dy));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% blur image with kernel
Ib = ifft2(fft2(I).*cFT);

% noise parameter - standard deviation  (one of the ones you will test)
sigma = 0.003;
lambda = 0.00001;
rho = 10;
W = size(I, 1);
H = size(I, 2);
maxIters = 100;
% add noise to blurred image
b = Ib + sigma.*randn(size(I));
% deconvolution using ADMM here
x = zeros(W, H);
z = zeros(W, H, 2);
u = zeros(W, H, 2);
Dx = zeros(W, H, 2);
residual = zeros(1, maxIters);
for k = 1:maxIters
    v = z - u;
    v1FT = fft2(v(:,:,1));
    v2FT = fft2(v(:,:,2));
    x = ifft2((cTFT.*fft2(b) + rho*(dxTFT.*v1FT + dyTFT.*v2FT))...
        ./(cTFT.*cFT + rho*(dxTFT.*dxFT + dyTFT.*dyFT)));
    DxTx = real(ifft2(fft2(x).*dxFT));
    DyTx = real(ifft2(fft2(x).*dyFT));
    Dx(:,:,1) = DxTx;
    Dx(:,:,2) = DyTx;
    v = Dx + u;
    kai = lambda/rho;
    z = zeros(W, H, 2);
    z(v > kai) = v(v > kai) - kai;
    z(v < -kai) = v(v < -kai) + kai;
    u = u + Dx - z;
    residual(k) = 0.5*sum(sum(((ifft2(fft2(x).*cFT) - b).^2)))...
        + lambda*(sum(sum(sum(abs(Dx)))));
end


