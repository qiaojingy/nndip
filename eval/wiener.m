function y = wiener(x)
    std = 0.003;
    y = zeros(size(x));
    psf = fspecial('gaussian', [9 9], 3);
    [numRows, numCols, ~] = size(x);
    otf = psf2otf(psf, [numRows numCols]);
    for c = 1: 3
        img = x(:,:,c);
        img_blur = ifft2(fft2(img).*otf);
        img_noisy = img_blur + std*randn(size(img_blur));
        H = 1./otf.* ((abs(otf).^2)./(abs(otf).^2 + ...
            std/mean(img_noisy(:))));
        img_wnr = ifft2(fft2(img_noisy).*H);
        y(:, :, c) = img_wnr;
    end