function psnr = PSNR(restored, original)
    mse = sum((restored(:) - original(:)).^2)/length(restored(:));
    psnr = 10*log10(max(original(:))^2/mse);
end