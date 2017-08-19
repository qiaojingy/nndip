%% 
clear; close all; clc;
%%
target = 'Face';
%%
len = 64;
imgTests = im2double(imread(['testDeconv' target '.png']));
figure;
imshow(imgTests);
%%

imgNum = 8;
imgResult = zeros(len*imgNum, len*5, 3);
psnrsADMM = zeros(1, length(imgNum));
psnrsWNR = zeros(1, length(imgNum));
psnrsCNN = zeros(1, length(imgNum));
for i = 1:imgNum
    img = imgTests((i-1)*64+1: i*64, 129:end, :);
    imgCNN = imgTests((i-1)*64+1: i*64, 65:128, :);
    imgBlur = imgTests((i-1)*64+1: i*64, 1:64, :);
    imgResult((i-1)*64+1: i*64, 1:64, :) = img;
    imgResult((i-1)*64+1: i*64, 65:128, :) = imgBlur;
    imgResult((i-1)*64+1: i*64, 257:end, :) = imgCNN;     
    imgADMM = zeros(size(img));
    for c = 1:3
        imgADMM(:,:,c) = ADMM(img(:,:,c));
    end
    imgWNR = wiener(img);
    imgResult((i-1)*64+1: i*64, 129:192, :) = imgWNR; 
    imgResult((i-1)*64+1: i*64, 193:256, :) = imgADMM;
    psnrsWNR(i) = PSNR(imgWNR, img);
    psnrsCNN(i) = PSNR(imgCNN, img);
    psnrsADMM(i) = PSNR(imgADMM, img);
end
%%
figure; imshow(imgResult)
imwrite(imgResult, ['deconvResult' target '.png']);
fileID = fopen(['deconv' target 'PSNR' '.txt'], 'w');
fprintf(fileID, '%12s %12s %12s\n','CNN','wiener', 'ADMM');
fprintf(fileID,'%12.8f %12.8f\n',[psnrsCNN', psnrsWNR', psnrsADMM']);
fclose(fileID);
%%
CNN_mean = mean(psnrsCNN)
CNN_std = std(psnrsCNN)
WNR_mean = mean(psnrsWNR)
WNR_std = std(psnrsWNR)
ADMM_mean = mean(psnrsADMM)
ADMM_std = std(psnrsADMM)