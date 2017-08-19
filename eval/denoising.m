%% 
clear; close all; clc;
addpath('../HW6');
%%
target = 'Face';
%%
len = 64;
imgTests = im2double(imread(['testDenoising' target '.png']));
figure;
imshow(imgTests);
%%
imgNum = 8;
imgResult = zeros(len*imgNum, len*5, 3);
psnrsNLM = zeros(1, length(imgNum));
psnrsCNN = zeros(1, length(imgNum));
psnrsMED = zeros(1, length(imgNum));
for i = 1:imgNum
    img = imgTests((i-1)*64+1: i*64, 129:end, :);
    imgCNN = imgTests((i-1)*64+1: i*64, 65:128, :);
    imgResult((i-1)*64+1: i*64, 1:64, :) = img;
    imgResult((i-1)*64+1: i*64, 257:end, :) = imgCNN; 
    [numRows, numCols, ~] = size(img);
    imgNoisy = imgTests((i-1)*64+1: i*64, 1:64, :);
    imgResult((i-1)*64+1: i*64, 65:128, :) = imgNoisy;
    Options.filterstrength = 0.1;
    imgNLM = NLMF(imgNoisy, Options);
    imgMED = cat(3, medfilt2(imgNoisy(:,:,1), [3 3]), ...
        medfilt2(imgNoisy(:,:,2), [3 3]), ...
        medfilt2(imgNoisy(:,:,3), [3 3]));
    imgResult((i-1)*64+1: i*64, 193:256, :) = imgNLM;
    imgResult((i-1)*64+1: i*64, 129:192, :) = imgMED;
    psnrsNLM(i) = PSNR(imgNLM, img);
    psnrsCNN(i) = PSNR(imgCNN, img);
    psnrsMED(i) = PSNR(imgMED, img);
    if i == 8
        imgSample = zeros(64, 64*5, 3);
        imgSample(1:64, 1:64, :) = img;
        imgSample(1:64, 65:128, :) = imgNoisy;
        imgSample(1:64, 129:192, :) = imgMED;
        imgSample(1:64, 193:256, :) = imgNLM;
        imgSample(1:64, 257:320, :) = imgCNN;
        imwrite(imgSample, ['sample_denoising_' target '_result.jpg']);
    end
end
%%
figure; imshow(imgResult)
imwrite(imgResult, ['denoisingResult' target '.png']);
fileID = fopen(['denoising' target 'PSNR.txt'], 'w');
fprintf(fileID, '%12s %12s %12s\n','CNN','NLM', 'MED');
fprintf(fileID,'%12.8f %12.8f %12.8f\n',[psnrsCNN', psnrsNLM', psnrsMED']);
fclose(fileID);
%%
CNN_mean = mean(psnrsCNN)
CNN_std = std(psnrsCNN)
MED_mean = mean(psnrsMED)
MED_std = std(psnrsMED)
NLM_mean = mean(psnrsNLM)
NLM_std = std(psnrsNLM)