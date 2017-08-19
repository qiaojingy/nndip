%% 
clear; close all; clc;
%%
target = 'Face';
scale = 4;
%%
len = 64;
imgTests = im2double(imread(['testInterp' target num2str(scale) '.png']));
figure;
imshow(imgTests);
%%
imgNum = 8;
imgResult = zeros(len*imgNum, len*4, 3);

psnrsInter = zeros(1, length(imgNum));
psnrsCNN = zeros(1, length(imgNum));
for i = 1:imgNum
    img = imgTests((i-1)*64+1: i*64, 129:end, :);
    imgCNN = imgTests((i-1)*64+1: i*64, 65:128, :);
    imgResult((i-1)*64+1: i*64, 1:64, :) = img;
    imgResult((i-1)*64+1: i*64, 193:end, :) = imgCNN; 
    [numRows, numCols, ~] = size(img);
    imgSmall = imresize(img, 1/scale, 'bicubic');
    imgSub = imresize(imgSmall, [numRows, numCols], 'nearest');
    imgResult((i-1)*64+1: i*64, 65:128, :) = imgSub;
    imgInter = imresize(imgSmall, [numRows, numCols], 'bicubic');
    imgResult((i-1)*64+1: i*64, 129:192, :) = imgInter;
    
    psnrsInter(i) = PSNR(imgInter, img);
    psnrsCNN(i) = PSNR(imgCNN, img);
    if i == 1
        imgSample = zeros(64*2, 64*2, 3);
        imgSample(1:64, 1:64, :) = img;
        imgSample(1:64, 65:end, :) = imgSub;
        imgSample(65:end, 1:64, :) = imgInter;
        imgSample(65:end, 65:end, :) = imgCNN;
        imwrite(imgSample, ['sample_' target '_result'...
            num2str(scale) '.jpg']);
    end
end
%%
figure; imshow(imgResult)
imwrite(imgResult, ['interpResult' target num2str(scale) '.png']);
fileID = fopen(['interp' target 'PSNR' num2str(scale) '.txt'], 'w');
fprintf(fileID, '%12s %12s\n','CNN','Bicubic');
fprintf(fileID,'%12.8f %12.8f\n',[psnrsCNN', psnrsInter']);
fclose(fileID);
%%
CNN_mean = mean(psnrsCNN)
CNN_std = std(psnrsCNN)
Inter_mean = mean(psnrsInter)
Inter_std = std(psnrsInter)
