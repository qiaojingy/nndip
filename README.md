This folder contains the code for training the DCGAN and evaluation of the PSNR scores. The training code will periodically print results on dev set and test set. The structure of the code is adapted from the srez project by David Garcia (https://github.com/david-gpu/srez) and we modified the model, the input preprocessing and evaluation of PSNR. Please follow the precedure below to run the code. 


I. TRAINING DCGAN (code in train folder): 

Setup: 
Please make sure you have python3.4, pip and have installed these packages: 
  numpy, 
  scipy, 
  tensorflow r0.12
TensorFlow's API is different in different versions so please make sure to install version 0.12. 


Configuration: 
Please fill out the configuration in the top of nndip_main.py. You will need to specify training data folder, checkpoint folder and learning rates. You also need to specify the preprocessing piplines for different tasks: 

1. super-resolution
downsampling: downsampling ratio, can be 2, 4, 8. Otherwise set it to 1. 

2. denoising: 
noise: Guassian noise added to the image. Set it to 0 if you are doing other tasks. 

3. deconvolution: 
blur_sigma: Gaussian convolution window size. Set it to 0 if you are doing other tasks. 
blur_size: Gaussian convolution window size. Set it to 0 if you are doing other tasks. 


Run the code: 
python3 nndip_main.py --run train

The training code would periodically print training results. 


II. EVALUATION OF RESULTS (code in evaluation, sample data from our results are attached in eval): 

NLMF.m, PSNR.m, wiener.m, ADMM.m: helper functions. don't run
deconv.m: put input images, ADMM.m, wiener.m PSNR.m in the same folder and run
denoising.m: setup NLMF.m as in the instructions in HW6 in a the folder path '../HW6', and put input images, PSNR.m in the same denoising folder and run
interpolation.m: put PSNR.m and input images in the same folder and run
