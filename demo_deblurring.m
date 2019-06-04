
close all
clear

pd = 8;
K = 20;

experiment = 1;

maxIter = 200;
updateIter = 100;%floor(maxIter/2);%inf;%

singleMix = 1; % 1 - only one model; 0 - multiple models

mu = 0.05;

f=imresize(im2double(imread('mix1.png')), 1);

if max(max(f)) > 2
    f = double(f)/255;
end

if size(f,3)==3
    f = rgb2gray(f);
elseif size(f,3)==2
    f = squeeze(f(:,:,1));
end

image = f;

% Start by training a GMM from external dataset
database = 'berkeley_trainpd8K20ps4';
[database] = custommix('berkeley_train', 'jpg', 8, 20, 4);

[x, isnr, psnr] = plugplayadmm(image, experiment, 'mu', mu, 'database', database, 'maxiter', maxIter, 'updateiter', updateIter, 'single', 1, 'pd', pd);

[x, isnr, psnr] = plugplayadmm(image, experiment, 'mu', mu, 'database', database, 'maxiter', maxIter, 'updateiter', updateIter, 'single', 0, 'pd', pd);

