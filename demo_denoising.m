% demo denoising all experiments

close all
clear

pd = 8;
ps = 1;

f=imresize(im2double(imread('mix1.png')), 1);

if max(max(f)) > 2
    f = double(f)/255;
end

if size(f,3)==3
    f = rgb2gray(f);
elseif size(f,3)==2
    f = squeeze(f(:,:,1));
end

x = zeros(size(f));



K = 20;

sigma = 50;
sub = 1;
dis = 1;
% Gaussian observation matrix

randn('seed', 0)
y = f + (sigma/255)*randn(size(f));
%                 % BM3D baseline
%                 [psnrb(count)] = BM3D(f,y,sigma, 'np', 0)


% Internal GMM denoising
[z1, psnr1] = denoising('clean', f, 'sigma', sigma, 'pd', pd, 'K', K, 'sub', sub, 'dis', dis);


% External GMM denoising

% Start by training a GMM from external dataset
modelname = 'berkeley_trainpd8K20ps4';
% [modelname] = custommix('berkeley_train', 'jpg', 8, 20, 4);

[z2, psnr2] = denoising('clean', f, 'sigma', sigma, 'pd', pd, 'K', K, 'external', modelname, 'sub', sub, 'dis', dis);



% Internal and Multi-class GMM denoising
[z3, psnr3] = denoising('clean', f, 'sigma', sigma, 'pd', pd, 'K', K, 'class', 1, 'sub', sub, 'dis', dis);


% External and Multi-class GMM denoising
[z4, psnr4] = denoising('clean', f, 'sigma', sigma, 'pd', pd, 'K', K, 'class', 1, 'external', modelname, 'sub', sub, 'dis', dis);




