%% Compressed Sensing ADMM with GMM as plug and play prior

close all
clear
%clf


tic
iter0 = 300;
updateIter = 100;%ceil(iter0/2);%inf;%

database = 'berkeley_trainpd8K20ps4';
if exist(strcat(database, '.mat'), 'file') == 0
    % Start by training a GMM from external dataset
    [database] = custommix('berkeley_train', 'jpg', 8, 20, 4);
end

singleModel = 1;


mu = 0.1; % Set ADMM parameter

rate = 0.3; % Set compression rate


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
if max(max(f)) > 2
    f = double(f)/255;
end

if size(f,3)==3
    f = rgb2gray(f);
elseif size(f,3)==2
    f = squeeze(f(:,:,1));
end

f0 = f;

pd = 8;
ps = 1;
K = 20;

% Gaussian observation matrix

randn('seed', 0)

block = 16;
ps = block;
sigma = 0/255;
N = block^2;

M = floor(rate*N);

phi = orth(randn(N, N))';
phi = phi(1:M, :);

roundTargets = block*[1:64];
vRounded = interp1(roundTargets,roundTargets,size(f0),'nearest'); % stretch to multiples of block size

f = imresize(f0, vRounded, 'bicubic');
f1 = f;
[row, col] = size(f1);


best_psnr_run = -inf;
f = wextend(2,'sym',f,[block,block]);

[m, n] = size(f);

R = @(x) phi*x;
RT = @(x) phi'*x;

normf = norm(f,'fro')^2;

iter = iter0;

invPhi = (1/mu)*(eye(N) - phi'*(( (phi*phi') + mu*eye(M) )\phi));

fprintf('Iterating...\n')

patches_clean = im2colstep(f, [block, block], [ps, ps]);
weights = ones(size(patches_clean));
normalize = col2imstep(weights, size(f), [block, block], [ps, ps]);
criterion = 0;
prev_NMSE = inf;
prev_ISNR = -inf;
best_nmse = inf;
best_psnr = -inf;
y = R(patches_clean) + sigma*randn(M,size(patches_clean,2));

RTy = RT(y);

for i = 1:iter
    
    x0 = RTy;
    
    r0 = R(x0) - y;
    if i == 1 % initializing
        x = x0;
        v2 = zeros(size(x));        
        d = v2 - x;
        
        Rx = R(x);
        
        resid =  y-Rx;
        psnr_y = 10*log10(1/mean( (f(:)-x0(:)).^2));
    end
    vprev = v2;
    xprev = x;
    
    r = mu*(v2 + d) + RTy;
    
    x = invPhi*r;
    
    im = col2imstep(x-d, size(f), [block, block], [ps, ps]);
    im = im./normalize;
    
    sigma_hat = NoiseEstimation(im, 8);
    if sigma_hat < 2/255
        sigma_hat = 2/255;
    end
    
%     if mod(i-1, updateIter) == 0 && i > 1
%         if ~iscell(database)
%             load(database);
%         end
%         pd = 8;
%         K = 20;
%         
%         xx = wextend(2,'sym',im,[pd,pd]);
%         
%         patches = im2colstep(xx,[pd,pd],[1,1]);
%         
%         scale = 1;
%         min_im = 0;
%         
%         patches = (patches-min_im)/scale;
%         
%         patches_dc=mean(patches);
%         patches= bsxfun(@minus, patches , patches_dc); % Remove DC
%         
%         [prob,Scomp,Ucomp,~, supportvar] = ...
%             EM_zeromean(patches,K,sigma_hat/scale);
%         
%         database = {prob; Scomp; Ucomp};
%     end
    
    scale = 1;
    min_im = 0;
    im = (im-min_im)/scale;
    
    [v, psnrv] = denmix_cs(im, f, sigma_hat/scale, database, singleModel);
    
    v = v*scale + min_im;
    v2 = im2colstep(v, [block, block], [ps, ps]);
    
    
    d = d - (x - v2);
    
    psnrx(i) = 10*log10(1/mean( (f(:)-v(:)).^2));
    
    ISNR(i) = psnrx(i)-psnr_y;
    
    dISNR = prev_ISNR - ISNR(i);
    prev_ISNR = ISNR(i);
    
    if psnrx(i) > best_psnr
        best_psnr = psnrx(i);
        best_i = i;
        best_x = x;
    end
    
    if (mod(i,10)==0 || i == 1)
        fprintf(1,'Iteration: %d;\t PSNR: %4.2f; PSNR2: %4.2f; ISNR: %4.2f\n',...
            i, psnrx(i), psnrv, ISNR(i))
    end
    
end

est = col2imstep(x, size(f), [block, block], [ps, ps]);
est = est./normalize;
est = max(0, min(est, 1));
est = est(block+1:block + row, block+1:block + col);
nmse = norm(f1-est,'fro')^2/normf;

disp('-------- Results --------');
fprintf('Final estimate NMSE: %4.4f \t (dB: %4.2f). \tBest NMSE: %4.4f \t (dB: %4.2f); \t PSNR: %4.2f dB\n', nmse, 10*log10(nmse), best_nmse, 10*log10(best_nmse), 10*log10(1/mean( (f1(:)-est(:)).^2)));
