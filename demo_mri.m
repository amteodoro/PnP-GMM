close all
clear


fN = 256;
fM = fN;

clean = phantom(128);

if max(max(clean)) > 2
    clean= double(clean)/255;
end

if size(clean,3)==3
    clean = rgb2gray(clean);
    %         noisy = rgb2gray(noisy);
elseif size(clean,3)==2
    clean = squeeze(clean(:,:,1));
    %         noisy = squeeze(noisy(:,:,1));
    
end
complex = 0;
f = clean;


f = f./max( abs(f(:)) );

randn('seed', 0)
angles = 45;

database1 = 'berkeley_trainpd8K20ps4'; % Generic

database1 = 'mri_trainpd8K20ps4'; % Class specific

database = database1;

if exist(strcat(database, '.mat'), 'file') == 0
    % Start by training a GMM from external dataset
    [database] = custommix('mri_train', 'tif', 8, 20, 4);
end

[A, At, P]    =  Radial_Line_Sensing(f, angles);

rate = 0.1;

R = A;
RT = At;

load(database);
K = length(prob);
pd = sqrt(size(Ucomp,1));
ps = 1;

[fN, fM] = size(f);

sigma = 0;

sigma2 = sigma^2;

y = R(f);
randn('seed',0);

y = y + sigma*(randn(size(y)) + 1i*randn(size(y)));
% figure(2), imagesc(mask), colormap gray, drawnow

initial_estimate = RT(y);


err_db=10*log10(1/(mean((abs(initial_estimate(:))-abs(f(:))).^2)));
fprintf('Iteration: 0; \tPSNR: %4.2f\n', err_db)

global_dc = 0;
initial_estimate = initial_estimate - global_dc;

iter = 200;
probi = prob;
Scompi = Scomp;
Ucompi = Ucomp;
updateIter = 100;
count = 1;
run1 = 1;
resultsAll = [];
mu = 0.1;
best_psnr_loop = -inf;


current_estimate = initial_estimate;
psnr_best = 0;
best_psnr = -inf;
psnr_prev = 0;
psnr = 0;
z_hat = initial_estimate;
d = 0;
prob = probi;
Scomp = Scompi;
Ucomp = Ucompi;
for i=1:iter
    
    filter_FFT = 1./(1 + mu);
    
    invLS = @(x, mu) (1/mu)*( x - filter_FFT.*( RT( R( x ) ) ) );
    
    x_hat = invLS(RT(y) + mu*(z_hat + d), mu);
    x_hat = reshape(x_hat, [fN, fM]);
    
    if complex == 1
        im = abs(x_hat-d);
    else
        im = real(x_hat-d);
    end
    
    [sigma_hat] = NoiseEstimation(im, 8);
    
    if sigma_hat < 2/255
        sigma_hat = 2/255;
    end
    
    if mod(i-1, updateIter) == 0 && i > 1
        
        pd = 8;
        K = 20;
        
        xx = wextend(2,'sym',im,[pd,pd]);
        
        patches = im2colstep(xx,[pd,pd],[ps,ps]);
        
        patches_dc=mean(patches);
        patches= bsxfun(@minus, patches , patches_dc); % Remove DC
        
        scale = 1;
        patches = (patches)/scale;
        
        [prob,Scomp,Ucomp,~, ~, supportvar] = ...
            EM_zeromean(patches,K,sigma_hat/scale);
        
        database = {prob; Scomp; Ucomp};
    end
    
    
    [z_hat, psnrz] = denmix_cs(im, f, sigma_hat, database, 1);
    
    d = d - x_hat + z_hat;
    
    mse(i) = norm(abs(f)-abs(z_hat),'fro')^2 /numel(f);
    err_db(i)=10*log10(1/(mean((abs(x_hat(:))-abs(f(:))).^2)));
    if ~mod(i, 5) || i == 1;
        fprintf('Iteration: %d; \t MSE: %e (%4.2f); \t PSNR: (%4.2f)\n', i, mse(i), 10*log10(1/(mse(i))), err_db(i))
    end
    
    
end

