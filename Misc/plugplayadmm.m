function [x, isnr, psnr, mu] = plugplayadmm(image, experiment_number, varargin)

% test for number of required parametres
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end

% % Add paths
% if ~exist('BM3D.m','file')
%     addpath(genpath(pwd))
% end


% Set default parameters
verbose = 1;
iter = 100;
tolA = 1e-4;
updateiter = inf; % inf = no refinement
pd = 8;
K = 20;
ps = 4;
database = 'berkeley_trainpd8K20ps4';
dataset = 'berkeley_train';
extension = 'png';
single_mixture = 1;
filt = [];
blurred = [];


% Read the optional parameters
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'MU'
                mu = varargin{i+1};
            case 'MAXITER'
                iter = varargin{i+1};
            case 'VERBOSE'
                verbose = varargin{i+1};
            case 'PD'
                pd = varargin{i+1};
            case 'K'
                K = varargin{i+1};
            case 'PS'
                ps = varargin{i+1};
            case 'SINGLE'
                single_mixture = varargin{i+1};
            case 'DATABASE'
                database = varargin{i+1};
            case 'UPDATEITER'
                updateiter = varargin{i+1};
            case 'FILTER'
                filt = varargin{i+1};
            case 'BLURRED'
                blurred = varargin{i+1};
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end
f = image;
[fM,fN]=size(f);

if ~isempty(filt)
    [f_blur, R, RT, sigma, hshift, h] = blurimage(f, experiment_number, filt);
else
    [f_blur, R, RT, sigma, hshift, h] = blurimage(f, experiment_number);
end


%%%% Create a blurred and noisy observation
randn('seed',0);


if ~isempty(blurred)
    y = blurred;
else
    y = f_blur + sigma*randn(fM, fN);
end
% figure(20), set(gcf,'position',[10 700 600 600])
% figure(20), imagesc(y), colormap(gray(255)), axis off, axis equal


bsnr=10*log10(norm(f_blur(:)-mean(f_blur(:)),2)^2 /sigma^2/fM/fN);
psnr_y = PSNR(f,y,1,0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Observation BSNR: %4.2f, PSNR: %4.2f\n', bsnr, psnr_y);

H_FFT = fft2(hshift);
H2 = abs(H_FFT).^2;

invLS = @(x, mu, filter_FFT) (1/mu)*( x - real( ifft2( filter_FFT.*fft2( x ) ) ) );

RTy = RT(y);
x0 = RTy;

r0 = R(x0) - y;

% ADMM Loop
for i = 1:iter
    if i == 1
        filter_FFT = H2./(H2 + mu);
        if ~iscell(database)
            try
                load(database);
                K = length(prob);
                pd = sqrt(size(Scomp,1));
                ps = 1;
            catch
                fprintf('That database does not exist. Training a new one...\n');
                database = custommix(dataset, extension, pd, K, 1);
                load(database)
            end
        else
            prob = database{1};
            Scomp = database{2};
            Ucomp = database{3};
            if length(database) == 4
                supportvar = database{4};
            end
        end
        % initializing
        x = x0;        
        v = x;
        r = r0;        
        d = r;
    end
    dprev = d;
    vprev = v;
    
    
    r = mu*(v + dprev) + RTy;
    
    x = (invLS(r,mu, filter_FFT));

    sigma_hat = NoiseEstimation(x-dprev, 8);
    if sigma_hat < 2/255
        sigma_hat = 2/255;
    end
    
    % Update the mixture
    if mod(i-1, updateiter) == 0 && i > 1
        
        im = reshape(x-dprev, [size(f,1), size(f,2)]);
        
        xx = wextend(2,'sym',im,[pd-1,pd-1]);
        scale = 1;
        auxim = xx;

        patches = im2colstep(auxim,[pd,pd],[ps,ps]);
        
        patches_dc=mean(patches);
        patches= bsxfun(@minus, patches , patches_dc); % Remove DC
        
        [prob,Scomp,Ucomp,~, ~, supportvar] = EM_zeromean(patches,K, sigma_hat/scale);
        
        database = {prob; Scomp; Ucomp};
        
    end
    
    min_im = 0;
    scale = 1;
    auxim = x-dprev;
%     max_im = max(max(auxim(:)),1);
%     min_im = min(min(auxim(:)),0);
%     scale = max_im - min_im;

    auxim = (auxim - min_im)/scale;
    
    if i >= inf
        [v, ~, supportvar] = denmix_deblurring(auxim, f, sigma_hat/scale, database, single_mixture, supportvar);
    else
        [v, ~, supportvar] = denmix_deblurring(auxim, f, sigma_hat/scale, database, single_mixture);
    end

    v = v*scale + min_im;

    d = dprev - (x - v);

    primalresidual(i) = norm(v(:) - x(:),2);
    dualresidual(i) = norm(-mu*(v(:) - vprev(:)),2);
    
    psnr1(i) = PSNR(f,x,1,0);
    ISNR(i) = psnr1(i)-psnr_y;
    
    if ~mod(i,5) || i == 1
        if verbose
            fprintf(1,'Iteration: %d;\t ISNR = %4.2f;\t Primal: %e; \t Dual: %e\n',...
                i, ISNR(i),primalresidual(i), dualresidual(i));
            %figure(5), semilogy(c, 'x'), drawnow
%             figure(33), set(gcf,'position',[550 700 600 600])
%             imagesc(v), axis off, axis equal, colormap gray, drawnow
            
        end
    end    
end

psnr = PSNR(f,x,1,0);
isnr = psnr-psnr_y;

disp('-------- Results --------');
fprintf('Final estimate ISNR: %4.2f, PSNR: %4.2f.\n', isnr, psnr);

