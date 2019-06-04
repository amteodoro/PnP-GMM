function [x, psnr, supportvar] = denmix_deblurring(y, f, sigma, database, single_mixture, supportvar)
%% Denoise using previously trained mixture
%
%   Inputs: y - noisy image in [0,1]
%           f - clean image for PSNR purposes
%           sigma - noise variance (Note that the input image is expected to be in [0,1])
%           database - string with the name of a .mat with the parameters
%           of a previously computed mixture, or a cell array with those
%           parameters
%           single_mixture - flag: 1 - use only the mixture from the
%           previous argument;
%                                  0 - use several mixtures and patch
%                                  classification
%   Output: x - clean image in [0,1]
%           psnr - Peak signal to noise ratio in dB


if ~iscell(database)
    try
        load(database);
    catch
        fprintf('Database: %s does not exist.\n', database)
    end
else
    prob = database{1};
    Scomp = database{2};
    Ucomp = database{3};
    if length(database) == 4
        supportvar = database{4};
    end
end
if ~exist('supportvar', 'var')
    %     fprintf('No fixed support.\n')
    supportvar = [];
end
pd = sqrt(size(Ucomp,1));
ps = 1;
K = size(Ucomp,3);

global_dc = 0;
y = y - global_dc;
yy = wextend(2,'sym',y,[pd-1,pd-1]);
y_patches_ac = im2colstep(yy,[pd,pd],[ps,ps]);

[dimens,num] = size(y_patches_ac);
[m, n] = size(f);

y_patches_dc=mean(y_patches_ac);
y_patches_ac= bsxfun(@minus, y_patches_ac , y_patches_dc);

scale = 1;
% max_im = max(max(y_patches_ac(:)), 0.5);
% min_im = min(min(y_patches_ac(:)), -0.5);
% scale = max_im - min_im;

y_patches_ac = (y_patches_ac)/scale;

x_hat_patches = y_patches_ac;

if single_mixture
    [x_hat_patches,post_cov, supportvar] = GMM_inference(y_patches_ac,zeros(dimens,K),prob,Scomp,Ucomp,sigma, []);
    weights = (1./(post_cov+eps));
    

    
    x_hat_patches = bsxfun(@plus, x_hat_patches , y_patches_dc);
    
    x_hat_patches = min(max(x_hat_patches,-global_dc),1-global_dc);
    
    x_hat_patches = (x_hat_patches)*scale;
    
    
    x_hat_raw = col2imstep(x_hat_patches.*(weights),size(yy),[pd,pd],[ps,ps]);
    normalize = col2imstep((weights),size(yy),[pd,pd],[ps,ps]);
    x_hat1 = x_hat_raw ./ normalize;
    
    x_hat = x_hat1(pd:(m+pd-1),pd:(n+pd-1)) + global_dc;
    
    x = x_hat;
    
    x = min(max(x_hat,0),1);
    x_hatb = x;
    psnr = 10*log10(1/mean( (f(:)-x(:)).^2));
    PSNR_best = psnr;
    
else
    
    p1 = prob;
    S1 = Scomp;
    U1 = Ucomp;
    
    l1 = length(p1);
    
    
    %% Try other databases
    database = 'text_trainpd8K20ps1';
    if exist(strcat(database, '.mat'), 'file')
        load(database);
    else
        database = 'text_train';
        extension = 'png';
        K = 20;
        [saveastext] = custommix(database, extension, pd, K);
        load(saveastext);
    end
    
    textp = prob;
    textU = Ucomp;
    textS = Scomp;
    %textCov = cov(:,:,:);
    
    l2 = length(prob);
    
    database = 'mri_trainpd8K20ps4';
    
    if exist(strcat(database, '.mat'), 'file')
        load(database);
    else
        database = 'mri_train';
        extension = 'tif';
        K = 20;
        [saveastext] = custommix(database, extension, pd, K);
        load(saveastext);
    end
    
    brainp = prob;
    brainU = Ucomp;
    brainS = Scomp;
    %brainCov = cov(:,:,:);
    
    l3 = length(prob);
    
    database = 'algore_trainpd8K20ps1';
    if exist(strcat(database, '.mat'), 'file')
        load(database);
    else
        database = 'algore_train';
        extension = 'png';
        K = 20;
        [saveastext] = custommix(database, extension, pd, K);
        load(saveastext);
    end
    
    facep = prob;
    faceU = Ucomp;
    faceS = Scomp;
    %faceCov = cov(:,:,:);
    
    
    l4 = length(prob);
    
    database = 'fingerprints_trainpd8K20ps4';
    if exist(strcat(database, '.mat'), 'file')
        load(database);
    else
        database = 'fingerprints_train';
        extension = 'tif';
        K = 20;
        [saveastext] = custommix(database, extension, pd, K);
        load(saveastext);
    end
    
    fingerp = prob;
    fingerU = Ucomp;
    fingerS = Scomp;
    %fingerCov = cov(:,:,:);
    
    
    l5 = length(prob);
    
    p = [p1; textp; brainp; facep; fingerp]/5;
    S = [S1, textS, brainS, faceS, fingerS];
    U = cat(3, U1, cat(3, textU, cat(3, brainU, cat(3, faceU, fingerU))));
    cov = zeros(size(U));
    %[x_hat, mapmode] = mix_MAP_svd_zero_mean(y_patches_ac,p,S,U,sigma);
    
    % clusters = reshape(mapmode, [sqrt(length(mapmode)) sqrt(length(mapmode))]);
    %
    % imidx = (clusters>l2+l3+l4+l5) + (clusters>l3+l4+l5) + (clusters>l4+l5) + (clusters>l5);
    alpha_expansion = 1;
    for th = 1/5 % 1 over number of classes;
        
        %[~, ~, normindic] = mix_Wiener_zeromean_svd3(y_patches_ac, p, S, U, sigma);
        for mod = 1:length(p)
            cov(:,:,mod) = U(:,:,mod)*bsxfun(@times,S(:,mod),U(:,:,mod)');
        end
        
        [indic] = computePost(y_patches_ac, p, zeros(dimens, length(p)), cov + repmat(sigma^2*eye(pd^2),[1, 1, length(p)]));
        
        %         for mod = 1:pathsNum
        %             [UcompPath(:,:,mod),S] = svd(covpp(:,:,mod));
        %             ScompPath(:,mod) = max(diag(S),1e-10);
        %             indic(mod,:) = prob(mod1)*multinorm_svd_zeromean(ysub,...
        %                 ScompPath(:,mod)+sigma2,UcompPath(:,:,mod));
        %         end
        
        A = bsxfun(@minus,indic,max(indic,[],1));
        B = exp(A);
        normindic = bsxfun(@rdivide,B,sum(B,1));
%             for mod=1:length(p)
%                 normindic(mod,:) = p(mod)*multinorm_svd_zeromean(y_patches_ac,S(:,mod)+sigma^2,U(:,:,mod));
%             end
%             normindic = bsxfun(@times,normindic,1./(realmin + sum(normindic,1)));
        supportvar = normindic;
        
        
        aux(1,:) = sum(normindic(1:l1,:));
        aux(2,:) = sum(normindic(l1+1:l2+l1,:));
        aux(3,:) = sum(normindic(l2+l1+1:l3+l2+l1,:));
        aux(4,:) = sum(normindic(l3+l2+l1+1:l4+l3+l2+l1,:));
        aux(5,:) = sum(normindic(l4+l3+l2+l1+1:end,:));
        
        if ~alpha_expansion
            
            [val,index] = max(aux);
            
            labels = ones(size(val));
            
            labels(val>th) = index(val>th);
            
            
        else
            aux_res = reshape(aux(1,:), [m + pd - 1, n + pd - 1]);
            aux_res(:,:,2) = reshape(aux(2,:), [m + pd - 1, n + pd - 1]);
            aux_res(:,:,3) = reshape(aux(3,:), [m + pd - 1, n + pd - 1]);
            aux_res(:,:,4) = reshape(aux(4,:), [m + pd - 1, n + pd - 1]);
            aux_res(:,:,5) = reshape(aux(5,:), [m + pd - 1, n + pd - 1]);
            
            probMatrix = log(aux_res + eps);
            beta = 1.5;
            Sc = ones(5) - eye(5);
            %Sc = [0 2 2 2 2; 2 0 1 1 1; 2 1 0 1 1; 2 1 1 0 1; 2 1 1 1 0];
            gch = GraphCut('open', -probMatrix, beta.*Sc);
            [gch, labels] = GraphCut('expand', gch);
            gch = GraphCut('close', gch);
            labels = labels + 1;
            
            
        end
        
        imidx2 = reshape(labels, [m + pd - 1, n + pd - 1]);
        threeLabelIm = zeros(size(imidx2));
        threeLabelIm(imidx2 == 2) = 3;
        threeLabelIm(imidx2 == 1) = 1;
        threeLabelIm(imidx2 == 3) = 2;
        
%             figure(50), set(gcf,'position',[10 10 600 600]), imagesc(threeLabelIm), axis off, axis equal, colormap gray, drawnow
        
%         normindic1 = normindic(1:l1,labels == 1)./repmat(sum(normindic(1:l1,labels == 1)), l1,1);
        if ~isempty(y_patches_ac(:,labels == 1))
            [x_hat_patches(:,labels == 1),post_cov(:,labels == 1)] = GMM_inference(y_patches_ac(:,labels == 1), zeros(dimens,K),p1, S1, U1, sigma,[]);
        end
        %         normindic2 = normindic(l1+1:l2+l1,labels == 2)./repmat(sum(normindic(l1+1:l2+l1,labels == 2)), l2,1);
        if ~isempty(y_patches_ac(:,labels == 2))
            [x_hat_patches(:,labels == 2),post_cov(:,labels == 2)] = GMM_inference(y_patches_ac(:,labels == 2), zeros(dimens,K), textp, textS, textU, sigma,[]);
        end
        
        %         normindic3 = normindic(l2+l1+1:l3+l2+l1,labels == 3)./repmat(sum(normindic(l2+l1+1:l3+l2+l1,labels == 3)), l3,1);
        if ~isempty(y_patches_ac(:,labels == 3))
            [x_hat_patches(:,labels == 3),post_cov(:,labels == 3)] = GMM_inference(y_patches_ac(:,labels == 3), zeros(dimens,K), brainp, brainS, brainU, sigma,[]);
        end
        
        %         normindic4 = normindic(l3+l2+l1+1:l4+l3+l2+l1,labels == 4)./repmat(sum(normindic(l3+l2+l1+1:l4+l3+l2+l1,labels == 4)), l4,1);
        if ~isempty(y_patches_ac(:,labels == 4))
            [x_hat_patches(:,labels == 4),post_cov(:,labels == 4)] = GMM_inference(y_patches_ac(:,labels == 4), zeros(dimens,K), facep, faceS, faceU, sigma,[]);
        end
        
        %         normindic5 = normindic(l4+l3+l2+l1+1:end,labels == 5)./repmat(sum(normindic(l4+l3+l2+l1+1:end,labels == 5)), l5,1);
        if ~isempty(y_patches_ac(:,labels == 5))
            [x_hat_patches(:,labels == 5),post_cov(:,labels == 5)] = GMM_inference(y_patches_ac(:,labels == 5), zeros(dimens,K), fingerp, fingerS, fingerU, sigma,[]);
        end
        
        weights = 1./post_cov;
        %weights = ones(size(y_patches_ac));
        
        %x_hat_patches = add_dc( x_hat_patches , y_patches_dc ,'columns');
        x_hat_patches = bsxfun(@plus, x_hat_patches , y_patches_dc);
        
        x_hat_patches = min(max(x_hat_patches,-global_dc),1-global_dc);
        
        x_hat_raw = col2imstep(x_hat_patches.*(weights),size(yy),[pd,pd],[ps,ps]);
        normalize = col2imstep((weights),size(yy),[pd,pd],[ps,ps]);
        x_hat1 = x_hat_raw ./ normalize;
        
        x_hat = x_hat1(pd:(m+pd)-1,pd:(n+pd)-1) + global_dc;
        
        x_hat = min(max(x_hat,0),1);
        
        psnr = 10*log10(1/mean( (f(:)-x_hat(:)).^2));
             
    end
    
    x_hatb = x_hat;
    PSNR_best = psnr;
end

patch_vars = var(y_patches_ac);

PSNR_best= psnr;
for var_th = 0.5:0.02:1.5
    
    aux_patches = x_hat_patches;
    weightsflat = weights;
    
    flat_patches = find(patch_vars < var_th*sigma^2);
    
    aux_patches(:,flat_patches) = kron(y_patches_dc(flat_patches),ones(dimens,1));
    
    %                                     weightsflat(:,flat_patches) = dim / sigma_hat;
    non_flat_patches = find(patch_vars >= var_th*sigma^2);
    
    %aux_patches(:,non_flat_patches) = add_dc(aux_patches(:,non_flat_patches), y_patches_dc(non_flat_patches), 'columns');
    
    aux_patches(:,non_flat_patches) = bsxfun(@plus, aux_patches(:,non_flat_patches) , y_patches_dc(non_flat_patches));
    
    aux_patches = min(max(aux_patches,-global_dc),1-global_dc);
    
    x_hat_raw = col2imstep(aux_patches.*(weightsflat),size(yy),[pd,pd],[ps,ps]);
    normalize = col2imstep((weightsflat),size(yy),[pd,pd],[ps,ps]);
    x_hat1 = x_hat_raw ./ normalize;
    
    x_hat = x_hat1(pd:(m+pd-1),pd:(n+pd-1)) + global_dc;
    
    x_hat = min(max(x_hat,0),1);
    
    PSNR_estim = -10*log10(var( (f(:)-x_hat(:))));
    
    if PSNR_estim > PSNR_best
        x_hatb = x_hat;
        %                 x_hateb = var_th;
        PSNR_best = PSNR_estim;
        %                 x_haty = x_hatx;
    end
    
end


x = x_hatb;
psnr = PSNR_best;
end