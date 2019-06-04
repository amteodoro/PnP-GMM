function [saveas] = custommix(database, extension, pd, K, ps)

%% Train a custom mixture from external database of clean images

fprintf('Custom mixture for %s dataset does not exist. Training the mixture...\n', database)

% Might need to change this according to OS

path = pwd;
% idcs   = strfind(path,'/');
% path = path(1:idcs(end)-1);
directory = strcat(path, '/Database/');
directory = strcat(strcat(directory, database), '/');
filetype = strcat('/*.', extension);
srcFiles = dir(strcat(directory, filetype));  % the folder in which ur images exists

db_patches = [];

rng('default');
samples = randperm(length(srcFiles), min(length(srcFiles), 1000));
if isempty(samples)
    disp('No images found. Please check if the path is correct.')
else
    
    numim = 0;
    usedImg = [];
    for i = samples
        usedImg = [usedImg; i];
        filename = strcat(directory,srcFiles(i).name);
        try
            x = rgb2gray(imread(filename, extension));
        catch
            x = (imread(filename, extension));
        end
        
        if max(x(:)) > 2
            x = double(x)/255;
        end
        
        y_patches_total = [];
        
        xx = wextend(2,'sym',x,[pd,pd]);
        
        patches = im2colstep(xx,[pd,pd],[ps,ps]);
        
        patches_dc=mean(patches);
        patches= bsxfun(@minus, patches , patches_dc); % Remove DC
        
        y_patches_total = [y_patches_total, patches];
        
        % Data augmentation
        %             yyrot = xx;
        %             for i = 1:3
        %
        %                 yyrot = rot90(yyrot);
        %                 y_patches_ac = im2colstep(yyrot,[pd,pd],[ps,ps]);
        %                 y_patches_total = [y_patches_total, y_patches_ac];
        %
        %             end
        
        x_patches_ac = y_patches_total;
        
        db_patches = [db_patches, x_patches_ac];
        numim = numim + 1;
        
        if size(db_patches, 2) > 2*512^2
            % Too many patches
            break
        end
    end
    
    fprintf('Training Mixture with %d patches from %d images. \n', size(db_patches, 2), numim)
    
    [prob,Scomp,Ucomp] = EM_zeromean(db_patches,K,(1/255));
    
    saveas = strcat(strcat('',database), strcat(strcat('pd', num2str(pd), strcat(strcat('K', num2str(K)), strcat('ps', strcat(num2str(ps),''))))));
    
    % Save to .mat file to reuse
    save(saveas, 'prob', 'Scomp', 'Ucomp');
end
