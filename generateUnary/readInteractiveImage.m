function [RGBX,LUVX,LUVXN,Y,A,Aext,map] = readInteractiveImage(OneImage)
%%%
%   X : color image, Height*Width*3 (RGB channels 0:1)
%   Y : ground truth, 0 - background; 1 - foreground;
%   A : user-labeled,
%%%

if ~exist('OneImage','var')
    
    file = dir('./Dataset/images');
    imageList = cell(length(file)-2,1);
    for i = 3:length(file)
        imageList{i-2} = file(i).name(1:end-4);
    end
    j = 0;
    % read in image
    for i = 1:length(imageList)
        imageName = strcat('./Dataset/images/',imageList{i},'.jpg');
        gtName = strcat('./Dataset/images_gt/',imageList{i},'.png');
        annoName = strcat('./Dataset/images_labels/',imageList{i},'-anno.png');        
        extName = strcat('./Dataset/images_extendedseed/',imageList{i},'-anno.png');
        
        if ~exist(imageName,'file') || ~exist(gtName,'file') || ~exist(annoName,'file')|| ~exist(extName,'file')
            fprintf([imageList{i}, ' has incomplete files!\n']);
            continue;
        end
        j = j + 1;
        RGBX{j,1} = im2double(imread(imageName));
        LUVX{j,1} = colorspace('RGB->Luv',RGBX{j,1});
        Y{j,1} = im2double(imread(gtName));
        [A{j,1},map] = imread(annoName);        
        [Aext{j,1},map] = imread(extName);
             
    end
    
    fprintf(['In total ',num2str(j),' samples have been generated !\n']);
    
else
    
    imageName = strcat('./Dataset/images/',OneImage,'.jpg');
    gtName = strcat('./Dataset/images_gt/',OneImage,'.png');
    annoName = strcat('./Dataset/images_labels/',OneImage,'-anno.png');
    extName = strcat('./Dataset/images_extendedseed/',OneImage,'-anno.png');
    
    if ~exist(imageName,'file') || ~exist(gtName,'file') || ~exist(annoName,'file')|| ~exist(extName,'file')
        fprintf([OneImage, ' has incomplete files!\n']);
        return;
    end
    RGBX = im2double(imread(imageName));
    LUVX = colorspace('RGB->Luv',RGBX);
    Y = im2double(imread(gtName));
    [A,map] = imread(annoName);
    [Aext,map] = imread(extName);
    for k=1:3
        t1=LUVX(:,:,k);
        t1 = (t1-min(min(t1)))./(max(max(t1))-min(min(t1)));
        LUVXN(:,:,k)=t1;
    end
end
end