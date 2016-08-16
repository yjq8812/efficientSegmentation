function [X,Y,A] = useInteractiveImage()

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
    
    if ~exist(imageName,'file') || ~exist(gtName,'file') || ~exist(annoName,'file')
        fprintf([imageList{i}, ' has incomplete files!\n']);
        continue;
    end
    j = j + 1;
X{j,1} = im2double(imread(imageName));
Y{j,1} = im2double(imread(gtName));
A{j,1} = im2double(imread(annoName));
end

fprintf(['In total ',num2str(j),' samples have been generated !\n']);
       
end