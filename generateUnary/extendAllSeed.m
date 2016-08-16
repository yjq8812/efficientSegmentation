clear; 
file = dir('./Dataset/images');
mkdir('./Dataset/images_extendedseed');
imageList = cell(length(file)-2,1);
for i = 3:length(file)
    imageList{i-2} = file(i).name(1:end-4);
end
j = 1;
% read in image
for i = 2:length(imageList)
    imageName = strcat('./Dataset/images/',imageList{i},'.jpg');
    gtName = strcat('./Dataset/images_gt/',imageList{i},'.png');
    annoName = strcat('./Dataset/images_labels/',imageList{i},'-anno.png');
    
    if ~exist(imageName,'file') || ~exist(gtName,'file') || ~exist(annoName,'file')
        fprintf([imageList{i}, ' has incomplete files!\n']);
        continue;
    end
    j = j+1;
    GT = im2double(imread(gtName));
    GT(find(GT>0)) = 1;
    [A,map] = imread(annoName);
    fprintf(['Calculating for ' imageList{i},' ...\n']);
    [Yextended] = extendedSeed(GT,A);
    
    filename = strcat('./Dataset/images_extendedseed/',imageList{i},'-anno.png');
    imwrite(Yextended,map,filename,'png');
    
    
end

fprintf(['In total ',num2str(j),' samples have been generated !\n']);