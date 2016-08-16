% function [RGBFeatures,LUVFeatures,Y,A] = generateUnaryFeatures(segOptions)
% generate more features from color images
% all the features are normalized to [0,1]

if ~exist('segOptions','var')
   opts=segGSCOpts();
end

file = dir('./Dataset/images');
mkdir('./Dataset/images_RGB');
mkdir('./Dataset/images_LUV');
mkdir('./Dataset/images_RGBGMM');
mkdir('./Dataset/images_LUVGMM');
mkdir('./Dataset/images_RGBdistFG');
mkdir('./Dataset/images_RGBdistBG');
mkdir('./Dataset/images_LUVdistFG');
mkdir('./Dataset/images_LUVdistBG');
mkdir('./Dataset/images_RGBGMMdistFG');
mkdir('./Dataset/images_RGBGMMdistBG');
mkdir('./Dataset/images_LUVGMMdistFG');
mkdir('./Dataset/images_LUVGMMdistBG');
mkdir('./Dataset/images_EuclidianFG');
mkdir('./Dataset/images_EuclidianBG');
imageList = cell(length(file)-2,1);
for i = 3:length(file)
    imageList{i-2} = file(i).name(1:end-4);
end
for i = 1:length(imageList)
% ------ RGB channels & CIELUV channels, H*W*3 ------ 
[RGBFeatures,LUVFeatures,LUVFeaturesN,Y,~,Aext,map] = readInteractiveImage(imageList{i});

% ------ GMM likelihood ------ 
% obj.gmmLikeli_gamma=0.05 may need to be choose by validation
RGBposteriors = getPosteriorImage(RGBFeatures,Aext,opts);
LUVposteriors = getPosteriorImage(LUVFeatures,Aext,opts);

% ------ 10 distance transform features ------ (40 in A.Osokin's paper)

[RGBdistFG,RGBdistBG] = distanceFeature(RGBFeatures,Aext,opts,opts.geoGamma);
[LUVdistFG,LUVdistBG] = distanceFeature(LUVFeatures,Aext,opts,opts.geoGamma);
[RGBGMMdistFG,RGBGMMdistBG] = distanceFeature(RGBposteriors,Aext,opts,opts.geoGamma);
[LUVGMMdistFG,LUVGMMdistBG] = distanceFeature(LUVposteriors,Aext,opts,opts.geoGamma);
[EuclidianFG,EuclidianBG] = distanceFeature(RGBFeatures,Aext,opts,0);

% -------------- Scaling the data to [0,1] ---------------%

RGBposteriors = (RGBposteriors-min(min(RGBposteriors)))./(max(max(RGBposteriors))-min(min(RGBposteriors)));
LUVposteriors = (LUVposteriors-min(min(LUVposteriors)))./(max(max(LUVposteriors))-min(min(LUVposteriors)));
RGBdistFG = (RGBdistFG-min(min(RGBdistFG)))./(max(max(RGBdistFG))-min(min(RGBdistFG)));
RGBdistBG = (RGBdistBG-min(min(RGBdistBG)))./(max(max(RGBdistBG))-min(min(RGBdistBG)));
LUVdistFG = (LUVdistFG-min(min(LUVdistFG)))./(max(max(LUVdistFG))-min(min(LUVdistFG)));
LUVdistBG = (LUVdistBG-min(min(LUVdistBG)))./(max(max(LUVdistBG))-min(min(LUVdistBG)));
RGBGMMdistFG = (RGBGMMdistFG-min(min(RGBGMMdistFG)))./(max(max(RGBGMMdistFG))-min(min(RGBGMMdistFG)));
RGBGMMdistBG = (RGBGMMdistBG-min(min(RGBGMMdistBG)))./(max(max(RGBGMMdistBG))-min(min(RGBGMMdistBG)));
LUVGMMdistFG = (LUVGMMdistFG-min(min(LUVGMMdistFG)))./(max(max(LUVGMMdistFG))-min(min(LUVGMMdistFG)));
LUVGMMdistBG = (LUVGMMdistBG-min(min(LUVGMMdistBG)))./(max(max(LUVGMMdistBG))-min(min(LUVGMMdistBG)));
EuclidianFG = (EuclidianFG-min(min(EuclidianFG)))./(max(max(EuclidianFG))-min(min(EuclidianFG)));
EuclidianBG = (EuclidianBG-min(min(EuclidianBG)))./(max(max(EuclidianBG))-min(min(EuclidianBG)));

% -------------- write the files ------------- %
filename = strcat('./Dataset/images_RGB/',imageList{i},'.png');
imwrite(RGBFeatures,filename,'png');
filename = strcat('./Dataset/images_LUV/',imageList{i},'.png');
imwrite(LUVFeaturesN,filename,'png');
filename = strcat('./Dataset/images_RGBGMM/',imageList{i},'.png');
imwrite(RGBposteriors,filename,'png');
filename = strcat('./Dataset/images_LUVGMM/',imageList{i},'.png');
imwrite(LUVposteriors,filename,'png');
filename = strcat('./Dataset/images_RGBdistFG/',imageList{i},'.png');
imwrite(RGBdistFG,filename,'png');
filename = strcat('./Dataset/images_RGBdistBG/',imageList{i},'.png');
imwrite(RGBdistBG,filename,'png');
filename = strcat('./Dataset/images_LUVdistFG/',imageList{i},'.png');
imwrite(LUVdistFG,filename,'png');
filename = strcat('./Dataset/images_LUVdistBG/',imageList{i},'.png');
imwrite(LUVdistBG,filename,'png');
filename = strcat('./Dataset/images_RGBGMMdistFG/',imageList{i},'.png');
imwrite(RGBGMMdistFG,filename,'png');
filename = strcat('./Dataset/images_RGBGMMdistBG/',imageList{i},'.png');
imwrite(RGBGMMdistBG,filename,'png');
filename = strcat('./Dataset/images_LUVGMMdistFG/',imageList{i},'.png');
imwrite(LUVGMMdistFG,filename,'png');
filename = strcat('./Dataset/images_LUVGMMdistBG/',imageList{i},'.png');
imwrite(LUVGMMdistBG,filename,'png');
filename = strcat('./Dataset/images_EuclidianFG/',imageList{i},'.png');
imwrite(EuclidianFG,filename,'png');
filename = strcat('./Dataset/images_EuclidianBG/',imageList{i},'.png');
imwrite(EuclidianBG,filename,'png');



end
% end