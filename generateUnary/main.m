function [X,Y]=main()
% COMPUTE UNARY FEATURES
% add Dataset at:
% images: original images;
% images_gt: groundtruth
% images_labels: user annotation

% --
% Implemented by 
% Jiaqian Yu @ 2016
% 
addpath(genpath('gmm'));
addpath(genpath('gsc'));
addpath(genpath('Dataset'));
addpath(genpath('colorspace'));

extendAllSeed;

% COMPUTE AND SAVE UNARY FEATURES
[RGBFeatures,LUVFeatures,Y,A] = generateUnaryFeatures();

% REFORMAT THE FEATURES
num = 33;
[X,Y] = generateData(num);
