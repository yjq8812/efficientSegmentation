function [posteriors] = getPosteriorImage(Features,labelImg,segOptions)

% Copyright:
% Matlab implementation of the segmentation algorithms used in:
% Geodesic Star Convexity for Interactive Image Segmentation
% V. Gulshan, C. Rother, A. Criminisi, A. Blake and A. Zisserman

% Function to compute posteriors, by learning color models
% from the user strokes in labelImg


features_rgb = extractPixels(Features);
fgFeatures=features_rgb(:,labelImg(:)==1);% make sure fg label = 1
fgGmm=init_gmmBS(fgFeatures,segOptions.gmmNmix_fg);

bgFeatures=features_rgb(:,labelImg(:)==2);% make sure bg label = 2
bgGmm=init_gmmBS(bgFeatures,segOptions.gmmNmix_bg);

% --- Now compute posteriors ------
posteriors=compute_gmmPosteriors_mixtured(features_rgb,fgGmm,bgGmm,segOptions.gmmLikeli_gamma,segOptions.gmmUni_value);
posteriors=reshape(posteriors,size(labelImg));


end