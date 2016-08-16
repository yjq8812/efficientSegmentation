function [dFG,dBG] = distanceFeature(img,labelImg,opts,geoGamma)
% Compute distance features defined in A.Osokin & P.Kholi paper
% Code from : gsc-1.2/packages/+gsc/preProcess.m?, getStarEdges.m   
% function [Wstar,starInfo]=getStarEdges(labelImg,nbrHood,geoImg,geoGamma)
[h w]=size(labelImg);
N=h*w;

switch(opts.starNbrhood_size)
  case 4
    roffset = [ 1,  0 ]; % 4 nbrhood
    coffset = [  0, 1  ];
  case 8
    roffset = [ 1, 1, 0, -1 ];
    coffset = [  0, 1, 1, 1 ];
end
[lEdges,rEdges,colorWeights,spWeights]=...
    mex_setupTransductionGraph(img,int32(roffset'),int32(coffset'));

avgEucEdge_sqr=sum(roffset.*roffset+coffset.*coffset)/length(roffset);
avgGeoEdge_sqr=mean(colorWeights);
rescale_geo=avgEucEdge_sqr/avgGeoEdge_sqr;



% label number: 1 - fg brush
%               2 - bg brush
%               0 - unlabeled

% --------- foreground ------------------ %
stPointsFG=find(labelImg==1);
numPts=length(stPointsFG);
pts=zeros(3,length(stPointsFG));

if(~isempty(stPointsFG))
    [pts(1,:),pts(2,:),pts(3,:)]=ind2sub([h w],stPointsFG);
else
    warning('No foreground seeds, so no star shape energies will be computed\n');
    dFG = [];
end

if(numPts>0)
    pts=pts(1:2,:);    
    [dFG,rootPoints,qFG]=shortestPaths_normalized(img,pts,geoGamma,...
        opts.starNbrhood_size,rescale_geo);    
end

% --------- background ------------------ %
stPointsBG=find(labelImg==2);
numPts=length(stPointsBG);
pts=zeros(3,length(stPointsBG));

if(~isempty(stPointsBG))
    [pts(1,:),pts(2,:),pts(3,:)]=ind2sub([h w],stPointsBG);
else
    warning('No background seeds, so no star shape energies will be computed\n');
    dBG = [];
end

if(numPts>0)
    pts=pts(1:2,:);    
    [dBG,rootPoints,qBG]=shortestPaths_normalized(img,pts,geoGamma,...
        opts.starNbrhood_size,rescale_geo);    
end


end