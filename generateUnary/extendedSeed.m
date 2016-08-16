function [Yextended] = extendedSeed(GT,A)
% 1) For each image we have GT object segment, FRG seeds, and BKG seeds.
FRG =zeros(size(A));
FRG(find(A==1))=1;
BKG =zeros(size(A));
BKG(find(A==2))=2;

% 2) For each pixel of FRG and BKG we compute the minimum distance to the object boundary.
[minDistFG,FGindex] = minDist(FRG,GT);
[minDistBG,BGindex] = minDist(BKG,GT);

% 3) Each seed is enlarged by circle of radius equal to the 50% of the computed distance - new FRG and BKG.
FRGlarge = enlargePixel(FRG,minDistFG,FGindex,1);
BKGlarge = enlargePixel(BKG,minDistBG,BGindex,2);

% 4) Adjusting the FRG seeds: 
% a) FRG = intersect FRG with GT eroded by 10 pixels so seeds are not too close to the boundary; 
GT10 = imerode(GT,strel('disk',10));
FRGnew = FRGlarge.*GT10;
% b) FRG = union FRG with GT eroded by 20 pixels so seeds are not too far from the boundary;
GT20 = imerode(GT,strel('disk',20));
FRGnew = min(FRGnew+GT20,1);
% 5) Do operations analogous to 4 with the BKG.

GT10BG = imdilate(GT,strel('disk',10));
GT20BG = imdilate(GT,strel('disk',20));
GT10BG = (GT10BG-1).*-2;
GT20BG = (GT20BG-1).*-2;
BKGnew = min(BKGlarge.*GT10BG,2);
BKGnew = min(BKGnew+GT20BG,2);

Yextended = BKGnew + FRGnew;

Yextended = uint8(Yextended);

end

function [min_dist,fgindex] = minDist(fg,gt)
% fg : user-labeled seed (fg or bg)
% gt : groundtruth
% B  : object boundary from the groundturth
Btemp = bwboundaries(gt); %  exterior boundaries and interior boundaries
if length(Btemp)>1
B = [Btemp{1,1};Btemp{2,1}];
else
    B = Btemp{1,1};
end
fgindex=find(fg); % find non-zero, 1 or 2 both ok
% [r,c]=ind2sub([size(fg)],fgindex);
min_dist(1:length(fgindex),1) = Inf;
for i=1:length(fgindex)
    [r,c]=ind2sub([size(fg)],fgindex(i));
    for j=1:length(B)
        dist = sqrt((B(j,1)-r)^2+(B(j,2)-c)^2);
        if dist < min_dist(i)
            min_dist(i,1) = dist;
        end
    end   
end

end

function FRGnew = enlargePixel(FRG,minDistFG,FGindex,label)

FRGnew = zeros(size(FRG));
for i=1:length(minDistFG)
    radius = ceil(minDistFG(i)/2);
    ThisSeed = zeros(size(FRG));
    [r,c] = ind2sub([size(FRG)],FGindex(i));
    ThisSeed(r,c) = 1;
    se = strel('disk',radius,8);
    NewSeed = imdilate(ThisSeed,se);
    FRGnew = min(FRGnew + NewSeed,label);
end
end
