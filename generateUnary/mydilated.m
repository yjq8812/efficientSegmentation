function [fg1]= mydilated(fg,GroundTruth)
fgindex=find(fg);
[r,c]=ind2sub([size(fg)],fgindex);
[h,w]=size(GroundTruth);
for i=1:length(r)
    if ~isempty(find(double(GroundTruth(:,c(i))),1,'first'))
        distup(i)=abs(r(i)-find(double(GroundTruth(:,c(i))),1,'first'));
    else
        distup(i) = h;
    end;
end
for i=1:length(r)
    if ~isempty(find(double(GroundTruth(:,c(i))),1,'last'))
        distdown(i)=abs(r(i)-find(double(GroundTruth(:,c(i))),1,'last'));
    else
        distdown(i)=h;
    end;
end;

for i=1:length(r)
    if ~isempty(find(double(GroundTruth(r(i),:)),1,'first'))
        distleft(i)=abs(c(i)-find(double(GroundTruth(r(i),:)),1,'first'));
    else
        distleft(i)=w;
    end;
end;

for i=1:length(r)
    if ~isempty(find(double(GroundTruth(r(i),:)),1,'last'))
        distright(i)=abs(c(i)-find(double(GroundTruth(r(i),:)),1,'last'));
    else
        distright(i)=w;
    end;
end

se = strel('line',max(distup)/2,90);
fgup=imdilate(fg,se);
se = strel('line',max(distdown)/2,-90);
fgdown=imdilate(fg,se);
se = strel('line',max(distleft)/2,180);
fgleft=imdilate(fg,se);
se = strel('line',max(distright)/2,0);
fgright=imdilate(fg,se);

fg1= min(fgup+fgdown+fgleft+fgright,1);

% rayon = ceil(max([distup distdown distleft distright])/2/2);
% se = strel('disk',rayon,8);
% fg2=imdilate(fg,se);

end