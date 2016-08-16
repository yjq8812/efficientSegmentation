
fgindex=find(fg);
[r,c]=ind2sub([size(fg)],fgindex);
min_dist(1:length(r),1) = Inf;
for i=1:length(r)
    for j=1:length(B)
        dist = sqrt((B(j,1)-r(i))^2+(B(j,1)-c(i))^2);
        if dist < min_dist(i)
            min_dist(i,1) = dist;
        end
    end   
end

