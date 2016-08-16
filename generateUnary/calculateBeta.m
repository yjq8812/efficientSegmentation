function beta = calculateBeta(feature)
[h,w,D] = size(feature);

roffset = zeros(h,w,D);
rdiff = zeros(h,w);
roffset(2:end,:,:) = feature(1:end-1,:,:);
rdiff(2:end-1,:) = sum((roffset(2:end-1,:,:) - feature(2:end-1,:,:)).^2,3);

loffset = zeros(h,w,D);
ldiff = zeros(h,w);
loffset(1:end-1,:,:) = feature(2:end,:,:);
ldiff(2:end-1,:) = sum((loffset(2:end-1,:,:) - feature(2:end-1,:,:)).^2,3);

uoffset = zeros(h,w,D);
udiff = zeros(h,w);
uoffset(:,1:end-1,:) = feature(:,2:end,:);
udiff(:,2:end-1) = sum((uoffset(:,2:end-1,:) - feature(:,2:end-1,:)).^2,3);

doffset = zeros(h,w,D);
ddiff = zeros(h,w);
doffset(:,2:end,:) = feature(:,1:end-1,:);
ddiff(:,2:end-1) = sum((doffset(:,2:end-1,:) - feature(:,2:end-1,:)).^2,3);

average = sum(sum(rdiff+ldiff+udiff+ddiff))/((h-1)*(w-1)*4);

beta = 1/ (2*average);
end