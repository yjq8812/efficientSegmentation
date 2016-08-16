function mynormalization(l,power)
ids = dir('./Overfeat/feat/*.features');
newpath = strcat('./Overfeat/normalized',num2str(l),num2str(power),'/');
mkdir(newpath);
for i=1:length(ids)
    f = strcat(newpath,ids(i,1).name(1:6),'.feat');
    fid=fopen(f,'w');
    % find ground truth image
    feature_read = textread(strcat('Overfeat/feat/',ids(i,1).name));
    if power
        fd = sign(feature_read(2,1:4096)).*abs(feature_read(2,1:4096)).^2.5;
        fd = fd./norm(fd,l);
    else
        fd = feature_read(2,1:4096)./norm(feature_read(2,1:4096),l);
    end
    fprintf(fid,'%f\n',fd);
    fclose(fid);

end
end