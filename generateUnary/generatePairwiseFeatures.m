function generatePairwiseFeatures()
% pairewise feature
% exp(-c_k*beta*|I-i-I_j|^2)[y_i=y_j] in Eq.(19) in A.Osokin's paper
% beta = (2*average(|x_i-x_j|^2))^-1
% c_k = 0,0.1,0.3,1,3,10,  k = 6
featuredir = './Dataset/images_RGB/';
mkdir('./Dataset/PairwiseFeatures/');
files = dir(featuredir);
for img = 3:length(files) % for each image
    fprintf(['** ', num2str(img-2), '/', num2str(length(files)-2),' iamges...\n']);
    imageName = files(img).name(1:end-4);
    feature = double(imread(strcat(featuredir,imageName,'.png')))./255;% make sure [0,1]
    [H,W,D] = size(feature);
    
    beta = calculateBeta(feature);% calculate beta for each image
    Cs =[0,0.1,0.3,1,3,10];
    %     pairwise = sparse(H*W,H*W);
    
    for i =1:length(Cs)    
        c = Cs(i);
        fprintf(['calculating with c = ', num2str(c),' ...\n']);
        pairwise = sparse(H*W,H*W);
        for row = 0:H-1 % inspired by ~/GCMex/GCMex_test.m
            for col = 0:W-1% 4 connected pixel grid: to be changed to 8 connected
                pixel = 1+ row*W + col;
                if row+1 < H,
                    pairwise(pixel, 1+col+(row+1)*W) = P_ij(feature,pixel,1+col+(row+1)*W,beta,c);
                end
                if row-1 >= 0,
                    pairwise(pixel, 1+col+(row-1)*W) = P_ij(feature,pixel, 1+col+(row-1)*W,beta,c);
                end
                if col+1 < W,
                    pairwise(pixel, 1+(col+1)+row*W) = P_ij(feature,pixel, 1+(col+1)+row*W,beta,c);
                end
                if col-1 >= 0,
                    pairwise(pixel, 1+(col-1)+row*W) = P_ij(feature,pixel, 1+(col-1)+row*W,beta,c);
                end
            end
        end
%         pairwiseFeature = pairwiseFeature + pairwise;
%         clear pairwise
        filename = strcat('./Dataset/PairwiseFeatures/',imageName,'_',num2str(c),'.mat');
        save(filename,'pairwise','c');
        clear pairwise;
    end
    

end
end

function p_ij = P_ij(f,p1,p2,beta,c)

% exp(-c_k*beta*|I-i-I_j|^2) 
[H,W,D] = size(f);
[j1,i1] = ind2sub([W H],p1);% to check
[j2,i2] = ind2sub([W H],p2);

p_ij = exp(-c*beta*(sum((f(i1,j1,:)-f(i2,j2,:)).^2,3)));
end
