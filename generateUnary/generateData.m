function [X,Y] = generateData(num)
% UnaryFeature:

file = dir('./Dataset/images_people');
gtpath = './Dataset/images_gt';

labelpath = './Dataset/images_labels';

imageList = cell(length(file)-2,1);
for i = 3:length(file)
    imageList{i-2} = file(i).name(1:end-4); % set up the names of images 
                                            % only accroding to FILE
                                            % direction
end

if num>length(imageList)
    warning('The images required are more than the total number! All images are going to be used.');
    num = length(imageList);
end
X = cell(num,1);
Y = cell(num,1);
for i=1:num
    dim = 0;
    for j = 1:14
        filepath = unaryFeatureDir(j);
        feature = double(imread(strcat(filepath,'/',imageList{i},'.png')))./255;
        [m,n,D] = size(feature);
        
        for d = 1:D
            dim = dim +1;
            X{i,1}.features(:,:,dim) = feature(:,:,d);
        end
    end
    Y{i,1}(:,:) = double(imread(strcat(gtpath,'/',imageList{i},'.png')))./255;
    Y{i,1} = (Y{i,1}-0.5)*2; % [0,1] to [-1 1]
    X{i,1}.squiggle = imread(strcat(labelpath,'/',imageList{i},'-anno.png'));
    
    B = [kron(speye(n),makeDmatrix(m));kron(makeDmatrix(n),speye(m))];
    PAIRWISE = B'*B;% unique = [-1 1 2];
    PAIRWISE = PAIRWISE - diag(diag(PAIRWISE));
    PAIRWISE(find(PAIRWISE))=1;% unique = 1;
    maxDegree = full(max(sum(PAIRWISE)));
    X{i,1}.PAIRWISE = PAIRWISE;
    X{i,1}.maxDegree = maxDegree;
    clear B PAIRWISE;
    
    if length(unique(Y{i,1}))~=2
        ind = find((Y{i,1}~=-1) & (Y{i,1}~=1));
        Y{i,1}(ind) = 0; % manually set the uncertain label to 0 from some weird greyvalue
    end
end
fprintf([num2str(num),' images are generated. \n']);
end

function Dm = makeDmatrix(m)
Dm = spalloc(m,m,0);
Dm(2:end,2:end) = speye(m-1,m-1);
Dm(2:end,1:end-1) = Dm(2:end,1:end-1) - speye(m-1,m-1);
end