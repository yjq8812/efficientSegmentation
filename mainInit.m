% initialization

addpath(genpath('sfo')); % http://www.mathworks.com/matlabcentral/fileexchange/20504-submodular-function-optimization
addpath(genpath('GCMex')); % http://vision.ucla.edu/~brian/gcmex.html
% addpath(genpath('Bk_matlab')); % http://vision.csd.uwo.ca/code/ Matlab Wrapper by Andrew Delong: BK_matlab.zip
% --------------------- Generate Data ---------------------------%
% Generate/Load your data here
% X : patterns, a cell in size of n*1; each cell in size of p*d
% Y : labels, a cell in size of n*1; each cell in size of p*1
% n : number of patterns
% p : size of bags
% d : dimension of feature vectors

load('Sampled690.mat','X','Y')
% load('OriginalData.mat','X','Y')
% load('orderFixed.mat','orderFixed')
% order = orderFixed(order_num,:);
order = randperm(length(X));
X = X(order);
Y = Y(order);
Xtrainval   = X(1:ceil(2*length(order)./3));
Ytrainval   = Y(1:ceil(2*length(order)./3));
Xtest  = X(ceil(2*length(order)./3)+1:end);
Ytest  = Y(ceil(2*length(order)./3)+1:end);

Xtrain = Xtrainval(1:ceil(length(Xtrainval)/2));
Ytrain = Ytrainval(1:ceil(length(Ytrainval)/2));
Xval = Xtrainval(ceil(length(Xtrainval)/2)+1:end);
Yval = Ytrainval(ceil(length(Xtrainval)/2)+1:end);

% ------- Input Loss function, with their related optimization ------------%
hamming = customLossFunction('Hamming'); % hamming loss
ourloss = customLossFunction('8connected',gamma); % 8-connected loss
% ourloss = customLossFunction('square');

ifuseADMM = 1;
ADMMrho = 0.1;
ifuseDD = 0;
    
