function [err,w] = main(order_num,gamma)
% An implementation for an efficient segmentation with user-defined loss
% function and user-defined optimisation for the loss maximization and
% inference procedure
% --
% Implemented by 
% Jiaqian Yu @ 2016
% 
% For more details: 
% Yu, J. and M. B. Blaschko: Efficient Learning for Discriminative 
% Segmentation with Supermodular Losses. British Machine Vision Conference 
% (BMVC), 2016.

if ~exist('order_num','var')
    order_num = 1;
end
if ~exist('gamma','var')
    gamma = 1;
end

% --------------- Problem Setting ---------------------%
% ---- Prepare trainning/validation/testing data ------%
mainInit;

Cs = power(10,[-3:3]);
if length(Cs)>2
    err_train = zeros(length(Cs),1);
    
    for k = 1: crossValtime
    fprintf(['\n **  Cross-validation K =  %d...  **  \n'], k);
    for iC = 1:length(Cs)
        fprintf(['\n **  First training with C =  %f...  **  \n'], Cs(iC));
        
        %--------------------- SOSVM Parameters ----------------------%
        C = Cs(iC);
        
        w_train = mainTrain(Xtrain{k},Ytrain{k},order_num,gamma,C,'train',ifuseDD,ifuseADMM,ADMMrho,ourloss,k);
        
        [errCross,~] = testEvaluation(Xval{k},Yval{k},w_train,ourloss.function);
        
        err_train(iC) = err_train(iC) + errCross;
        
    end
    fprintf('\n **  Done ! **  \n');
    end
    Cbest = Cs(find(err_train==min(err_train),1));
else
    Cbest = Cs;
end
% ---------------------- RE-TRAINING & TESTING-----------------------------%
fprintf('\n **  Retraining on trainval set...  **  \n');

w = mainTrain(Xtrainval,Ytrainval,order_num,gamma,Cbest,'retrain',ifuseDD,ifuseADMM,ADMMrho,ourloss);


fprintf(['**  Testing...    \n']);
err = zeros(1,2);

[err(1,1),list] = testEvaluation(Xtest,Ytest,w, ourloss.function);
% [err(1,2),~] = testEvaluation(Xtest,Ytest,w, hamming.function);

fprintf(['** Done with  ! **\n']);

end
