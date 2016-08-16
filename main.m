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
mainInit;

Cs = power(10,[-3:3]);
if length(Cs)>2
    err_train = zeros(length(Cs),1);
    for t = 1:length(Cs)
        fprintf(['\n **  First training with C =  %f...  **  \n'], Cs(t));
        
        %--------------------- SOSVM Parameters ----------------------%
        C = Cs(t);
        
        w_train = mainTrain(order_num,gamma,C,'train');
        
        [err_train(t),~] = testEvaluation(Xval,Yval,w_train,ourloss.function);
        
    end
    fprintf('\n **  Done ! **  \n');
    
    Cbest = Cs(find(err_train==min(err_train),1));
else
    Cbest = Cs;
end
% ---------------------- RE-TRAINING & TESTING-----------------------------%
fprintf('\n **  Retraining on trainval set...  **  \n');

w = mainTrain(order_num,gamma,Cbest,'retrain');


fprintf(['**  Testing...    \n']);
err = zeros(1,2);

[err(1,1),list] = testEvaluation(Xtest,Ytest,w, ourloss.function);
% [err(1,2),~] = testEvaluation(Xtest,Ytest,w, hamming.function);

fprintf(['** Done with  ! **\n']);

end
