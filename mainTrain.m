function w = mainTrain(order_num,gamma,C,mode)

mainInit;
switch mode
    case 'train'
        filename = strcat('trained_order',num2str(order_num),'_C',num2str(C),'_gamma',num2str(gamma),'.mat');
        if exist(filename,'file')
            fprintf(' W is computed for this splitting, C, and gamma, loading DATA...\n')
            load(filename,'w')
        else
            [w,~,~,~]=implement_SOSVM_Learning(Xtrain,Ytrain,C,ifuseDD,ifuseADMM,ADMMrho,ourloss);
            save(filename,'w');
        end
    case 'retrain'
        filename = strcat('retrained_order',num2str(order_num),'_Cbest',num2str(C),'_gamma',num2str(gamma),'.mat');
        if exist(filename,'file')
            fprintf(' W is computed for this splitting, Cbest, and gamma, loading DATA...\n')
            load(filename,'w')
        else
            [w,~,~,~]=implement_SOSVM_Learning(Xtrainval,Ytrainval,C,ifuseDD,ifuseADMM,ADMMrho,ourloss);
            save(filename,'w');
        end
        
end

end