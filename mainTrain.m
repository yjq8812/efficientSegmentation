function w = mainTrain(X,Y,order_num,gamma,C,mode,ifuseDD,ifuseADMM,ADMMrho,ourloss,k)

% mainInit;
switch mode
    case 'train'
        filename = strcat('trained_order',num2str(order_num),'_C',num2str(C),'_K',num2str(k),'_gamma',num2str(gamma),'.mat');
        if exist(filename,'file')
            fprintf(' W is computed for this splitting, C, and gamma, loading DATA...\n')
            load(filename,'w')
        else
            [w,~,~,~]=implement_SOSVM_Learning(X,Y,C,ifuseDD,ifuseADMM,ADMMrho,ourloss);
            save(filename,'w');
        end
    case 'retrain'
        filename = strcat('retrained_order',num2str(order_num),'_Cbest',num2str(C),'_gamma',num2str(gamma),'.mat');
        if exist(filename,'file')
            fprintf(' W is computed for this splitting, Cbest, and gamma, loading DATA...\n')
            load(filename,'w')
        else
            [w,~,~,~]=implement_SOSVM_Learning(X,Y,C,ifuseDD,ifuseADMM,ADMMrho,ourloss);
            save(filename,'w');
        end
        
end

% end