function err = mainTest(order_num,Cs)
gamma = 0;
mainInit

fprintf(['**  Testing...    \n']);
err = zeros(1,2);

name = strcat('result/EmpiricalErrors_',num2str(order_num),'_Crange',num2str(min(Cs)),'_',num2str(max(Cs)),'_gamma',num2str(gamma),'.mat');
load(strcat(name),'w');
w_our = w; clear w;

name = strcat('hamming/EmpiricalErrors_',num2str(order_num),'_Crange',num2str(min(Cs)),'_',num2str(max(Cs)),'_gamma',num2str(gamma),'.mat');
load(strcat(name),'w');
w_ham = w; clear w;

[err(1,1),list] = testEvaluation(Xtest,Ytest,w_our, ourloss.function);
[err(1,2),~] = testEvaluation(Xtest,Ytest,w_our, hamming.function);


[err(2,1),list] = testEvaluation(Xtest,Ytest,w_ham, ourloss.function);
[err(2,2),~] = testEvaluation(Xtest,Ytest,w_ham, hamming.function);

end