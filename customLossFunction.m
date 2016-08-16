function loss = customLossFunction(name,g)
% User should defined your loss function here, following the format: 
% loss.type: the submodularity of the loss function: submodular, modular, supermodular
% loss.function: 
%      input two vector£º Y - the groundtruth, YBAR - the output prediction
%      output a real value: the loss/error value
% loss.name: defined a name for your loss in order to call it in other
% funcitons
% loss.MaximazationMethod: defined your maximazation strategy for this loss
% involved with the quadratic term from ADMM
% '8connected' is the loss in YU&Blaschko,BMVC2016
 
switch name
    case 'square'
        loss.type = 'supermodular';
%         loss.falsevalue = [0 g];
        loss.function = @(y,ybar)((length(find(y~=ybar))/20)^2);
        loss.name = name;
        loss.MaximizationMethod = @violateLossADMM;
    case '8connected'
        loss.type = 'supermodular';
        loss.falsevalue = [0 g];
        loss.function = @(y,ybar)(eightConnectedLoss(y,ybar,loss.falsevalue));
        loss.name = name;
        loss.MaximizationMethod = @violateLossADMM;
    case 'labelcount'
        loss.type = 'supermodular';
%         loss.falsevalue = [0 g];
        loss.function = @(y,ybar)(labelcount(y,ybar));
        loss.name = name;
        loss.MaximizationMethod = @violateLossADMM;
    case 'Hamming'
        loss.type = 'modular';
        loss.falsevalue = [1 1];        
        loss.function = @(y,ybar)(ModularLoss(y,ybar,loss.falsevalue));
        loss.name = name;
        loss.MaximizationMethod = @violateLossADMM;
    case 'HammingPotts'  
        loss.type = 'supermodular';
        loss.falsevalue = [0.1 1];        
        loss.function = @(y,ybar)(HammingPotts(y,ybar,loss.falsevalue));
        loss.name = name;
        loss.MaximizationMethod = @violateLossADMM;
    case 'TestSupermodular'  
        loss.type = 'supermodular';     
        loss.function = @TestSupermodular;
        loss.name = name;
        loss.MaximizationMethod = @violateLossADMM;
    otherwise
        error('The name of function does not exist!');
end
end

function delta = labelcount(y,ybar)

delta = abs(length(find(y==1))-length(find(ybar==1)));

end

function delta = eightConnectedLoss(y,ybar,fv)

ybar(y==0)=0;

y = double(y~=ybar);

    delta = sum(sum(y)) + (...
            sum(sum(y(1:end-1,:).*y(2:end,:))) + ...
            sum(sum(y(:,1:end-1).*y(:,2:end))) + ...
            sum(sum(y(1:end-1,1:end-1).*y(2:end,2:end))) + ...
            sum(sum(y(1:end-1,2:end).*y(2:end,1:end-1))))*fv(2);


end

function delta = ModularLoss(y,ybar,fv)
% y: groundtruths
% ybar: prediction outcomes
% falsepositive: y = -1, ybar = +1;
% falsenegative: y = +1, ybar = -1;
fp = fv(1);
fn = fv(2);

ind_fp = find(y==-1);
delta_fp = sum(sum(((y(ind_fp) -ybar(ind_fp))./2)*fp));
ind_fn = find(y==1);
delta_fn = sum(sum(((y(ind_fn) -ybar(ind_fn))./2)*fn));
delta = abs(delta_fp)+abs(delta_fn);
end

function delta = TestSupermodular(y,ybar)
if ~exist('ybar','var')
delta = length(find(y))^2./length(y);
else
delta = length(find(y~=ybar))^2./length(y);
end
end

function delta = HammingPotts(y,ybar,fv)
% y: groundtruths
% ybar: prediction outcomes

fsingle = fv(1);
fall = fv(2);

diff = abs(y-ybar)./2;

hamming = sum(sum(diff));

% Potts model 
diffshift = diff;
diffshift(:,1:end-1) = diff(:,2:end);
diff01 = length(find(diff-diffshift==-1));

diffshift = diff;
diffshift(:,1:end-1) = diff(:,2:end);
diff01 = diff01 + length(find(diff(:,1:end-1)-diffshift(:,1:end-1)==1));

diffshift = diff;
diffshift(1:end-1,:) = diff(2:end,:);
diff01 = diff01 + length(find(diff-diffshift==-1));

diffshift = diff;
diffshift(1:end-1,:) = diff(2:end,:);
diff01 = diff01 + length(find(diff(1:end-1,:)-diffshift(1:end-1,:)==1));


diffshift = diff;
diffshift(:,1:end-1) = diff(:,2:end);
diff11 = length(find(diff(:,1:end-1)-diffshift(:,1:end-1)==0 & diff(:,1:end-1)~=0));

diffshift = diff;
diffshift(1:end-1,:) = diff(2:end,:);
diff11 = diff11 + length(find(diff(1:end-1,:)-diffshift(1:end-1,:)==0 & diff(1:end-1,:)~=0));

potts = fsingle * diff01 + fall * diff11;

delta = hamming + potts;


end
