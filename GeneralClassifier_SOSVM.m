function [model,sparm,state,iteration] = GeneralClassifier_SOSVM(sparm, oldstate)

global numIterations
state = bundler(); % initialize state

state.lambda = 1 ./ (sparm.C);

if (~isfield(sparm,'convergenceThreshold'))
    sparm.convergenceThreshold = 0.008;
end

maxIterations = 1000;

sparm.w = zeros(sparm.sizePsi,1);
state.w = sparm.w;

% add constraints on w e.g. <w, phi> being supermodular
for i=1:length(sparm.hardConstraints)
    state = bundler(state,sparm.hardConstraints(i).a_w,sparm.hardConstraints(i).b_w,0);% 
end
    
model.w = state.w;

if (exist('oldstate','var'))
    for i=1:length(oldstate.b)
        if(oldstate.softVariables(i))
            state = bundler(state,oldstate.a(:,i),oldstate.b(i));
        end
    end
end

minIterations = 1;
numIterations = 0;

bestPrimalObjective = Inf;

iteration.iter = 0;
iteration.gap = [];
while (((bestPrimalObjective - state.dualObjective)/state.dualObjective > sparm.convergenceThreshold ...
        || minIterations>0) && numIterations < maxIterations )

    numIterations = numIterations + 1;
    minIterations = minIterations - 1;
    
    %------------------------------ Decompostion ------------------------------%
    if size(sparm.formulationTypeSub)~=0
        switch sparm.formulationTypeSub
            case 'lovasz'
                [phi_g, b_g] = computeOneslackLovasz(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSub);
            case 'margin'
                [phi_g, b_g] = computeOneslackMargin(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSub,0);
            case 'slack'
                [phi_g, b_g] = computeOneslackSlack(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSub,0);
            otherwise
                error('The type has not been well defined for the submodular loss!')
        end
    else
        phi_g = 0;
        b_g = 0;
    end
    
    if size(sparm.formulationTypeSup)~=0
        switch sparm.formulationTypeSup
            case 'lovasz'
                error('Lovasz hinge cannot work with supermodular increasing set function!')
            case 'margin'
                [phi_h, b_h] = computeOneslackMargin(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSupIn,1);
            case 'slack'
                [phi_h, b_h] = computeOneslackSlack(sparm, model, sparm.patterns, sparm.labels,sparm.setlossFnSupIn,1);
            otherwise
                error('The type has not been well defined for the supermodular loss!')
        end
    else
        phi_h = 0 ;
        b_h = 0 ;
    end
    phi = phi_g +phi_h;
    b = b_g + b_h;
    %------------------------------ End of decomp. ------------------------------%

    if (norm(phi)==0)
        phi= zeros(size(state.w));
        warning('None contraints added!!!\n\n');
    end
    primalobjective = (state.lambda / 2) * (state.w' * state.w) + b - dot(state.w, phi);
    if (primalobjective < bestPrimalObjective)
        bestPrimalObjective = primalobjective;
        bestState = state;
    end
    
    gap = (bestPrimalObjective - state.dualObjective) / state.dualObjective;
   
    fprintf([' %d primal objective: %f, best primal: %f, dual objective: %f, gap: %f  ' datestr(now) '\n'], ...
        numIterations, primalobjective, bestPrimalObjective, state.dualObjective, gap);
    
    state = bundler(state, phi, b);
    
    sparm.w = state.w;
    model.w = state.w;
    
    iteration.iter = numIterations;
    iteration.gap = [iteration.gap gap];
    
    if norm(model.w)==0
        warning('Learned Weight Vector is Empty!!!\n\n');
    end
end

sparm.w = bestState.w;
model.w = bestState.w;

end

function [phi, b] = computeOneslackLovasz(sparm,model,X,Y,setfn)
phi = 0;
b = 0;

% For each pattern
for i = 1 : length(X);
    [gamma,deltaPsi] = sparm.findMostViolatedLovasz(sparm, model, X{i}, Y{i},setfn);
    if (dot(model.w,deltaPsi) < gamma)
        b = b + gamma;
        phi = phi + deltaPsi;
    end
end
end


function [phi, b] = computeOneslackMargin(sparm,model,X,Y,setfn,issuper)
global numIterations
phi = 0;
b = 0;

compare = 0;
if numIterations>1 && mod(numIterations,10)==1
    compare = 0;
end
% For each pattern
for i = 1 : length(X);
    if ~issuper
        [tildeY] = sparm.findMostViolatedMargin(sparm, model, X{i}, Y{i},setfn);
    else
        if sparm.ifuseDD==1
            [tildeY] = sparm.findMostViolatedMarginDD(sparm, model, X{i}, Y{i},setfn);
        elseif sparm.ifuseADMM==1
%             tic
            [tildeY,~] = sparm.findMostViolatedMarginADMM(sparm, model, X{i}, Y{i},setfn);
%             admmtime(i) = toc;
            if compare
                tic
                [tildeYlp] = LPrelaxation(sparm, model, X{i}, Y{i},setfn);
                lptime(i) = toc;  
%                 [tildeYgc] = sparm.findMostViolatedMarginGC(sparm, model.w, X{i}, Y{i});% loss augmented inference
%                 ind = find(tildeY~=0);
%                 diffADMMLP(i) = length(find(tildeY(ind)~=tildeYlp(ind)))/prod(size(tildeY));
%                 diffADMMGC(i) = length(find(tildeY(ind)~=tildeYgc(ind)))/prod(size(tildeY));
%                 [energyADMM(i),energyLP(i)] = compareEnergyLP(sparm,X{i}, Y{i},tildeY,tildeYlp,model.w,setfn);
%                 [energyADMM(i),energyGC(i)] = compareEnergyLP(sparm,X{i}, Y{i},tildeY,tildeYgc,model.w,setfn);
            end
        else
            [tildeY] = sparm.findMostViolatedMarginGC(sparm, model.w, X{i}, Y{i});% loss augmented inference
        end;
    end
    
    delta = setfn(Y{i},tildeY);%(y,ybar)
    
    deltaPsi =  sparm.featureCB(sparm, X{i}, Y{i}) - sparm.featureCB(sparm,X{i},tildeY);
    
    if (delta - dot(model.w,deltaPsi) > 0)
        b = b + delta;
        phi = phi + deltaPsi;
    end
%     fprintf([' %u / %u\n'],i,length(X));
end
if compare
%     save(strcat('compareADMM_LP_GC_',sparm.setFnname, datestr(now), '.mat'),'diffADMMLP','diffADMMGC','energyADMM','energyLP','energyGC')
    save(strcat('compareADMM_LP_Time270_',sparm.setFnname, datestr(now), '.mat'),'admmtime','lptime')
end
end


function [phi, b] = computeOneslackSlack(sparm,model,X,Y,setfn,issuper)
phi = 0;
b = 0;
assert(false,'Slack-rescaling codes are not completed!!'); 

% For each pattern
for i = 1 : length(X);
    if ~issuper
        [tildeY] = sparm.findMostViolatedSlack(sparm, model, X{i}, Y{i},setfn);
    else
        if sparm.ifuseDD==1
            error('For now ADMM does not work with Slack-rescaling');
        else
            assert(false,'Graph Cut with slack-rescaling are not completed!!'); 
            [tildeY] = sparm.findMostViolatedSupIn(sparm, model, X{i}, Y{i},setfn); %[A,subopt] = sfo_min_norm_point(F,V, opt)
        end
    end
    delta = setfn(Y{i}~=tildeY,length(Y{i}));
    deltaPsi = sparm.psiFn(sparm, X{i}, Y{i}) - sparm.psiFn(sparm, X{i}, tildeY);
    if (dot(model.w,deltaPsi) < 1) % delta(1-dot(model.w,deltaPsi))>0
        b = b + delta;
        phi = phi + deltaPsi.*delta;
    end
end

phi = phi';
end