function [yhat,history] = violateMarginADMM(param, model, x, y,ourloss)
%%---------------- ADMM with Margin rescaling ----------------%%
%%--- function call back by implement_SOSVM_Learning.m -----%%
%%--- function then used in GeneralClassifier_SOSVM.m -----%%
% code refered from : ADMM functions by S.Boyd
% max violated y <w,phi(x,ybar)>+\Delta(y,ybar) 
% ADMM for finding saddling point of Lagrangian

assert(strcmp(param.formulationTypeSup,'margin'),'This model only works for margin rescaling!')

% set ADMM parameters
[m,n] = size(y);
y     = y(:)';
ind   = find(y~=0); 

% initialization
rho_init = param.stepsizeRho;% 
y_a_old  = -y;
y_b_old  = -y;
u        = zeros(length(y),1)';

iteration = 0;
MAX_ITER  = 500;
QUIET     = 1;
ABSTOL    = 1e-4; % from ADMM functions by S.Boyd
RELTOL    = 1e-2;

history.r_norm   = zeros(MAX_ITER,1);
history.s_norm   = zeros(MAX_ITER,1);
eps_pri  = zeros(MAX_ITER,1);
eps_dual = zeros(MAX_ITER,1);
history.objval   = zeros(MAX_ITER,1);

t_start = tic;

for k = 1:MAX_ITER
    
    iteration = iteration +1;
    % setting rho's strategy
    rho = rho_init;

    % y_a updates, inference part (supermodular)
    y_a = param.findMostViolatedMarginADMMInference(param,model,x,y,y_a_old,y_b_old,rho,u);  
    
    % y_b updates, lossFunciton part(supermodular)
    y_b = param.findMostViolatedMarginADMMLoss(param,ourloss,model,x,y,y_a,y_b_old,rho,u); 
    
    history.objval(k) = - dot(model.w,param.featureCB(param,x,reshape(y_a,m,n)))...
        - ourloss(reshape(y,m,n),reshape(y_b,m,n)) ;
    
    % u updates, sum up the running residuals
    u(ind) = u(ind) + y_a(ind) - y_b(ind);
    
    % calculate residuals
    history.r_norm(k) = norm((y_a(ind)-y_b(ind)));
    history.s_norm(k) = norm(rho*(y_b(ind)-y_b_old(ind)));
    
    % convergence checking
    eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(y_a(ind)), norm(y_b(ind)));
    eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if ~QUIET
        fprintf('admm: %3d  %10.4f  %10.4f  %10.4f  %10.4f  %10.2f\n', k, ...
            history.r_norm(k), eps_pri(k), ...
            history.s_norm(k), eps_dual(k), history.objval(k));
    end
    
    % stopping criterion
    if ( history.r_norm(k) < eps_pri(k) && history.s_norm(k) < eps_dual(k) )
        break;
    end
    
    y_b_old = y_b;
    y_a_old = y_a;
    
end
if ~QUIET
    toc(t_start);
end
% fprintf([' ADMM runs %u iteration.  \n'],iteration);
y_b(find(y==0)) = 0;
yhat = reshape(y_b,m,n);

end

