function [w, model,iteration,parm]=implement_SOSVM_Learning(Xtrain,Ytrain,C,ifuseDD,ifuseADMM,ADMMrho,setfn)
% last modification: 21-Apr-2015, function for ICCV2015 ADMM algo
if ~exist('setfn','var')
    setfn.type = 'modular';
    setfn.function = @lossFunction;
    setfn.falsevalue = [1 1];
end
 
% ------------------- data property  ----------------------------%
parm.patterns = Xtrain ;
parm.labels = Ytrain ;
parm.numclass=2; % fixed to 2 for now.
parm.dim=size(parm.patterns{1,1}.features,3);% feature dimension

% ------------------- inference model ----------------------------%
parm.featureCB = @featureFunction;
parm.sizePsi = parm.dim*2+3; % to be modified for different inference model
% phi = [phi_u,phi_p]
% phi_u : parm.dim * 2 for binary
% phi_p : parm.dim * kron([1;0],[1,0]) etc
% ------------------- loss function ------------------------------%
% arbitrary loss l = g + h;
parm.setFnname = setfn.name;
parm.setlossFn = {[];setfn.function}; 
parm.setlossFnType = setfn.type; 
parm.setlossFnSub = parm.setlossFn {1}; % g, submodular
parm.submodFnIsIncreasing = 1;
parm.setlossFnSupIn = parm.setlossFn {2}; % h, supermodular
if isfield(setfn,'falsevalue')
switch setfn.type
    case 'modular'
        parm.falsepositive = setfn.falsevalue(1);
        parm.falsenegative = setfn.falsevalue(2);
    case 'supermodular'
        parm.falsesingle = setfn.falsevalue(1);
        parm.falseall = setfn.falsevalue(2);
end
end

% ------------------- constraints --------------------------------%
type = {[];'margin'}; % for now ADMM only works with margin-rescaling
parm.formulationTypeSub = type{1}; % choose only for g
parm.formulationTypeSup = type{2}; % choose only for h, can be empty
parm.isgreedy = 1;% not used
assert(parm.isgreedy==1,'Non greedy algorithm are not working!!!!!')
parm.C = C;

% -------------------   method    --------------------------------%
parm.ifuseDD = ifuseDD;%
parm.ifuseADMM = ifuseADMM;%
parm.stepsizeRho = ADMMrho;
parm.findMostViolatedMarginGC = @lossAugmentedInference;% only use this constraint for now

parm.findMostViolatedMarginDD = @violateMarginDD; % dual decomposition only works with margin-rescaling for now 
parm.findMostViolatedMarginADMM = @violateMarginADMM; % dual decomposition only works with margin-rescaling for now 
parm.findMostViolatedMarginADMMInference = @violateInferenceADMM; % dual decomposition only works with margin-rescaling for now 
parm.findMostViolatedMarginADMMLoss = setfn.MaximizationMethod; % dual decomposition only works with margin-rescaling for now 

% Hard constraints to ensure submodularity of learned model
% relatived to the inference model
% w_p = [w(end-1) w(end);w(end) w(end-2)]
% hard constraints: w(end) <=0; w(end-1)>=0; w(end-2)>=0; 
% SEE bundler.m for the formulation of a_w and b_w 
a_w = zeros(parm.sizePsi,1); 
a_w(end) = -1;
b_w = 0;
parm.hardConstraints(1).a_w = a_w;
parm.hardConstraints(1).b_w = b_w;

a_w = zeros(parm.sizePsi,1); 
a_w(end-1) = 1;
b_w = 0;
parm.hardConstraints(2).a_w = a_w;
parm.hardConstraints(2).b_w = b_w;

a_w = zeros(parm.sizePsi,1); 
a_w(end-2) = 1;
b_w = 0;
parm.hardConstraints(3).a_w = a_w;
parm.hardConstraints(3).b_w = b_w;


% ---- followings are not used for now -----%
parm.findMostViolatedLovasz = @violateLovasz;
parm.findMostViolatedMargin = @violateMargin_greedy;
parm.findMostViolatedSlack = @violateSlack_greedy;

   
% ------------------------------------------------------------------
%                                                    Run SVM struct
% ------------------------------------------------------------------

[model,parm,state,iteration] = GeneralClassifier_SOSVM(parm);

w = model.w;


end

% --------------------------------------------------------------------
%                                                SVM struct callbacks
% --------------------------------------------------------------------

function psi = featureFunction(parm,x,y);
% compute unary features
% assume x is R^{mXnXd} for an mXn image where mXn = p pixels
% assume y is {-1,+1}^{mXn}
phi_yp = zeros(size(y));
phi_yp(find(y>0)) = 1;
phi_yn = zeros(size(y));
phi_yn(find(y<0)) = 1;

phi_u = [squeeze(sum(sum(x.features.*repmat(phi_yn,[1 1 size(x.features,3)]),1),2));...
    squeeze(sum(sum(x.features.*repmat(phi_yp,[1 1 size(x.features,3)]),1),2))];

% compute pairwise features, y^i in {-1,0,1}
dh = y(:,1:end-1)-y(:,2:end);
dv = y(1:end-1,:)-y(2:end,:);
dh(find(dh))=1;
dv(find(dv))=1;
phi_p00 = length(find(dh==0 & y(:,1:end-1)==-1)) + length(find(dv==0 & y(1:end-1,:)==-1));
phi_p11 = length(find(dh==0 & y(:,1:end-1)==1)) + length(find(dv==0 & y(1:end-1,:)==1));
phi_p01 = sum(sum(dh)) + sum(sum(dv)); 
phi_p = [phi_p11;phi_p00;phi_p01]; % NOTE: CONSISTENT WITH W_P

psi = [phi_u;phi_p];

end

function delta = lossFunction(y,ybar)

ind = find(y~=0);
delta = sum(sum(abs(y(ind)-ybar(ind))))./2;% y^i in {-1,1}

end

function result = lossAugmentedInferenceSFOHelper(ind,parm,lossFun,w,x,y,m,n)

% argmax_ybar <w,phi(x,ybar)> + delta(y,ybar)
y_temp=zeros(size(y));
y_temp(ind) = 1;
y_temp = y_temp*2-1;
y_temp = reshape(y_temp,[m n]);
y = reshape(y,[m n]);
result = -dot(w,featureFunction(parm,x,y_temp))-lossFun(y,y_temp) ;

end

% most violated constraint - margin rescaling for now
% argmax_ybar <w,phi(x,ybar)> + delta(y,ybar)
function ybar = lossAugmentedInference(parm,w,x,y)


assert(strcmp(parm.formulationTypeSup,'margin'),'This model only works for margin rescaling!')
m = size(x.features,1);
n = size(x.features,2);
y = y(:)';

w_u = w(1:end-3);
w_p = w(end-2:end);
if(w_p(3)>0)
    w_p(3) =0;
end
if(w_p(2)<0)
    w_p(2) =0;
end
if(w_p(1)<0)
    w_p(1) =0;
end
w_u = reshape(w_u,[length(w_u)/2 2]);
x.features = reshape(x.features,[size(x.features,1)*size(x.features,2) size(x.features,3)]);
UNARY = -(x.features*w_u)';
% loss augmented inference on unary potential
ind = find(y~=0);
UNARY(1,ind) = UNARY(1,ind) - (y(ind)+1)/2;% if ybar = -1 ; y^i in {-1,1}
UNARY(2,ind) = UNARY(2,ind) - (1-y(ind))/2;% if ybar = 1; y^i in {-1,1}  
CLASS = ones(1,m*n);% y^i in {-1,1}
LABELCOST = - [w_p(2) w_p(3);w_p(3) w_p(1)];
EXPANSION = 0;

% now create adjacency matrix for grid graph
PAIRWISE = x.PAIRWISE;
maxDegree = x.maxDegree;

%if(strcmp(pram.setFnname,'8connected')
%    LABELCOST =  LABELCOST - [0 0; 0 param.falseall];
%end

% now clamp the graph cuts to use the input squiggles
if isfield(x,'squiggle')
    ourInf = max(max(abs(LABELCOST)))*maxDegree + max(max(UNARY));
    x.squiggle = reshape(x.squiggle,[m*n 1]);
    UNARY(1,find(x.squiggle==1))=+ourInf;
    UNARY(1,find(x.squiggle==2))=-ourInf;
    UNARY(2,find(x.squiggle==2))=+ourInf;
    UNARY(2,find(x.squiggle==1))=-ourInf;
end
%assert(LABELCOST(1,1)+LABELCOST(2,2)-LABELCOST(1,2)-LABELCOST(2,1)<=0) %submodularity required

% University of Western Ontario Delong wrapper
h = BK_Create(size(PAIRWISE,1));
% set unary potentials
BK_SetUnary(h,UNARY);
% set pairwise potentials
[Edgesi,Edgesj] = find(PAIRWISE~=0);
ind = find(Edgesi<Edgesj);
Edgesi = Edgesi(ind);
Edgesj = Edgesj(ind);
Edges = [Edgesi,Edgesj,repmat(LABELCOST(:)',[length(Edgesi),1])];
% Each row is of the format [i,j,e00,e01,e10,e11]
if strcmp(parm.setFnname, '8connected')
    Edges(:,3) = Edges(:,3) - [y(Edgesi)==1]'.*[y(Edgesj)==1]'.*parm.falseall;
    Edges(:,4) = Edges(:,4) - [y(Edgesi)==1]'.*[y(Edgesj)~=1]'.*parm.falseall;
    Edges(:,5) = Edges(:,5) - [y(Edgesi)~=1]'.*[y(Edgesj)==1]'.*parm.falseall;
    Edges(:,6) = Edges(:,6) - [y(Edgesi)~=1]'.*[y(Edgesj)~=1]'.*parm.falseall;
end
BK_SetPairwise(h,Edges);
ENERGYAFTER = BK_Minimize(h);
LABELS = BK_GetLabeling(h)-1;
BK_Delete(h);


% UCLA Fulkerson wrapper can't handle different LABELCOST per edge
%[LABELS ENERGY ENERGYAFTER] = GCMex(CLASS, single(UNARY), PAIRWISE, single(LABELCOST),EXPANSION);

% if isfield(x,'squiggle')
%     if ~isempty(find(x.squiggle'==1))
%         assert(unique(LABELS(find(x.squiggle'==1)))==1)
%     end
%     if ~isempty(find(x.squiggle'==2))
%     assert(unique(LABELS(find(x.squiggle'==2)))==0)
%     end
% end
ybar = (double(LABELS)*2-1)';
ybar(find(y==0)) =0;

ybar = reshape(ybar,m,n);

return;

assert(strcmp(parm.formulationTypeSup,'margin'),'This model only works for margin rescaling!')
m = size(x.features,1);
n = size(x.features,2);
y=y(:)';
lossFun = parm.setlossFn{2};
F = @(A)(lossAugmentedInferenceSFOHelper(A,parm,lossFun,w,x,y,m,n));
% tic
[A,subopt] = sfo_min_norm_pointYu(F, [1:length(y)]);
% toc
ybar = zeros(size(y));
ybar(A) = 1;
ybar = ybar*2-1;
if isfield(x,'squiggle')
    x.squiggle = reshape(x.squiggle,[prod(size(y)) 1]);
    ybar(find(x.squiggle==2))=-1;
    ybar(find(x.squiggle==1))=+1;
end
ybar(y==0)=0;
ybar = reshape(ybar,[m n]);
return;

m = size(x.features,1);
n = size(x.features,2);

w_u = w(1:end-3);
w_p = w(end-2:end);
oureps = 1e-4;
if norm(w_p,1)~=0
    scaling = norm(w_p,1);
else
    scaling = 1;
end
% assert(w_p(3)*scaling<=oureps);
% assert(w_p(2)*scaling>=-oureps);
% assert(w_p(1)*scaling>=-oureps);% -6.2463e-11
if(w_p(3)>0)
    w_p(3) =0;
end
if(w_p(2)<0)
    w_p(2) =0;
end
if(w_p(1)<0)
    w_p(1) =0;
end
y = y(:)';
w_u = reshape(w_u,[length(w_u)/2 2]);
x.features = reshape(x.features,[size(x.features,1)*size(x.features,2) size(x.features,3)]);
UNARY = -(x.features*w_u)';

% NOW ADD LOSS AUGMENTATION

% false positive penalty = 0.5 : y =-1, ybar = 1
% false negative penalty = 1.5 : y = 1, ybar =-1
ind = find(y~=0);
switch parm.setlossFnType
    case 'modular' %hamming
        UNARY(1,ind) = UNARY(1,ind) - parm.falsenegative*(y(ind)+1)/2;% if ybar = -1 ; y^i in {-1,1}
        UNARY(2,ind) = UNARY(2,ind) - parm.falsepositive*(1-y(ind))/2;% if ybar = 1; y^i in {-1,1}  
        LABELCOST = [-w_p(2) -w_p(3);-w_p(3) -w_p(1)];
    case 'supermodular'
        UNARY(1,ind) = UNARY(1,ind) - (y(ind)+1)/2;% if ybar = -1 ; y^i in {-1,1}
        UNARY(2,ind) = UNARY(2,ind) - (1-y(ind))/2;% if ybar = 1; y^i in {-1,1}
        
        LABELCOST = [-w_p(2) -w_p(3);-w_p(3) -w_p(1)] - [0 parm.falsesingle; parm.falsesingle parm.falseall];
        error('the code here is wrong!')
end

CLASS = ones(1,m*n);% y^i in {-1,1} but not sure if works for labels<0;
EXPANSION = 0;

% now create adjacency matrix for grid graph
PAIRWISE = x.PAIRWISE;
maxDegree = x.maxDegree;
% now clamp the graph cuts to use the input squiggles
if isfield(x,'squiggle')
    ourInf = max(max(abs(LABELCOST)))*maxDegree + 1;
    x.squiggle = reshape(x.squiggle,[m*n 1]);
    UNARY(1,find(x.squiggle==2))=-ourInf;
    UNARY(2,find(x.squiggle==2))=+ourInf;
    UNARY(2,find(x.squiggle==1))=-ourInf;
    UNARY(1,find(x.squiggle==1))=+ourInf;
end

assert(LABELCOST(1,1)+LABELCOST(2,2)-LABELCOST(1,2)-LABELCOST(2,1)<=0) %submodularity required

[LABELS ENERGY ENERGYAFTER] = GCMex(CLASS, single(UNARY), PAIRWISE, single(LABELCOST),EXPANSION);
if isfield(x,'squiggle')
    if ~isempty(find(x.squiggle'==1))
        assert(unique(LABELS(find(x.squiggle'==1)))==1)
    end
    assert(unique(LABELS(find(x.squiggle'==2)))==0)
end

ybar = reshape(LABELS*2-1,[m n]);

end

%-------------------- Lovas Hinge ----------------------------%

function [gamma,deltaPsi] = violateLovasz(param, model, x, y,setfn)
% max w.r.t. permutations of sum_k s_{\pi_k}^i(f(\{\pi_1,...,\pi_k\}) - f(\{\pi_1,...,\pi_{k-1}\})
% where f is submodular loss function and s_k^i is the margin violation of the kth sample in bag i
% see notes 09/05/2014(i)

w = model.w;

s = 1-(x*w).*y; % margin violations

[~,ind] = sort(s,'descend');
gammak = zeros(length(y),1);

for k=1:length(ind)
    gammak(k) = setfn(ind(1:k),length(y))-setfn(ind(1:k-1),length(y));
end

if(param.submodFnIsIncreasing)
    y(s<=0)=0;
end
gamma = sum(gammak.*double(y~=0));
deltaPsi = ((gammak.*y(ind))'*x(ind,:))';


end

%------------ Greedy Slack rescaling and Margin rescaling -----------------%

function yhat = violateSlack_greedy(param, model, x, y,setfn) % no longer used
% Greedy algorithm selection
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)

w = model.w;

constraint_max = -inf;% initialisation
yhat=y;% initialisation

for i=1:length(y)
    temp_y = yhat;
    temp_y(i) = temp_y(i)*(-1);
    [Psi_u,Psi_p] = param.featureFunction(param,x,temp_y);
    Psi = [Psi_u;Psi_p];
    constraint_new = setfn(temp_y,y)*(1+dot(w,Psi));
    
    if constraint_new>=constraint_max
        constraint_max = constraint_new;
        yhat = temp_y;
    end
end

end

function yhat = violateMargin_greedy(param, model, x, y,setfn) % no longer used
% Greedy algorithm selection
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
w = model.w;

% max violated y
constraint_max = -inf;% initialisation
yhat = y;% initialisation
for i=1:length(y)
    temp_y = yhat;
    temp_y(i) = temp_y(i)*(-1);
    [Psi_u,Psi_p] = param.featureFunction(param,x,temp_y);
    Psi = [Psi_u;Psi_p];
    constraint_new = setfn(temp_y,y) + dot(w,Psi);
    
    if constraint_new>=constraint_max
        constraint_max = constraint_new;
        yhat = temp_y;
    end
end
end
