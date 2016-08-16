function ybar = violateInferenceADMM(param,model,x,y,y_a_old,y_b,rho,u)
% Function Callback by VIOLATEMARGINADMM.m
% problem: argmin_y -<w,phi(x,y)> + rho/2*|y-y_b+u|_2^2
% via graph cuts

m = size(x.features,1);
n = size(x.features,2);

w = model.w;
w_u = w(1:end-3);
w_p = w(end-2:end);
oureps = 1e-5;
if norm(w_p,1)~=0
    scaling = norm(w_p,1);
else
    scaling = 1;
end
assert(w_p(3)*scaling<=oureps);
assert(w_p(2)*scaling>=-oureps);
assert(w_p(1)*scaling>=-oureps);% -6.2463e-11
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

ind = find(y~=0);
% trait y as in [0 1] or [-1 1]??
UNARY(1,ind) = UNARY(1,ind) + rho/2* (-1-y_b(ind)+u(ind)).^2;% if ybar = -1 
UNARY(2,ind) = UNARY(2,ind) + rho/2* ( 1-y_b(ind)+u(ind)).^2;% if ybar = 1

% % bug fixed 21-01-2015 adding norm, not correct
% UNARY(1,ind) = UNARY(1,ind) + rho/2* norm(-1-y_b(ind)+u(ind),2).^2;% if ybar = -1 
% UNARY(2,ind) = UNARY(2,ind) + rho/2* norm( 1-y_b(ind)+u(ind),2).^2;% if ybar = 1

CLASS = ones(1,m*n);% y^i in {-1,1}
LABELCOST = [-w_p(2) -w_p(3);-w_p(3) -w_p(1)];


%if(strcmp(param.setFnname,'8connected')
%    LABELCOST =  LABELCOST - [0 0; 0 param.falseall];
%end


EXPANSION = 0;

% now create adjacency matrix for grid graph
PAIRWISE = x.PAIRWISE;
maxDegree = x.maxDegree;

% now clamp the graph cuts to use the input squiggles
if isfield(x,'squiggle')
    ourInf = max(max(abs(LABELCOST)))*maxDegree + max(max(UNARY))+1;
    x.squiggle = reshape(x.squiggle,[m*n 1]);
    UNARY(1,find(x.squiggle==2))=-ourInf;
    UNARY(2,find(x.squiggle==2))=+ourInf;
    UNARY(2,find(x.squiggle==1))=-ourInf;
    UNARY(1,find(x.squiggle==1))=+ourInf;
end
assert(LABELCOST(1,1)+LABELCOST(2,2)-LABELCOST(1,2)-LABELCOST(2,1)<=0) %submodularity required

[LABELS ENERGY ENERGYAFTER] = GCMex(CLASS, single(UNARY), PAIRWISE, single(LABELCOST),EXPANSION);
% if isfield(x,'squiggle')
%     if ~isempty(find(x.squiggle'==1))
%         assert(unique(LABELS(find(x.squiggle'==1)))==1)
%     end
%     if ~isempty(find(x.squiggle'==2))
%         assert(unique(LABELS(find(x.squiggle'==2)))==0)
%     end
% end
ybar = (LABELS*2-1)';
ybar(find(y==0)) =0;
% ybar = reshape(LABELS*2-1,[m n]);


end 