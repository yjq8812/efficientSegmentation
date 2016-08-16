function [err,errList] = testEvaluation(X,Y,w,lossfn)
% NOTE: need to be consistent with inference model, loss function etc
errList = zeros(length(X),1);
for i=1:length(X)
    x = X{i};
    
    m = size(x.features,1);
    n = size(x.features,2);
    
    w_u = w(1:end-3);
    w_p = w(end-2:end);
    oureps = 1e-6;
if norm(w_p,1)~=0
    scaling = norm(w_p,1);
else
    scaling = 1;
end
%assert(w_p(3)*scaling<=oureps);
%assert(w_p(2)*scaling>=-oureps);
%assert(w_p(1)*scaling>=-oureps);% -6.2463e-11
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
    % DONT ADD LOSS AUGMENTATION FOR TESTING TIME
    
    CLASS = ones(1,m*n);% y^i in {-1,1}
    LABELCOST = [-w_p(2) -w_p(3);-w_p(3) -w_p(1)];
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
    
    [LABELS ENERGY ENERGYAFTER] = GCMex(CLASS, single(UNARY), PAIRWISE, single(LABELCOST),EXPANSION);
    
    if isfield(x,'squiggle')
        if ~isempty(find(x.squiggle'==1))
            assert(unique(LABELS(find(x.squiggle'==1)))==1)
        end
        if ~isempty(find(x.squiggle'==2))
            assert(unique(LABELS(find(x.squiggle'==2)))==0)
        end
    end
    ypred = reshape(LABELS*2-1,[m n]);
    
    errList(i) = lossfn(Y{i},ypred); % (y,ybar)
    
end

err = mean(errList);

end
