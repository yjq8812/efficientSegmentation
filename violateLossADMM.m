function ybar = violateLossADMM(param,lossFn,model,x,y,y_a,y_b_old,rho,u)
% Function Callback by VIOLATEMARGINADMM.m
% problem: argmin_ybar -\Delta(y,ybar) + rho/2*norm(y_a-ybar+u)^2
% via a method depends on the submodularity of the loss
% with y: groundtruth
%      y_a: fixed

m = size(x.features,1);
n = size(x.features,2);

switch param.setFnname
    case 'TestSupermodular'
        %
        
    case 'square'
        % argmin_y -lossFn(y,ybar) + rho/2*norm(y_a-ybar+u)^2:=F
        % F = -|y~=ybar|^2+rho/2*||y_a-y+u||^2
        % def: x = (y~=ybar),
        % then ybar(x==0) = y(x==0), ybar(x==1) = -y(x==1)
        % in the case that y in [-1,1]^p, ybar = y-2*y.*x or inversly
        % F = -|x|^2 + rho/2*norm(y_a-(y-2*y.*x )+u)^2 
        % which is symmetric submodular + asymmetric modular
        
        F = @(A)(violateLossSFOHelper(A,lossFn,y,y_a,rho,u));
%         tic
        A = sfo_min_norm_pointYu(F, [1:length(y)]);
%         fprintf(['sfo running: ' num2str(toc) ' s.\n']);
        ybar = y;
        ybar(A) = y(A)*-1;
        if isfield(x,'squiggle')
            x.squiggle = reshape(x.squiggle,[prod(size(y)) 1]);
            ybar(find(x.squiggle==2))=-1;
            ybar(find(x.squiggle==1))=+1;
        end
    
    case '8connected'
        % this is an 8-connected supermodular loss that pays a Hamming loss
        % + 1/8*number of neighbors that also make a mistake.  We
        % decompose to add 4 connected loss and Hamming loss to
        % loss-augmented inference, and here we do graph cuts for the
        % diagonal pairwise costs.

        UNARY = zeros(2,length(y));
        % U = - Hamming + rho/2*|y_a-ybar+u|_2^2
        ind = find(y~=0);
        
        UNARY(1,ind) = -0 + rho/2* (y_a(ind)-y(ind)+u(ind)).^2 ;% if ybar = y*
        UNARY(2,ind) = -1 + rho/2* (y_a(ind)+y(ind)+u(ind)).^2 ;% if ybar ~= y*
        
        CLASS = ones(1,length(y));% y^i in {-1,1}
        
        LABELCOST =  - [0 0; 0 param.falseall];% 
        EXPANSION = 0;
        % now create adjacency matrix for grid graph
        PAIRWISE = generateDiagonalPairwise(x.PAIRWISE,m,n);
        maxDegree = x.maxDegree;
        
        if isfield(x,'squiggle')
            ourInf = max(max(abs(LABELCOST)))*maxDegree + max(max(UNARY));
            x.squiggle = reshape(x.squiggle,[length(y) 1]);
            UNARY(1,find(x.squiggle==1))=-ourInf; % UNARY(1,:): ybar = y*
            UNARY(1,find(x.squiggle==2))=-ourInf;
            UNARY(2,find(x.squiggle==2))=+ourInf; % UNARY(2,:): ybar ~= y*
            UNARY(2,find(x.squiggle==1))=+ourInf;
        end
        assert(LABELCOST(1,1)+LABELCOST(2,2)-LABELCOST(1,2)-LABELCOST(2,1)<=0) %submodularity required

        % UCLA Fulkerson wrapper
        [LABELS ENERGY ENERGYAFTER] = GCMex(CLASS, single(UNARY), sparse(PAIRWISE), single(LABELCOST),EXPANSION);
        
        % LABELS==0 means ybar = y*
        % LABELS==1 means ybar ~= y*
        
        ybar(find(LABELS==0)) = y(find(LABELS==0));
        ybar(find(LABELS==1)) = y(find(LABELS==1))*-1;
        ybar(y==0) = 0;
    
    case 'Hamming'
%%------------ naive solution with Hamming loss ----------------%%
        ybar = y;
        ind = find(y==1);
        ybar(ind) = sign( (y_a(ind)+u(ind)+1).^2 - (y_a(ind)+u(ind)-1).^2 - 2/rho);
        ind = find(y==-1);
        ybar(ind) = sign( (y_a(ind)+u(ind)+1).^2 - (y_a(ind)+u(ind)-1).^2 + 2/rho);
        
        if isfield(x,'squiggle')
        x.squiggle = reshape(x.squiggle,[prod(size(y)) 1]);
        ybar(find(x.squiggle==2))=-1;
        ybar(find(x.squiggle==1))=+1;
        end
        
    case 'HammingPotts'
%%--- graph cut with a supermodular loss: Hamming + Potts -----%%
        UNARY = zeros(2,length(y));
        % NOW ADD MODULAR TERMS
        % U = - Hamming + rho/2*|y_a-y_b+u|_2^2
        ind = find(y~=0);
        UNARY(1,ind) = -(y(ind)+1)./2 + rho/2* (y_a(ind)+1+u(ind)).^2;% if ybar = -1
        UNARY(2,ind) = -(1-y(ind))./2 + rho/2* (y_a(ind)-1+u(ind)).^2;% if ybar = 1
        
        CLASS = ones(1,length(y));% y^i in {-1,1}
        
        LABELCOST =  - [0 param.falsesingle; param.falsesingle param.falseall];
        EXPANSION = 0;
        % now create adjacency matrix for grid graph
        PAIRWISE = x.PAIRWISE;
        maxDegree = x.maxDegree;
        
        if isfield(x,'squiggle')
            ourInf = max(max(abs(LABELCOST)))*maxDegree + max(max(UNARY))+1;
            x.squiggle = reshape(x.squiggle,[length(y) 1]);
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
            if ~isempty(find(x.squiggle'==1))
                assert(unique(LABELS(find(x.squiggle'==2)))==0)
            end
        end
        
        ybar = (LABELS*2-1)';
    case 'TestSupermodular_old'
        % M comes from: rho/2*|y_a-y+u|_2^2
        M = y_a + u;
        lossFunctionHelper = @(x)(-lossFn(x));
        fprintf(['CuttingPlane is running..  \n']);
        tic
        ybar = CuttingPlaneOpt(lossFunctionHelper,M',rho);
        toc
        yold =ybar;
        ybar = -ones(size(yold));
        ybar(find(yold>1e-6))=1;
        ybar = ybar';
        
    otherwise
        fprintf('** No specific method for maximizing this loss. Runing SFO!\n')
        %%---------- SFO: general solution with supermodular/modular loss --------%%
        % maximize a supermodular (minimize a submodular) by SFO
        % [A,subopt] = sfo_min_norm_point(F,V, opt)
        % function A = sfo_min_norm_point(F,V, opt)
        % F: Submodular function
        % V: index set
        
        F = @(A)(violateLossSFOHelper(A,lossFn,y,y_a,rho,u));
        tic
        A = sfo_min_norm_point(F, [1:length(y)]);
        toc
        ybar = y;
        ybar(A) = y(A)*-1;
        if isfield(x,'squiggle')
            x.squiggle = reshape(x.squiggle,[prod(size(y)) 1]);
            ybar(find(x.squiggle==2))=-1;
            ybar(find(x.squiggle==1))=+1;
        end
end


        
% function delta = lossFunctionHelper(x)
% if(~exist('x','var'))
%     delta = length(M);
%     return
% end
% 
% if(length(x)~=length(M) || length(unique(x))>2)
%     ind = x;
%     x = zeros(size(M));
%     x(ind) = 1;
% end
% 
% delta = -sum(x).^2 ./length(x);% should be consistent with LOSSFUNCTION!!
% 
% end      
    
end


function adj2 = generateDiagonalPairwise(gridAdj,m,n)
% generate a diagonal pairwise adjacency matrix for a mxn grid graph
% gridAdj - the adjacency matrix for a 4-connected grid graph
% m - number of rows
% n - number of columns


r = n;
c = m;
diagVec1 = repmat([ones(c-1,1); 0],r,1);  %# Make the first diagonal vector
                                          %#   (for horizontal connections)
diagVec1 = diagVec1(1:end-1);             %# Remove the last value
diagVec2 = [0; diagVec1(1:(c*(r-1)))];    %# Make the second diagonal vector
                                          %#   (for anti-diagonal connections)
diagVec3 = ones(c*(r-1),1);               %# Make the third diagonal vector
                                          %#   (for vertical connections)
diagVec4 = diagVec2(2:end-1);             %# Make the fourth diagonal vector
                                          %#   (for diagonal connections)
%adj = diag(diagVec1,1)+...                %# Add the diagonals to a zero matrix
%      diag(diagVec2,c-1)+...
%      diag(diagVec3,c)+...
%      diag(diagVec4,c+1);
  
  
adj2 = spalloc(size(gridAdj,1),size(gridAdj,2),1);%# Add the diagonals to a zero matrix
adj2 = spdiags([0;diagVec1],1,adj2);
adj2 = spdiags([zeros(c-1,1);diagVec2],c-1,adj2);
adj2 = spdiags([zeros(c,1);diagVec3],c,adj2);
adj2 = spdiags([zeros(c+1,1);diagVec4],c+1,adj2);
adj2 = adj2+adj2';

%adj = adj+adj.';                         %'# Add the matrix to a transposed
                                          %#   copy of itself to make it
                                          %#   symmetric
%adj = adj-gridAdj;  % subtract 4-connected grid graph from 8 connected graph.

%norm(adj(:)-adj2(:),1)

end

function result = violateLossSFOHelper(ind,lossFn,y,y_a,rho,u)

y_temp = y;
y_temp(ind) = y(ind)*-1;
% ind2 = find(y~=0);
result = -lossFn(y,y_temp) + rho./2 * (norm(y_a-y_temp+u,2)^2);


end