% function phi = unaryPotentialtest(H,W,x, y, w_u,l,ygt) % gamma
% GCMEX needs:
% UNARY:: A CxN matrix specifying the potentials (data term) for
%      each of the C possible classes at each of the N nodes.
% for now assuming binary classification
x=rand(10,3);
w_u=[1;2;3];
y=randint(10,1);
ygt=randint(10,1);
phiy=zeros(length(y),2);
phiy(:,1)=1-y;
phiy(:,2)=y;
phiygt=zeros(length(ygt),2);
phiygt(:,1)=1-ygt;
phiygt(:,2)=ygt;


k = size(x,2); % k features
phi = 0;
for i=1:k
    phi_i(:,1)=x(:,i).*(1-y);
    phi_i(:,2)=x(:,i).*y;
    phi = phi + w_u(i)*phi_i;% dim w_u = k; dot(phi_u,w_u)
end
loss = phiy-phiygt;
lossinferenc = phi+loss;
% function phi = unaryPotential(H,W,x, y, w_u,l,ygt) % gamma
% % GCMEX needs:
% % UNARY:: A CxN matrix specifying the potentials (data term) for
% %      each of the C possible classes at each of the N nodes.
% % for now assuming binary classification
% k = size(x,2); % k in A.Osokin's paper
% phi = 0;
% for i=1:k
%     phi_i = zeros(length(x),2); % C = 2 binary in GCMex
%     phi_i(:,1)=x(:,1).*y;
%     phi_i(:,2)=x(:,1).*(1-y);
%     phi = phi + w_u(i)*phi_i;% dim w_u = k; dot(phi_u,w_u)
% end
