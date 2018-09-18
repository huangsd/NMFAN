%
% l21_NMF with Adaptive Neighbors
% min_{S>=0, U>=0, V>=0}\sum(|x_i-x_j|^2*s+\gamma s^2)+\lambda Tr(V'LV)+|X-UV'|_{2,1}
% written by Shudong Huang on 30/3/2017
% Reference: Shudong Huang, Zenglin Xu, Zhao Kang, Yazhou Ren. Adaptive Local Structure Learning Regularized 
% Nonnegative Matrix Factorization. Neural Networks, under review. 
%
% ATTN: This package is free for academic usage. The code was developed by Mr. SD. Huang (huangsd@std.uestc.edu.cn). You can run
% it at your own risk. For other purposes, please contact Prof. Zenglin Xu (zenglin@gmail.com)
%
% ATTN2: This package was developed by Mr. SD. Huang (huangsd@std.uestc.edu.cn). For any problem concerning the code, please feel
% free to contact Mr. Huang.
%
function G = l21NMFAN(X,k,lambda)
%
% Input:
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% k: number of neighbors to determine the initial graph
% r: paremeter, which could be set to a large enough value. If r<0, then it is determined by algorithm with k
% nRepeat: iteration number
%
% Output:
% G: the clustering indicator matrix
%

% [result]=RMNMF(X,Y,K,mu,lambda)
% X: mFea*nSmp
% % this function implements the method described in the TKDD journal paper 
%  "Robust Manifold Non-Negative Matrix Factorization" by Jin Huang,
%   Feiping Nie, Heng Huang and Chris Ding. 
%  Please contact Jin Huang (huangjinsuzhou@gmail.com) if you had any question or 
%  found any bug in the code. Thanks!

% Input:
% X is row-based data matrix of size n by p
% Y is a vector of label of size n by 1
% K is the number of clusters initialized
% mu is the initial parameter value via ALM  0.001
% lambda is the regularity parameter 0.1

% Output:
% result:ACC, NMI, Purity 



% In this code, we use affinity graph construction code from others, please go
% to http://www.cad.zju.edu.cn/home/dengcai/Data/DimensionReduction.html
% to download constructW function and understand the parameter setting here
% you can also construct the affinity graph in your own way
%tic;
p=size(X,1);
n=size(X,2);
mu=0.001;
%lambda=0.1;

% This part of code constructs the Laplacian Graph
% options = [];
% options.Metric = 'Euclidean';
% options.NeighborMode = 'KNN';
% options.k = 5;
% options.WeightMode = 'Binary';

distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);
A = zeros(n);
rr = zeros(n,1);
for i = 1:n
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;
% if r <= 0
     r = mean(rr);
% end;
% lambda = mean(rr);

W0 = (A+A')/2;

W0 = lambda*W0;

D0 = diag(sum(W0));
L0 = D0 - W0; 

K = k;
% This part of code initialize the F and G of the data matrix
idx_G=kmeans(X',K,'Emptyaction','drop');
G=zeros(n,K);
for j=1:length(idx_G)
    G(j,idx_G(j))=1;
end
G=G+0.2;   % numerical stability purpose
idx_F=kmeans(X,K,'Emptyaction','drop');
F=zeros(p,K);
for i=1:length(idx_F)
    F(i,idx_F(i))=1;
end
F=F+0.2;

% This par of code initialize the gamma and sigma 
pho=8;
%eps=1e-4;
gamma=zeros(p,n);
sigma=zeros(size(G));
maxiter=50;

    % This part minimizes E
    temp1=X-F*G'+1/mu*gamma;
    [E]=L21_solver(temp1,1/mu);
    
    % This part minimizes F
    F=(X-E+1/mu*gamma)*G*(G'*G)^(-1);
    
    % This part minimizes H
    temp2=G+1/mu*sigma-lambda/mu*(G'*L0)';
    [H]=nonneg_L2(temp2);
    
    % This part minimizes G
    temp3=H-1/mu*sigma+lambda/mu*L0*H+(F'*(X-E+1/mu*gamma))';
    [U,S,V]=svd(temp3,0);
    G=U*V';
    
    % update the parameters
    gamma=gamma+mu*(X-F*G'-E);
    sigma=sigma+mu*(G-H);
    mu=pho*mu;
   
    
    
 iter=1;   % counting number of iterations

while iter<maxiter   
    
    distf = L2_distance_1(G',G');
    A = zeros(n);
    for i=1:n
           idxa0 = 1:idx(i,2:k+1);
        dfi = distf(i,idxa0);
        dxi = distX(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end;
    A = (A+A')/2;
    
    A = lambda*A;
    
    D = diag(sum(A));
    L = D-A;
    
    % This part minimizes E
    temp1=X-F*G'+1/mu*gamma;
    [E]=L21_solver(temp1,1/mu);
    
    % This part minimizes F
    F=(X-E+1/mu*gamma)*G*(G'*G)^(-1);
    
    % This part minimizes H
    temp2=G+1/mu*sigma-lambda/mu*(G'*L)';
    [H]=nonneg_L2(temp2);
    
    % This part minimizes G
    temp3=H-1/mu*sigma+lambda/mu*L*H+(F'*(X-E+1/mu*gamma))';
    [U,S,V]=svd(temp3,0);
    G=U*V';
    
    % update the parameters
    gamma=gamma+mu*(X-F*G'-E);
    sigma=sigma+mu*(G-H);
    mu=pho*mu; 
    
    iter=iter+1;
    
end

% Index = litekmeans(G, K, 'Replicates',20);



