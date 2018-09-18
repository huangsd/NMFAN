%
% NMF with Adaptive Neighbors
% min_{S>=0, U>=0, V>=0}\sum(|x_i-x_j|^{2}s+\gamma s^{2})+\lambda Tr(V'LV)+|X-UV'|^{2}
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
function  [V, objV] = NMFAN(X, c, k, lambda, nRepeat)
%
% Input:
% X: dim*num data matrix, each column is a data point
% c: number of clusters
% k: number of neighbors to determine the initial graph
% r: paremeter, which could be set to a large enough value. If r<0, then it is determined by algorithm with k
% nRepeat: iteration number
%
% Output:
% V: the clustering indicator matrix
% objV: objective value
%
[dim,num]=size(X);
Norm = 2;
NormV = 0;
distX = L2_distance_1(X,X);
[distX1, idx] = sort(distX,2);
S = zeros(num);
A = S;
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2);
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;
% if r <= 0
     r = mean(rr);
% end;
% lambda = mean(rr);

A0 = (A+A')/2;

A0 = lambda*A0;

D0 = diag(sum(A0));
L0 = D0 - A0;  % L = D - W;
%[F, temp, evs]=eig1(L0, c, 0);  F=V
%
U = [];
V = [];
if isempty(U)
    U = abs(rand(dim,c));
    V = abs(rand(num,c));
end


 % ===================== update V ========================
 XU = X'*U;  % mnk or pk (p<<mn)
        UU = U'*U;  % mk^2
        VUU = V*UU; % nk^2
        
        if lambda > 0
            WV = A0*V;
            DV = D0*V;
            
            XU = XU + WV;
            VUU = VUU + DV;
        end        
        V = V.*(XU./max(VUU,1e-10));      
        % ===================== update U ========================
        XV = X*V;   % mnk or pk (p<<mn)
        VV = V'*V;  % nk^2
        UVV = U*VV; % mk^2     
        U = U.*(XV./max(UVV,1e-10));

        
       % y=zeros(nRepeat);
        nIter = 0;
  while nIter < nRepeat
      % [U,V] = NormalizeUV(U, V, NormV, Norm);
        distf = L2_distance_1(V',V');
    A = zeros(num);
    for i=1:num
%         if islocal == 1
        %    idxa0 = idx(i,2:k+1);
%         else
           idxa0 = 1:num;
        %end;
        
%          di = distX1(i,2:k+2);
%     rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
%     id = idx(i,2:k+2);
%     A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
        
        dfi = distf(i,idxa0);
        dxi = distX(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end;
    
%      r = mean(rr);
%      lambda = mean(rr);

    A = (A+A')/2;
    
    A = lambda*A;
    
    D = diag(sum(A));
    L = D-A;
   % V_old = V;
           
        % ===================== update V ========================
         XU = X'*U;  % mnk or pk (p<<mn)
        UU = U'*U;  % mk^2
        VUU = V*UU; % nk^2
        
        if lambda > 0
            WV = A*V;
            DV = D*V;
            
            XU = XU + WV;
            VUU = VUU + DV;
        end        
        V = V.*(XU./max(VUU,1e-10));      
        % ===================== update U ========================
        XV = X*V;   % mnk or pk (p<<mn)
        VV = V'*V;  % nk^2
        UVV = U*VV; % mk^2     
        U = U.*(XV./max(UVV,1e-10));
        [obj, dV] = CalculateObj(X, U, V, L);
         nIter = nIter + 1;
         objV(nIter)=obj;
  end
  
% y = litekmeans(V, c, 'Replicates',20);

end

 

  function [obj, dV] = CalculateObj(X, U, V, L, deltaVU, dVordU)
    MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    nSmp = size(X,2);
    mn = numel(X);
    nBlock = ceil(mn/MAXARRAY);

    if mn < MAXARRAY
        dX = U*V'-X;
        obj_NMF = sum(sum(dX.^2));
        if deltaVU
            if dVordU
                dV = dX'*U + L*V;
            else
                dV = dX*V;
            end
        end
    else
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        PatchSize = ceil(nSmp/nBlock);
        for i = 1:nBlock
            if i*PatchSize > nSmp
                smpIdx = (i-1)*PatchSize+1:nSmp;
            else
                smpIdx = (i-1)*PatchSize+1:i*PatchSize;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV + L*V;
            end
        end
    end
    if isempty(L)
        obj_Lap = 0;
    else
        obj_Lap = sum(sum((V'*L).*V'));
    end
    obj = obj_NMF+obj_Lap;
  end
    




function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U,2);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K);
        end
    end
end

    


