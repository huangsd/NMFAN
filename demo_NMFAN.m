clear
close all
clc

%===========================================


load('cacmcisi.mat');


[n d] = size(data);

% number of clusters
c1 = length(unique(labels));

% k-NN, average number of neighbors 
k = 5;

lambda = 100;

ITE = 50;

[V, objV] = NMFAN(data', c1, k, lambda, ITE);
plot(objV)

% res = zeros(n,1);
% for j = 1:n
%    [tmp res(j)] = max(V(j,:));
% end

% kmeans discretization
res = kmeans(V,c1);

%============= evaluate:ACC NMI Purity
result = ClusteringMeasure(labels, res)



