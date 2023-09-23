clc;
clear;

A = diag([1,2,3,4,5]);
B = [1; 1; 1; 1; 1];

% A = [0 1;-2 3];
% B = [0; 1];
% k = [1 -4];
dim  = size(A,1);
eigenvalues = [-9 -10 -11 -12 -13];

k  = -(place(A,B, eigenvalues));

Q = eye(dim);
Ac=A+B*k;
P=lyap(Ac',Q);
s=0.99/(2*norm(P*B*k))