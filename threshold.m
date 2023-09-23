clc;
clear;

A = [1 0 0;
     0 2 0;
     0 0 3];
B = [1; 1; 1];

dim  = size(A,1);
eig = [-5 -6 -7];

k  = -(place(A,B, eig));

Q = eye(dim);
Ac=A+B*k;
P=lyap(Ac',Q);
s=0.99/(2*norm(P*B*k))