using LinearAlgebra

n = 4;

a = rand(1:4,n);
#a = zeros(n);
#for i=1:n
#    a[i] = i%4 + 1
#end

A = zeros(2*n+3,1);
A[1] = 1;
A[2] = -1;
A[3] = -1;

B = vcat(zeros(3,n),Matrix(1.0I, n, n),Matrix(-1.0I, n, n));

B[3,1:n] = a;

ub = sum(a);
lb = 0;

b = vcat( [ub; -lb; 0], ones(n,1), zeros(n,1));

#objective is Max sum(y) - b, then we make it min
Fx = [1/4]
Fy = -ones(n,1)

fx = [0]
fy = -rand(n,1)

Gx = []
Gy = []

#ugly hack to match parsing
gx = A
gy = B
bg = -b
