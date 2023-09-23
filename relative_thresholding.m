clc
clear all

A=[0 1; -2 3];
B=[0;1];
K=[0 -6];
Ac=A+B*K;

Q=eye(2);
P=lyap(Ac',Q);
s=0.99/(2*norm(P*B*K));
alpha=1/(1+s);

% Plot tau_s(theta) for theta in [0,pi)
Te=[];
Theta1=[];
for theta = 0:.02:pi
x0=[cos(theta); sin(theta)];
tspan=[0 10];
opts=odeset('Events',@(t,x)event(t,x,x0,s));
% opts=odeset('Events',@(t,x)event(t,x,x0,A,B,K,P));
[t,x,te,xe,ie] = ode45(@(t,x) A*x+B*K*x0,tspan,x0,opts);
Te=[Te,te];
end

theta = 0:.02:pi;
plot(theta,Te,'k','LineWidth',3)
grid
xlabel('\theta','FontSize',30);
ylabel('\tau_s(\theta)','FontSize',30);
axis([0 pi 0 inf]);
set(gca,'FontSize',30);


% MATLAB code to find t_k+1 - t_k when x(t_k)= [1;2]
x_k=[1 2];
tspan=[0 10];
opts=odeset('Events',@(t,x)event(t,x,x0,s));
% opts=odeset('Events',@(t,x)event(t,x,x0,A,B,K,P));
[t,x,te,xe,ie] = ode45(@(t,x) A*x+B*K*x0,tspan,x0,opts);
tau_k=te

function [value,isterminal,direction] = event(t,x,x0,s)
value = norm(x0-x)-s*norm(x);     % Detect height
% value=x'*P*x-x0'*P*x0*exp(-s*t);
% value=2*x'*P*(A*x+B*K*x0);
isterminal = 1;   % Stop the integration
direction = 0;   % Negative direction only
end