% Compute the CBF from the super hyperbolic boundary
clear all; close all; 
clc;

% here we assume that the system is a single integrator

% superhyperbola degree
n = 6;

delta = 2;

syms x y u1 u2 a b real;

% Superhyperbola
sh(x) = a * (1 + (x / b).^n).^(1/n);

% CBF
h(x, y) = sh(x) - y;

alpha = delta * h(x,y);

cbf = gradient(h)' * [0;0] + gradient(h)' * eye(2) * [u1; u2] +  alpha;


%% PLOTTING

a_ = 2;
b_ = 0.83;

Hsym = subs(h,[a,b],[a_,b_]);
% CBFsym = subs(cbf,[a,b],[a_,b_]);

% Sampling grid
xv = linspace(-3,3,300);
yv = linspace(-3,3,300);
[X,Y] = meshgrid(xv,yv);

% Evaluate your symbolic h on the grid
Hfun = matlabFunction(Hsym,'Vars',[x,y]);
H = Hfun(X,Y);

% Evaluate your symbolic cbf on the grid
% CBFfun = matlabFunction(CBFsym,'Vars',[x,y]);
% CBF = CBFfun(X,Y);

figure;

subplot(1,2,1);
hold on;
axis equal;
grid on;

% Positive region
idx_pos = H > 0;
scatter(X(idx_pos), Y(idx_pos), 5, [0 0.7 0], 'filled');

% Negative region
idx_neg = H < 0;
scatter(X(idx_neg), Y(idx_neg), 5, [0.8 0 0], 'filled');

% Zero level set
contour(X,Y,H,[0 0],'k','LineWidth',2);

xlabel('x');
ylabel('y');
legend('h > 0','h < 0','h = 0');

% subplot(1,2,2);
% hold on;
% axis equal;
% grid on;
% 
% % Positive region
% idx_pos = CBF > 0;
% scatter(X(idx_pos), Y(idx_pos), 5, [0 0.7 0], 'filled');
% 
% % Negative region
% idx_neg = CBF < 0;
% scatter(X(idx_neg), Y(idx_neg), 5, [0.8 0 0], 'filled');
% 
% % Zero level set
% contour(X,Y,CBF,[0 0],'k','LineWidth',2);
% 
% xlabel('x');
% ylabel('y');
% legend('cbf > 0','cbf < 0','cbf = 0');