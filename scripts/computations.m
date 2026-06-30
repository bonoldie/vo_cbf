clear all; close all;
clc;

% syms vx vy a b n R d real;
% 
% 
% sh = (vy / a).^n - (vx / b).^n - 1;
% circ = vx^2 + (vy - d).^2 - R; 
% 
% dsh = gradient(sh, [vx vy]);
% 
% dcirc = gradient(circ, [vx vy]);


%% dh_dx

% syms a(x) b(x) v_x v_y n real;
% 
% dh_dx = gradient((v_y / b(x)).^n - (v_x / a(x)).^n, x);
% 
% title(['$$' latex(dh_dx) '$$'], 'Interpreter','latex', FontSize=35)

%% da_dx

% syms x x_obs y y_obs R tau real;

% da_dx = simplify(gradient(((((x - x_obs)^2 + (y - y_obs)^2 )^(1/2) - R) / tau), x));

% title(['$$' latex(da_dx) '$$'], 'Interpreter','latex', FontSize=35)

%% db_dx

% syms d(x) v_y_tan(x) a(x) R  n real

% db_dx = simplify(gradient((a * (R^2 - (v_y_tan - d)^2)^(1/2)) / ((v_y_tan^n - a^n)^(1/n)), x));

% title(['$$' latex(db_dx) '$$'], 'Interpreter','latex', FontSize=35)

%% dP_dx

syms n d(x) v_y_tan(x) a R real

P = d * v_y_tan^n - v_y_tan * a^n - (d^2 - R^2) * v_y_tan^(n-1) + d * a^n;

dP_dx = simplify(gradient(P, x));