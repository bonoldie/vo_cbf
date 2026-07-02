clear all; close all; clc;

% robot state
syms x_r y_r vx_r vy_r u_x u_y real;

system_state = [x_r; y_r; vx_r; vy_r];

u = [u_x; u_y];

A = [ 0 0 1 0; 0 0 0 1; 0 0 0 0; 0 0 0 0];

F = A*system_state;

G = [ 0 0; 0 0; 1 0; 0 1];

% obstacle state
syms x_ob y_ob vx_ob vy_ob real;

obstacle_state = [x_ob; y_ob; vx_ob; vy_ob];

% Velocity-space variables

syms vx vy;

% velocity to check
v_rel = [vx_r - vx_ob; vy_r - vy_ob];

vx_rel = v_rel(1);
vy_rel = v_rel(2);

% super-hyperbola parameters

syms R tau n real;

d = sqrt((x_ob - x_r)^2 + (y_ob - y_r)^2); % distance between the obstacle and the 
a = (d - R) / tau;

% these two symbols are the tangency coordinates
syms vy_t(x_r, y_r) dvy_t_dx_r dvy_t_dy_r real;

% coord at which the curve is tangent to the circle
vx_t = sqrt(R^2 - (vy_t - d)^2);

b = (a * vx_t) / (vy_t^n - a^n)^(1/n);

db_dx_r = gradient(b, [x_r]);

% equation to find dvy_t/dx_r and dvy_t/dy_r (from the tangency condition)
eq_vy_t = d * vy_t^n -  a^n * vy_t - (d^2 - R^2) * vy_t ^ (n-1) + d * a^n == 0;
eq_vy_t = simplify(eq_vy_t);

grad_eq_vy_t_x_r = simplify(gradient(lhs(eq_vy_t), [x_r]));
grad_eq_vy_t_y_r = simplify(gradient(lhs(eq_vy_t), [y_r]));

grad_eq_vy_t_x_r = subs(grad_eq_vy_t_x_r,diff(vy_t(x_r, y_r), x_r),dvy_t_dx_r);

grad_eq_vy_t_y_r = subs(grad_eq_vy_t_y_r, diff(vy_t(x_r, y_r), y_r),dvy_t_dy_r);
 
% dvy_t/dx_r and dvy_t/dy_r
dvy_t_dx_r_sol = solve(grad_eq_vy_t_x_r, dvy_t_dx_r, ReturnConditions=true);
dvy_t_dy_r_sol = solve(grad_eq_vy_t_y_r, dvy_t_dy_r, ReturnConditions=true);

% Now we can subs into db_dx_r
db_dx_r = simplify(subs(db_dx_r, diff(vy_t(x_r, y_r), x_r), dvy_t_dx_r_sol.dvy_t_dx_r));

% Candidate CBF
h = a * (1 + (vx_rel / b)^n)^(1/n) - vy_rel;

% CBF gradient

dh_dx_r = simplify(gradient(h, [x_r]));
dh_dx_r = simplify(subs(dh_dx_r,[diff(vy_t(x_r, y_r), x_r)], [dvy_t_dx_r_sol.dvy_t_dx_r]));

dh_dy_r = simplify(gradient(h, [y_r]));
dh_dy_r = simplify(subs(dh_dy_r,[diff(vy_t(x_r, y_r), y_r)], [dvy_t_dy_r_sol.dvy_t_dy_r]));

dh_dvx_r = simplify(gradient(h, [vx_r]));
dh_dvy_r = simplify(gradient(h, [vy_r]));

grad_h_T = [dh_dx_r, dh_dy_r, dh_dvx_r, dh_dvy_r];

% Computing gradient condition on CBF
syms alpha real;

U_cbf = grad_h_T*F + grad_h_T*G*u + alpha * h;
U_cbf = simplify(U_cbf);

%% Example

robot_state1 = [0; 0; 0; 0];
robot_r = 0.2;

obstacle_state1 = [1; 0; 0; 0];
obstacle_r = 0.2;

d_val = double(subs(d, [x_r; y_r; x_ob; y_ob], [robot_state1(1);robot_state1(2);obstacle_state1(1);obstacle_state1(2)]));
R_val = robot_r + obstacle_r;
tau_val = 1.2;
n_val = 4;

sh_out = SHVO(d_val, R_val, tau_val, n_val);

alpha_val = 1;

U_cbf1 = subs(U_cbf, [vy_t], [sh_out.superHyperbola.y_tan_super]);
U_cbf2 = subs(U_cbf1, [tau; R; alpha; n], [tau_val;R_val; alpha_val;n_val]);
U_cbf3 = subs(U_cbf2, [x_r; y_r; vx_r; vy_r; x_ob; y_ob; vx_ob; vy_ob], [robot_state1; obstacle_state1]);
