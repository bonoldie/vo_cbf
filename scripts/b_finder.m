clear all;close all;
clc;

syms vx vy b d r tau real;

n = 2;

a = (d - r) / tau;

circle_eq = vx^2 + (vy - d)^2 - r^2;
sh_eq = (vy/a).^(n) - (vx/b).^(n) - 1;

dcircle_eq = gradient(circle_eq, [vx vy]);
dsh_eq = gradient(sh_eq, [vx vy]);

% circle_eq_vx = d + (r^2 - vx^2).^(1/2);
% sh_eq_vx = a * (1 + (vx/b).^(n)).^(1/n);

% dcircle_eq_vx = diff(circle_eq_vx, vx);
% dsh_eq_vx = diff(sh_eq_vx, vx);

conditions = [ ...
    tau >= 1;  ...
    r > 0;  ...
    d > r; ...
];

sol = solve([ ...
    sh_eq == 0; ...
    circle_eq == 0; ...
    dcircle_eq(1)*dsh_eq(2) - dcircle_eq(2)*dsh_eq(1) == 0; ... % dcircle_eq_vx == dsh_eq_vx; ... % dcircle_eq' * dsh_eq == 1; ... 
    conditions
], [vx, b], "Real", true, "ReturnConditions", true);

% sol_vy = solve([ ...
%     dcircle_eq(1)*dsh_eq(2) - dcircle_eq(2)*dsh_eq(1) == 0; ... % dcircle_eq_vx == dsh_eq_vx; ... % dcircle_eq' * dsh_eq == 1; ... 
% ], vy, "Real", true, "ReturnConditions", true);



%% Plotting


b_test = simplify(subs(sol.b, [d, r, tau], [2.0, 1.0, 1.25]));
conditions_test = simplify(subs(sol.conditions, [d, r, tau], [2.0, 1.0, 1.25]));

