clear all; close all;
clc;

syms xa_ xb_ ya_ yb_ vxa_ vya_ vxb_ vyb_ real;
syms b n vx_ vy_ r tau real;

% positions
xa = [xa_; ya_];
xb = [xb_; yb_];

% distance
d = norm(xb - xa);
a = (d - r) / tau;

%% superhyperbola
sh = a * (1 + (vx_/b)^n)^(1/n);

% 1) point lies on superhyperbola
eq1 = vy_ == sh;

% 2) point lies on circle centered at (0,d)
eq2 = vx_^2 + (vy_ - d)^2 == r^2;

% 3) slope of superhyperbola
dsh = diff(sh, vx_);

% circle slope (implicit derivative)
slope_circle = -vx_/(vy_ - d);

% tangency condition
eq3 = dsh == slope_circle;

%% solve for b

eqs = [eq1, eq2, eq3];

eqs = subs(eqs, [n], [2]);
sol = solve(eqs, b);


