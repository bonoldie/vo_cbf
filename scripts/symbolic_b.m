clear all;
close all;
clc;

syms a b vx vy d r real;

n = 2;

CIRCLE = vx^2 + (vy - d) ^ 2 == r^2;

SH = (vy/a)^n - (vx/b)^n == 1;

GRAD_CIRCLE = gradient(lhs(CIRCLE), [vx; vy]);
GRAD_SH = gradient(lhs(SH), [vx; vy]);

%tangency condition
TANGENCY_COND = GRAD_CIRCLE(1)*GRAD_SH(2) - GRAD_CIRCLE(2)*GRAD_SH(1);

vy_tan = a^2*d / (a^2 + b^2);
vx_tan = b * sqrt((vy_tan/a)^2 - 1);


SUBS_CIRCLE = subs(CIRCLE, [vx; vy], [vx_tan; vy_tan]);