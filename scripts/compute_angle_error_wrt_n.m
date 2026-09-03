% Obstacle-Tangent Super-Hyperbolic Velocity Obstacle (Exact Numerical Tangency)
% ------------------------------------------------------------
clear; clc;

%% 1. Define Physical Parameters
d = 2;    % Distance to the center of the obstacle (m)
r = 1;    % Enlarged safety radius (Robot radius + Obstacle radius) (m)
tau = 1.25;  % Time horizon (seconds)

N = [2,4,6,8,10,12];

angle_errors = [];
m_shs = [];

% Physical constraints checks
if d <= r
    error('Distance to obstacle (d) must be strictly greater than safety radius (r).');
end
if tau < 1.0
    error('Tau must be >= 1.0 for the hyperbola to wrap the physical obstacle.');
end

%% 2. Compute the Standard VO and FVO Cutoff Parameters
theta = asin(r / d);      
m_vo = cot(theta);        
y_cap = d / tau;          
R_cap = r / tau;   

%% 3. Compute the Analytical Tangent Hyperbola (Base n=2)
a = (d - r) / tau; 

D_term = d^2 - r^2 - a^2;
inner_term = max(0, D_term^2 - 4 * a^2 * r^2);
b_n2 = sqrt(0.5 * (D_term - sqrt(inner_term)));


%% 4. Compute the PROPER Tangency for the Super-Hyperbola 
for n=N
    % We use a numerical solver to find the exact 'b' that makes the minimum 
    % distance from the super-hyperbola to the obstacle center exactly equal to 'r'.
    
    % Use the analytical n=2 result as an excellent initial guess for the solver
    options = optimset('Display', 'off', 'TolX', 1e-8);
    err_func = @(b_guess) get_tangency_error(b_guess, a, n, d, r);
    b_super = fzero(err_func, b_n2, options);
    
    m_sh = a/b_super;
    m_shs(end+1) = m_sh;

    angle_error = atan(abs((m_sh - m_vo)/(1 + m_vo*m_sh)));

    angle_errors(end+1) = rad2deg(angle_error);
end

%% 7. Local Functions for Numerical Optimization
function err = get_tangency_error(b, a, n, d, r)
    % Find the x-coordinate that minimizes the distance from the curve to the obstacle center
    % Curve equation: y(x) = a * (1 + (x/b)^n)^(1/n)
    % Distance squared to (0, d) = x^2 + (y(x) - d)^2
    
    % Define the distance squared objective function
    dist_sq = @(x) x.^2 + (a * (1 + (x./b).^n).^(1/n) - d).^2;
    
    % Find the minimum distance (search between x=0 and x=r is safe for this geometry)
    [~, min_dist_sq] = fminbnd(dist_sq, 0, r);
    
    % The error is the difference between the actual minimum distance and the desired radius 'r'
    err = sqrt(min_dist_sq) - r;
end