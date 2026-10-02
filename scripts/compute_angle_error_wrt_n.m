% Obstacle-Tangent Super-Hyperbolic Velocity Obstacle (Exact Numerical Tangency)
% ------------------------------------------------------------
clear; clc;

%% 1. Define Physical Parameters
d = 2;    % Distance to the center of the obstacle (m)
r = 1;    % Enlarged safety radius (Robot radius + Obstacle radius) (m)
tau = 1.25;  % Time horizon (seconds), the slopes do not depend on it

N = [2,4,6,8,10,12];

angle_errors = [];
m_shs = [];

% Physical constraints checks
if d <= r
    error('Distance to obstacle (d) must be strictly greater than safety radius (r).');
end
if tau <= 0
    error('Tau must be positive.');
end

%% 2. Compute the Standard VO and FVO Cutoff Parameters
theta = asin(r / d);      
m_vo = cot(theta);        
y_cap = d / tau;          
R_cap = r / tau;   

%% 3. Vertex of the super-hyperbolas, on the FVO cut-off disk
a = (d - r) / tau; 


%% 4. Compute the PROPER Tangency for the Super-Hyperbola 
for n=N
    % Narrowest super-hyperbola containing the FVO cut-off disk
    b_super = sh_width(a, y_cap, R_cap, n);
    
    m_sh = a/b_super;
    m_shs(end+1) = m_sh;

    angle_error = atan(abs((m_sh - m_vo)/(1 + m_vo*m_sh)));

    angle_errors(end+1) = rad2deg(angle_error);
end

%% 7. Local Functions
function [b, y_star] = sh_width(a, d, r, n)
    % Width b of the narrowest super-hyperbola with vertex a that contains the
    % disk of centre (0, d) and radius r, and ordinate y_star of the tangency
    % point (Lemma 1 of the paper)
    if n == 2
        % The hyperbola osculates the disk at the vertex
        b = sqrt(a * r);
        y_star = a;
        return;
    end

    % Tangency polynomial P(y) deflated by its root y = a, negative at a and
    % positive at d - r^2/d
    k = 0:(n - 1);
    deflated_tangency_poly = @(y) (d + r - y) * y^(n - 1) + (y - d) * sum(y.^(n - 1 - k) .* a.^k);

    y_star = fzero(deflated_tangency_poly, [a, d - r^2 / d]);
    b = a * sqrt(r^2 - (y_star - d)^2) / (y_star^n - a^n)^(1/n);
end