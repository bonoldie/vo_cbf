% Obstacle-Tangent Super-Hyperbolic Velocity Obstacle (Exact Numerical Tangency)
% ------------------------------------------------------------
clear; clc;

%% 1. Define Physical Parameters
d = 0.8;    % Distance to the center of the obstacle (m)
r = 0.5;    % Enlarged safety radius (Robot radius + Obstacle radius) (m)
tau = 1.25;  % Time horizon (seconds)
n_tune = 6; % Exponent for the Super-Hyperbola (n > 2 flattens the bottom)

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

% Analytical tangency points for n=2
y_tan_n2 = d / (1 + (b_n2/a)^2);
x_tan_n2 = b_n2 * sqrt((y_tan_n2 / a)^2 - 1);


%% 4. Compute the PROPER Tangency for the Super-Hyperbola (n > 2)
% We use a numerical solver to find the exact 'b' that makes the minimum 
% distance from the super-hyperbola to the obstacle center exactly equal to 'r'.

% Use the analytical n=2 result as an excellent initial guess for the solver
options = optimset('Display', 'off', 'TolX', 1e-8);
err_func = @(b_guess) get_tangency_error(b_guess, a, n_tune, d, r);
b_super = fzero(err_func, b_n2, options);

% Now that we have the exact b_super, find the exact coordinates of tangency
dist_sq_super = @(x) x.^2 + (a * (1 + (x./b_super).^n_tune).^(1/n_tune) - d).^2;
[x_tan_super, ~] = fminbnd(dist_sq_super, 0, r);
y_tan_super = a * (1 + (x_tan_super./b_super).^n_tune).^(1/n_tune);


%% 5. Generate Data for Plotting
vx = linspace(-4, 4, 500); 

% A. Obstacle Circle
ang = linspace(0, 2*pi, 100);
obs_vx = r * cos(ang);
obs_vy = d + r * sin(ang);

% B. Standard VO V-Cone & Cap
vy_vo = abs(vx) * m_vo;
cap_vx = R_cap * cos(ang);
cap_vy = y_cap + R_cap * sin(ang);

% C. The Standard Tangent Hyperbola (n = 2)
vy_hyperbola_n2 = a * (1 + abs(vx / b_n2).^2).^(1/2);

% D. The PROPERLY Tangent Super-Hyperbola
vy_super_hyperbola = a * (1 + abs(vx / b_super).^n_tune).^(1/n_tune);


%% 6. Visualization
figure(1);
clf;
hold on; grid on;

% Fill the physical obstacle
fill(obs_vx, obs_vy, [1, 0.6, 0.6], 'EdgeColor', 'r', 'LineWidth', 1.5, ...
    'DisplayName', 'Physical Obstacle Circle');

plot(vx, vy_vo, '--k', 'LineWidth', 1.5, 'DisplayName', 'Standard VO Cone');
plot(cap_vx, cap_vy, '--b', 'LineWidth', 1.5, 'DisplayName', 'FVO Time Horizon (\tau) Cap');

% Plot the Standard Hyperbola (n=2)
plot(vx, vy_hyperbola_n2, '-b', 'LineWidth', 2, 'DisplayName', 'Standard Hyperbola (n=2)');
plot([x_tan_n2, -x_tan_n2], [y_tan_n2, y_tan_n2], 'o', 'MarkerEdgeColor', 'b', ...
    'MarkerFaceColor', 'y', 'MarkerSize', 6, 'DisplayName', 'Tangency (n=2)');

% Plot the Super-Hyperbola
plot(vx, vy_super_hyperbola, '-g', 'LineWidth', 3, ...
    'DisplayName', sprintf('Proper Super-Hyperbola (n=%d)', n_tune));
plot([x_tan_super, -x_tan_super], [y_tan_super, y_tan_super], 'o', 'MarkerEdgeColor', 'g', ...
    'MarkerFaceColor', 'y', 'MarkerSize', 8, 'DisplayName', sprintf('Tangency (n=%d)', n_tune));

% Formatting
xlabel('Lateral Relative Velocity v_x (m/s)', 'FontWeight', 'bold');
ylabel('Forward Relative Velocity v_y (m/s)', 'FontWeight', 'bold');
title(sprintf('Exact Tangent Super-Hyperbolic VO\n(d = %.1fm, r = %.1fm, \\tau = %.1fs, n = %d)', d, r, tau, n_tune));
axis equal; 
xlim([-4, 4]);
ylim([0, d + r + 1]);
legend('Location', 'southeast', 'FontSize', 10);
set(gca, 'FontSize', 12);
hold off;

saveas(gcf,'hyperbolic_vo.eps','epsc')

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