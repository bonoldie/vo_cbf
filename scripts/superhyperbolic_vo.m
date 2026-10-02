% Obstacle-Tangent Super-Hyperbolic Velocity Obstacle (Exact Numerical Tangency)
% ------------------------------------------------------------
clear; clc;

%% 1. Define Physical Parameters
d = 2;    % Distance to the center of the obstacle (m)
r = 1;    % Enlarged safety radius (Robot radius + Obstacle radius) (m)
tau = 1.25;  % Time horizon (seconds)
n_tune = 6; % Exponent for the Super-Hyperbola (n > 2 flattens the bottom)

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

%% 3. Compute the Analytical Tangent Hyperbola (Base n=2)
% Vertex on the FVO cut-off disk
a = (d - r) / tau; 

% The hyperbola osculates the cut-off disk at the vertex
[b_n2, y_tan_n2] = sh_width(a, y_cap, R_cap, 2);
x_tan_n2 = sqrt(R_cap^2 - (y_tan_n2 - y_cap)^2);


%% 4. Compute the PROPER Tangency for the Super-Hyperbola (n > 2)
% Narrowest super-hyperbola containing the FVO cut-off disk, and its
% tangency points
[b_super, y_tan_super] = sh_width(a, y_cap, R_cap, n_tune);
x_tan_super = sqrt(R_cap^2 - (y_tan_super - y_cap)^2);


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

purple = [.5 0 .5];

% Fill the physical obstacle
fill(obs_vx, obs_vy, [1, 0.6, 0.6], 'EdgeColor', 'r', 'LineWidth', 1.5, ...
    'DisplayName', 'Physical Obstacle Circle');

plot(vx, vy_vo, '--k', 'LineWidth', 1.5, 'DisplayName', 'Standard VO Cone');
plot(cap_vx, cap_vy, '--', 'Color', purple, 'LineWidth', 1.5, 'DisplayName', 'FVO Time Horizon (\tau) Cap');

cap_lib_approx = vx(a > vy_vo);
plot([min(cap_lib_approx), max(cap_lib_approx)], [a, a], 'Color', purple, 'LineWidth', 1.0, 'DisplayName', 'FVO Cap Linear Approximation');

% Plot the Standard Hyperbola (n=2)
plot(vx, vy_hyperbola_n2, '-b', 'LineWidth', 1.4, 'DisplayName', 'Standard Hyperbola (n=2)');
plot([x_tan_n2, -x_tan_n2], [y_tan_n2, y_tan_n2], 'o', 'MarkerEdgeColor', 'b', ...
    'MarkerFaceColor', 'y', 'MarkerSize', 6, 'DisplayName', 'Tangency (n=2)');

% Plot the Super-Hyperbola
plot(vx, vy_super_hyperbola, '-g', 'LineWidth', 1.4, ...
    'DisplayName', sprintf('Proper Super-Hyperbola (n=%d)', n_tune));
plot([x_tan_super, -x_tan_super], [y_tan_super, y_tan_super], 'o', 'MarkerEdgeColor', 'g', ...
    'MarkerFaceColor', 'y', 'MarkerSize', 6, 'DisplayName', sprintf('Tangency (n=%d)', n_tune));

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

saveas(gcf,'hyperbolic_vo.eps', 'epsc');

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