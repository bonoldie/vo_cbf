function out = SHVO(d, r, tau, n_tune, doPlot, savePath)
% Obstacle-Tangent Super-Hyperbolic Velocity Obstacle
% --------------------------------------------------
% Callable function version.
%
% By default, it computes everything numerically and returns data only.
% No figure is opened unless doPlot = true.
%
% Usage:
%   out = obstacleTangentSHVO()
%   out = obstacleTangentSHVO(2, 1, 1.25, 6)
%   out = obstacleTangentSHVO(2, 1, 1.25, 6, true)
%   out = obstacleTangentSHVO(2, 1, 1.25, 6, true, 'hyperbolic_vo.eps')
%
% Inputs:
%   d        : distance to obstacle center
%   r        : enlarged safety radius
%   tau      : time horizon
%   n_tune   : super-hyperbola exponent
%   doPlot   : optional, true/false
%   savePath : optional, e.g. 'hyperbolic_vo.eps'
%
% Output:
%   out      : struct containing parameters, tangency values, and curves

    %% Defaults
    if nargin < 1 || isempty(d)
        d = 2;
    end

    if nargin < 2 || isempty(r)
        r = 1;
    end

    if nargin < 3 || isempty(tau)
        tau = 1.25;
    end

    if nargin < 4 || isempty(n_tune)
        n_tune = 6;
    end

    if nargin < 5 || isempty(doPlot)
        doPlot = false;
    end

    if nargin < 6 || isempty(savePath)
        savePath = '';
    end

    %% Physical constraint checks
    if d <= r
        error('Distance to obstacle d must be strictly greater than safety radius r.');
    end

    if tau <= 0
        error('Tau must be positive.');
    end

    if n_tune <= 2
        error('n_tune must be > 2 for a super-hyperbola.');
    end

    if mod(n_tune, 2) ~= 0
        error('n_tune should be even to preserve left-right symmetry.');
    end

    %% Standard VO and FVO cutoff parameters
    theta = asin(r / d);
    m_vo = cot(theta);

    y_cap = d / tau;
    R_cap = r / tau;

    %% Tangent hyperbola, n = 2
    % Vertex on the FVO cut-off disk
    a = (d - r) / tau;

    % The hyperbola osculates the cut-off disk at the vertex
    [b_n2, y_tan_n2] = shWidth(a, y_cap, R_cap, 2);
    x_tan_n2 = sqrt(R_cap^2 - (y_tan_n2 - y_cap)^2);

    %% Tangent super-hyperbola
    % Narrowest super-hyperbola containing the FVO cut-off disk
    [b_super, y_tan_super] = shWidth(a, y_cap, R_cap, n_tune);
    x_tan_super = sqrt(R_cap^2 - (y_tan_super - y_cap)^2);

    % Minimum distance from the centre of the cut-off disk, equal to R_cap
    % when the super-hyperbola touches the disk without crossing it
    dist_sq_super = @(x) ...
        x.^2 + ...
        (a * (1 + (x ./ b_super).^n_tune).^(1 / n_tune) - y_cap).^2;

    [~, min_dist_sq_super] = fminbnd(dist_sq_super, 0, R_cap);

    min_dist_super = sqrt(min_dist_sq_super);

    %% Generate sampled curves
    vx = linspace(-4, 4, 500);

    ang = linspace(0, 2*pi, 100);

    % Physical obstacle circle in velocity space
    obs_vx = r * cos(ang);
    obs_vy = d + r * sin(ang);

    % Standard VO cone
    vy_vo = abs(vx) * m_vo;

    % FVO time horizon cap
    cap_vx = R_cap * cos(ang);
    cap_vy = y_cap + R_cap * sin(ang);

    % Standard hyperbola, n = 2
    vy_hyperbola_n2 = a * ...
        (1 + abs(vx ./ b_n2).^2).^(1 / 2);

    % Super-hyperbola
    vy_super_hyperbola = a * ...
        (1 + abs(vx ./ b_super).^n_tune).^(1 / n_tune);

    %% Pack output
    out = struct();

    out.params.d = d;
    out.params.r = r;
    out.params.tau = tau;
    out.params.n_tune = n_tune;

    out.standardVO.theta = theta;
    out.standardVO.m_vo = m_vo;
    out.standardVO.y_cap = y_cap;
    out.standardVO.R_cap = R_cap;

    out.hyperbola.a = a;
    out.hyperbola.b_n2 = b_n2;
    out.hyperbola.x_tan_n2 = x_tan_n2;
    out.hyperbola.y_tan_n2 = y_tan_n2;

    out.superHyperbola.b_super = b_super;
    out.superHyperbola.x_tan_super = x_tan_super;
    out.superHyperbola.y_tan_super = y_tan_super;
    out.superHyperbola.min_dist_super = min_dist_super;
    out.superHyperbola.tangency_error = min_dist_super - R_cap;

    out.data.vx = vx;

    out.data.obs_vx = obs_vx;
    out.data.obs_vy = obs_vy;

    out.data.vy_vo = vy_vo;

    out.data.cap_vx = cap_vx;
    out.data.cap_vy = cap_vy;

    out.data.vy_hyperbola_n2 = vy_hyperbola_n2;
    out.data.vy_super_hyperbola = vy_super_hyperbola;

    %% Optional plot
    if doPlot
        figure;
        clf;
        hold on;
        grid on;

        fill(obs_vx, obs_vy, [1, 0.6, 0.6], ...
            'EdgeColor', 'r', ...
            'LineWidth', 1.5, ...
            'DisplayName', 'Physical Obstacle Circle');

        plot(vx, vy_vo, '--k', ...
            'LineWidth', 1.5, ...
            'DisplayName', 'Standard VO Cone');

        plot(cap_vx, cap_vy, '--b', ...
            'LineWidth', 1.5, ...
            'DisplayName', 'FVO Time Horizon \tau Cap');

        plot(vx, vy_hyperbola_n2, '-b', ...
            'LineWidth', 2, ...
            'DisplayName', 'Standard Hyperbola n = 2');

        plot([x_tan_n2, -x_tan_n2], ...
             [y_tan_n2, y_tan_n2], 'o', ...
            'MarkerEdgeColor', 'b', ...
            'MarkerFaceColor', 'y', ...
            'MarkerSize', 6, ...
            'DisplayName', 'Tangency n = 2');

        plot(vx, vy_super_hyperbola, '-g', ...
            'LineWidth', 3, ...
            'DisplayName', sprintf('Proper Super-Hyperbola n = %d', n_tune));

        plot([x_tan_super, -x_tan_super], ...
             [y_tan_super, y_tan_super], 'o', ...
            'MarkerEdgeColor', 'g', ...
            'MarkerFaceColor', 'y', ...
            'MarkerSize', 8, ...
            'DisplayName', sprintf('Tangency n = %d', n_tune));

        xlabel('Lateral Relative Velocity v_x (m/s)', ...
            'FontWeight', 'bold');

        ylabel('Forward Relative Velocity v_y (m/s)', ...
            'FontWeight', 'bold');

        title(sprintf(['Exact Tangent Super-Hyperbolic VO\n', ...
            'd = %.1f m, r = %.1f m, \\tau = %.2f s, n = %d'], ...
            d, r, tau, n_tune));

        axis equal;
        xlim([-4, 4]);
        ylim([0, d + r + 1]);

        legend('Location', 'southeast', 'FontSize', 10);
        set(gca, 'FontSize', 12);

        hold off;

        if ~isempty(savePath)
            saveas(gcf, savePath, 'epsc');
        end
    end
end

%% Local helper function
function [b, y_star] = shWidth(a, d, r, n)
%SHWIDTH Width of the narrowest super-hyperbola containing a disk.
%
% Curve:
%   y(x) = a * (1 + (x / b)^n)^(1/n), with the vertex a on the disk of
%   centre (0, d) and radius r
%
% b is the ratio beta(y) of the paper at the tangency ordinate y_star, the
% root of the tangency polynomial P(y) other than a (Lemma 1)

    if n == 2
        % The hyperbola osculates the disk at the vertex
        b = sqrt(a * r);
        y_star = a;
        return;
    end

    % P(y) deflated by its root y = a, negative at a and positive at d - r^2/d
    k = 0:(n - 1);
    deflatedTangencyPoly = @(y) ...
        (d + r - y) * y^(n - 1) + ...
        (y - d) * sum(y.^(n - 1 - k) .* a.^k);

    y_star = fzero(deflatedTangencyPoly, [a, d - r^2 / d]);

    b = a * sqrt(r^2 - (y_star - d)^2) / (y_star^n - a^n)^(1 / n);
end