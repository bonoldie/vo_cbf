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

    if tau < 1.0
        error('Tau must be >= 1.0 for the hyperbola to wrap the physical obstacle.');
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

    %% Analytical tangent hyperbola, n = 2
    a = (d - r) / tau;

    D_term = d^2 - r^2 - a^2;
    inner_term = max(0, D_term^2 - 4 * a^2 * r^2);

    b_n2 = sqrt(0.5 * (D_term - sqrt(inner_term)));

    y_tan_n2 = d / (1 + (b_n2 / a)^2);
    x_tan_n2 = b_n2 * sqrt((y_tan_n2 / a)^2 - 1);

    %% Numerical tangent super-hyperbola
    options = optimset( ...
        'Display', 'off', ...
        'TolX', 1e-8);

    err_func = @(b_guess) getTangencyError(b_guess, a, n_tune, d, r);

    b_super = fzero(err_func, b_n2, options);

    dist_sq_super = @(x) ...
        x.^2 + ...
        (a * (1 + (x ./ b_super).^n_tune).^(1 / n_tune) - d).^2;

    [x_tan_super, min_dist_sq_super] = fminbnd(dist_sq_super, 0, r);

    y_tan_super = a * ...
        (1 + (x_tan_super ./ b_super).^n_tune).^(1 / n_tune);

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
    out.superHyperbola.tangency_error = min_dist_super - r;

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
function err = getTangencyError(b, a, n, d, r)
%GETTANGENCYERROR Distance error between curve and obstacle circle.
%
% Curve:
%   y(x) = a * (1 + (x / b)^n)^(1/n)
%
% Tangency condition:
%   min_x distance((x, y(x)), (0, d)) = r

    if b <= 0 || ~isfinite(b)
        err = Inf;
        return;
    end

    dist_sq = @(x) ...
        x.^2 + ...
        (a * (1 + (x ./ b).^n).^(1 / n) - d).^2;

    [~, min_dist_sq] = fminbnd(dist_sq, 0, r);

    err = sqrt(min_dist_sq) - r;
end