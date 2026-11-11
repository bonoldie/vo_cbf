% Figures of the paper on the degree n in the sphere crossing of run_sweep.py,
% exported in ../data/sweep/figures/:
%
%   degree_speed.pdf          completion time, extra path length, maximum lateral
%                             deviation and CBF intervention of the speed study, in %
%                             of the hyperbola (n = 2), for every speed factor k
%   lateral_deviation.pdf     distance from the straight path on one layout, for a
%                             few speed factors
%   crossing_n<n>_t<t>s.png   3D view of render_crossing.py at time t
%   velocity_n<n>_t<t>s.pdf   velocity space of the red obstacle at the same time
%
% Also prints the numbers of the text. Reads, in the git-ignored ../data/:
%   sweep/speed_metrics.csv   metrics of the speed study (../analysis/analyzeSpeed.m)
%   sweep/speed/              raw logs of the lateral layout (python run_sweep.py speed)
%   crossing_video/           snapshots of
%       python render_crossing.py --seed 3 --n 2 4 --tau 5 --speed 2.5 --snapshots 6
%
% Rebuilt on 2026-11-11 from the figures of the paper, the original was lost.
clear; close all; clc;

%% Settings
exportFigures = true;
fontSize = 8;
lateralSeed = 3;                % layout of the lateral deviation and of the snapshots
lateralSpeeds = [1, 2.5, 4];    % speed factors k of the lateral deviation
snapshotDegrees = [2, 4];
snapshotSpeed = 2.5;
snapshotTime = 6;               % [s]

% Colours of the collision sets in the velocity space, as in render_crossing.py
setColors = containers.Map({2, 4, 6, 8, 12}, ...
    {[220, 100, 220] / 255, [40, 200, 220] / 255, [90, 200, 90] / 255, [255, 150, 60] / 255, [200, 200, 200] / 255});
relativeColor = [25, 204, 25] / 255;

dataFolder = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
sweepFolder = fullfile(dataFolder, 'sweep');
videoFolder = fullfile(dataFolder, 'crossing_video');
figureFolder = fullfile(sweepFolder, 'figures');
if exportFigures && ~exist(figureFolder, 'dir')
    mkdir(figureFolder);
end

metrics = readtable(fullfile(sweepFolder, 'speed_metrics.csv'), 'TextType', 'string');
degrees = unique(metrics.n)';
speeds = unique(metrics.speed)';

%% Numbers of the text
% Averages over the layouts, then ratios of the averages
meanOf = @(column, n, k) mean(metrics.(column)(metrics.n == n & metrics.speed == k));

fprintf('Reduction of n = 4 with respect to n = 2, per speed factor k:\n');
for column = ["completionTime", "maxLateral", "meanModulation"]
    reduction = arrayfun(@(k) 100 * (1 - meanOf(column, 4, k) / meanOf(column, 2, k)), speeds);
    fprintf('  %-16s %4.1f-%4.1f %%  (%s)\n', column, min(reduction), max(reduction), ...
        strjoin(compose('k = %g: %.1f', [speeds; reduction]'), ', '));
end

gapHyperbola = arrayfun(@(k) 100 * meanOf('minGap', 2, k), speeds);
gapOthers = arrayfun(@(k) 100 * mean(metrics.minGap(metrics.n >= 4 & metrics.speed == k)), speeds);
fprintf('Average minimum gap: n = 2 %.1f-%.1f cm, n >= 4 %.1f-%.1f cm\n', ...
    min(gapHyperbola), max(gapHyperbola), min(gapOthers), max(gapOthers));

stepTime = arrayfun(@(n) mean(metrics.tCommandMean(metrics.n == n)), degrees);
fprintf('Control step: %.1f-%.1f ms (per degree), OSQP %.2f ms\n', ...
    min(stepTime), max(stepTime), mean(metrics.tOsqpMean));
fprintf('Completed runs: %d of %d, collisions: %d\n', sum(metrics.completed), height(metrics), sum(metrics.collision));

%% Metrics against the degree, in % of the hyperbola
panels = {
    'completionTime', 'Completion time'
    'pathExcess', 'Extra path length'
    'maxLateral', 'Max. lateral deviation'
    'meanModulation', 'CBF intervention'
};
speedColors = parula(numel(speeds) + 1);

fig = figure('Units', 'centimeters', 'Position', [2, 2, 8.6, 8.0], 'Color', 'w');
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
lines_ = gobjects(1, numel(speeds));
for p = 1:size(panels, 1)
    ax = nexttile(layout);
    hold(ax, 'on');
    for i = 1:numel(speeds)
        values = arrayfun(@(n) meanOf(panels{p, 1}, n, speeds(i)), degrees);
        values = 100 * values / values(degrees == 2);
        lines_(i) = plot(ax, degrees, values, '-o', 'Color', speedColors(i, :), 'LineWidth', 1.2, ...
            'MarkerSize', 2.5, 'MarkerFaceColor', speedColors(i, :), 'DisplayName', sprintf('$k = %g$', speeds(i)));
        % Best degree for this speed factor
        [~, best] = min(values);
        plot(ax, degrees(best), values(best), 'p', 'MarkerSize', 8, 'MarkerFaceColor', speedColors(i, :), ...
            'MarkerEdgeColor', 0.2 * [1, 1, 1], 'HandleVisibility', 'off');
    end
    title(ax, panels{p, 2}, 'Interpreter', 'latex', 'FontWeight', 'normal', 'FontSize', fontSize + 1);
    xticks(ax, degrees);
    xlim(ax, [1.5, 12.5]);
    styleAxes(ax, fontSize);
end
ylabel(layout, '\% of the hyperbola ($n = 2$)', 'Interpreter', 'latex', 'FontSize', fontSize + 1);
lgd = legend(lines_, 'Interpreter', 'latex', 'FontSize', fontSize - 1, 'Orientation', 'horizontal', ...
    'NumColumns', numel(speeds));
lgd.ItemTokenSize = [7, 6];
lgd.Layout.Tile = 'north';
exportFigure(fig, fullfile(figureFolder, 'degree_speed.pdf'), exportFigures);

%% Distance from the straight path on one layout
degreeColors = lines(numel(degrees));

fig = figure('Units', 'centimeters', 'Position', [2, 2, 8.6, 8.0], 'Color', 'w');
layout = tiledlayout(fig, numel(lateralSpeeds), 1, 'TileSpacing', 'compact', 'Padding', 'compact');
lines_ = gobjects(1, numel(degrees));
for i = 1:numel(lateralSpeeds)
    ax = nexttile(layout);
    hold(ax, 'on');
    for j = 1:numel(degrees)
        row = metrics.seed == lateralSeed & metrics.speed == lateralSpeeds(i) & metrics.n == degrees(j);
        runFile = fullfile(sweepFolder, 'speed', metrics.run_id(row) + ".mat");
        if ~isfile(runFile)
            error('%s not simulated yet, run python run_sweep.py speed', runFile);
        end
        run = load(runFile, 'p');
        position = double(run.p);
        lines_(j) = plot(ax, position(:, 1), vecnorm(position(:, 2:3), 2, 2), 'Color', degreeColors(j, :), ...
            'LineWidth', 1.2, 'DisplayName', sprintf('$n = %d$', degrees(j)));
    end
    text(ax, 0.02, 0.9, sprintf('$k = %g$', lateralSpeeds(i)), 'Units', 'normalized', ...
        'Interpreter', 'latex', 'FontSize', fontSize, 'VerticalAlignment', 'top');
    xlim(ax, [-2, 2]);
    ylim(ax, [0, 1.2]);
    yticks(ax, 0:0.5:1);
    if i < numel(lateralSpeeds)
        ax.XTickLabel = {};
    end
    styleAxes(ax, fontSize);
end
xlabel(layout, '$x$ [m]', 'Interpreter', 'latex', 'FontSize', fontSize + 1);
ylabel(layout, 'Distance from the straight path [m]', 'Interpreter', 'latex', 'FontSize', fontSize + 1);
lgd = legend(lines_, 'Interpreter', 'latex', 'FontSize', fontSize - 1, 'Orientation', 'horizontal', ...
    'NumColumns', numel(degrees));
lgd.ItemTokenSize = [12, 6];
lgd.Layout.Tile = 'north';
exportFigure(fig, fullfile(figureFolder, 'lateral_deviation.pdf'), exportFigures);

%% Snapshots of the crossing: 3D view and velocity space of the red obstacle
for n = snapshotDegrees
    pattern = sprintf('crossing_M060_n%02d_tau5.00_*_seed%d_speed%g_t%05.2fs', n, lateralSeed, snapshotSpeed, snapshotTime);
    snapshot = dir(fullfile(videoFolder, [pattern, '.mat']));
    if isempty(snapshot)
        error('No snapshot %s, run python render_crossing.py --seed %d --n %s --tau 5 --speed %g --snapshots %g', ...
            pattern, lateralSeed, num2str(snapshotDegrees), snapshotSpeed, snapshotTime);
    end
    stem = fullfile(snapshot(1).folder, erase(snapshot(1).name, '.mat'));
    space = load([stem, '.mat']);
    if space.selected < 0 || ~isfield(space, 'a')
        error('No obstacle, or overlapping, in the velocity space of %s', stem);
    end

    if exportFigures
        copyfile([stem, '.png'], fullfile(figureFolder, sprintf('crossing_n%d_t%gs.png', n, snapshotTime)));
    end

    % Same scale on both velocities, with room for the legend below the vertex
    % of the collision sets
    fig = figure('Units', 'centimeters', 'Position', [2, 2, 5.0, 6.4], 'Color', 'w');
    ax = axes(fig);
    hold(ax, 'on');

    % Boundaries v_r = a (1 + (|v_t| / b)^n)^(1 / n), the collision set of this case filled
    vT = linspace(-1.5, 1.5, 1201);
    own = space.a * (1 + (abs(vT) / space.b) .^ double(space.n)) .^ (1 / double(space.n));
    other = space.a_other * (1 + (abs(vT) / space.b_other) .^ double(space.other_n)) .^ (1 / double(space.other_n));
    fill(ax, [vT, fliplr(vT)], [own, 10 * ones(size(vT))], setColors(double(space.n)), ...
        'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    % FVO: the cut-off disk D(p / tau, R / tau) and the cone tangent to it
    angle = linspace(0, 2 * pi, 200);
    plot(ax, space.R / space.tau * cos(angle), space.ell / space.tau + space.R / space.tau * sin(angle), ...
        'k', 'LineWidth', 0.6, 'HandleVisibility', 'off');
    sinTheta = space.R / space.ell;
    cosTheta = sqrt(1 - sinTheta ^ 2);
    tangency = sqrt(space.ell ^ 2 - space.R ^ 2) / space.tau;
    for side = [-1, 1]
        fvo = plot(ax, side * sinTheta * [tangency, 10], cosTheta * [tangency, 10], 'k', 'LineWidth', 0.6);
    end

    otherLine = plot(ax, vT, other, 'Color', setColors(double(space.other_n)), 'LineWidth', 0.8);
    ownLine = plot(ax, vT, own, 'Color', setColors(double(space.n)), 'LineWidth', 2.0);
    relative = plot(ax, space.relative(1), space.relative(2), 'o', 'MarkerSize', 4.5, ...
        'MarkerFaceColor', relativeColor, 'MarkerEdgeColor', 'k');
    preferred = plot(ax, space.preferred(1), space.preferred(2), 'o', 'MarkerSize', 4.5, ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');

    [~, order] = sort(double([space.n, space.other_n]));
    sets = [ownLine, otherLine];
    labels = compose('$n = %d$', sort(double([space.n, space.other_n])));
    lgd = legend(ax, [sets(order), fvo, relative, preferred], ...
        [labels, {'FVO', '$v_{AB_j}$', '$v_{\mathrm{pref}} - v_{B_j}$'}], ...
        'Interpreter', 'latex', 'FontSize', fontSize - 2, 'NumColumns', 2, 'Location', 'south');
    lgd.ItemTokenSize = [10, 6];

    xlim(ax, [-0.6, 0.6]);
    ylim(ax, [-0.35, 1.05]);
    daspect(ax, [1, 1, 1]);
    xticks(ax, -0.5:0.5:0.5);
    yticks(ax, 0:0.5:1);
    xlabel(ax, '$v_t$ [m/s]', 'Interpreter', 'latex');
    ylabel(ax, '$v_r$ [m/s]', 'Interpreter', 'latex');
    styleAxes(ax, fontSize);
    exportFigure(fig, fullfile(figureFolder, sprintf('velocity_n%d_t%gs.pdf', n, snapshotTime)), exportFigures);
end

%% Functions
function styleAxes(ax, fontSize)
    set(ax, 'TickLabelInterpreter', 'latex', 'FontSize', fontSize, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
        'GridAlpha', 0.15, 'LineWidth', 0.5, 'Layer', 'top');
end

function exportFigure(fig, file, exportFigures)
    if exportFigures
        exportgraphics(fig, file, 'ContentType', 'vector');
        fprintf('saved %s\n', file);
    end
end
