% Analysis of the crossing study of run_sweep.py: degree n and horizon tau of the
% super-hyperbola, M = 60 obstacles, over several layouts (seeds).
%
% Loads the raw logs in ../data/sweep/crossing/, computes the metrics of every run
% and saves them in ../data/sweep/crossing_metrics.csv, with their mean and
% standard deviation over the layouts in crossing_summary.csv. Runs that are not
% simulated yet are skipped.
%
% Rebuilt on 2026-11-11 from the recovered crossing_metrics.csv, the original was lost.
clear; close all; clc;

%% Settings
% Agent at its target below this distance [m]: twice ARRIVAL_DISTANCE of
% run_sweep.py, since the positions are only logged every LOG_PERIOD
arrivalDistance = 0.02;

sweepFolder = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'sweep');

%% Metrics
study = loadStudy(fullfile(sweepFolder, 'crossing'), arrivalDistance);
if isempty(study.metrics)
    error('No run of the crossing study yet, run python run_sweep.py crossing');
end
% Time of the first control step with an active CBF constraint, NaN if none
study.metrics.firstActiveTime = cellfun(@firstActiveTime, study.runs)';
writetable(study.metrics, fullfile(sweepFolder, 'crossing_metrics.csv'));

summary = groupsummary(study.metrics, {'n', 'tau'}, {'mean', 'std'}, ...
    {'completionTime', 'pathRatio', 'minGap', 'collision', 'minH', 'meanModulation', 'maxLateral', ...
     'qpFailureRate', 'infeasibleRate', 'tCommandMean'});
writetable(summary, fullfile(sweepFolder, 'crossing_summary.csv'));
fprintf('saved crossing_metrics.csv and crossing_summary.csv in %s\n', sweepFolder);

disp(summary(:, {'n', 'tau', 'GroupCount', 'mean_completionTime', 'mean_minGap', 'mean_collision', ...
    'mean_maxLateral', 'mean_meanModulation'}));

%% Functions
function t = firstActiveTime(run)
    first = find(run.n_active(:) > 0, 1);
    t = NaN;
    if ~isempty(first)
        t = double(run.time(first));
    end
end
