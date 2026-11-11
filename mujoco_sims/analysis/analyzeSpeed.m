% Analysis of the speed study of run_sweep.py: degree n against the speed factor k
% of the agent and of the obstacles, on the layouts of the crossing study at a
% fixed horizon tau.
%
% Loads the raw logs in ../data/sweep/speed/, computes the metrics of every run
% and saves them in ../data/sweep/speed_metrics.csv (read by
% ../paper/paperFigures.m), with their mean and standard deviation over the
% layouts in speed_summary.csv. Runs that are not simulated yet are skipped.
%
% Rebuilt on 2026-11-11 from the recovered speed_metrics.csv, the original was lost.
clear; close all; clc;

%% Settings
% Agent at its target below this distance [m]: twice ARRIVAL_DISTANCE of
% run_sweep.py, since the positions are only logged every LOG_PERIOD
arrivalDistance = 0.02;

sweepFolder = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'sweep');

%% Metrics
study = loadStudy(fullfile(sweepFolder, 'speed'), arrivalDistance);
if isempty(study.metrics)
    error('No run of the speed study yet, run python run_sweep.py speed');
end
metrics = study.metrics;

% With speed factor k every speed is k times larger, so the metrics are also
% compared without units: time over the straight crossing at the reference
% speed, CBF intervention over the reference speed and extra path length
straightTime = cellfun(@(run) norm(double(run.targets) - double(run.starts)) / double(run.ref_speed), study.runs)';
referenceSpeed = cellfun(@(run) double(run.ref_speed), study.runs)';
metrics.timeRatio = metrics.completionTime ./ straightTime;
metrics.relativeModulation = metrics.meanModulation ./ referenceSpeed;
metrics.pathExcess = metrics.pathRatio - 1;
writetable(metrics, fullfile(sweepFolder, 'speed_metrics.csv'));

summary = groupsummary(metrics, {'speed', 'n'}, {'mean', 'std'}, ...
    {'completionTime', 'timeRatio', 'pathExcess', 'minGap', 'collision', 'minH', 'meanModulation', ...
     'relativeModulation', 'maxLateral', 'qpFailureRate', 'infeasibleRate', 'tCommandMean'});
writetable(summary, fullfile(sweepFolder, 'speed_summary.csv'));
fprintf('saved speed_metrics.csv and speed_summary.csv in %s\n', sweepFolder);

disp(summary(:, {'speed', 'n', 'GroupCount', 'mean_completionTime', 'mean_pathExcess', 'mean_minGap', ...
    'mean_maxLateral', 'mean_meanModulation'}));
