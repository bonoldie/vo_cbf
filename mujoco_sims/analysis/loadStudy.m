function study = loadStudy(folder, arrivalDistance)
% Loads the runs of the manifest of run_sweep.py that are already simulated and
% computes their metrics, one row per run after the parameters of the manifest
    study.runs = {};
    study.metrics = table();

    manifestFile = fullfile(folder, 'manifest.csv');
    if ~exist(manifestFile, 'file')
        return;
    end

    opts = detectImportOptions(manifestFile);
    opts = setvartype(opts, 'run_id', 'char');
    manifest = readtable(manifestFile, opts);

    rows = {};
    for k = 1:height(manifest)
        runFile = fullfile(folder, [manifest.run_id{k}, '.mat']);
        if ~exist(runFile, 'file')
            continue;
        end

        run = load(runFile);
        study.runs{end + 1} = run;
        rows{end + 1} = [manifest(k, :), struct2table(runMetrics(run, arrivalDistance))]; %#ok<AGROW>
    end

    if ~isempty(rows)
        study.metrics = vertcat(rows{:});
    end
    fprintf('%s: %d of %d runs available\n', folder, numel(rows), height(manifest));
end
