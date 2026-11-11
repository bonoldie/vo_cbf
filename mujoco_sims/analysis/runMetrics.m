function m = runMetrics(run, arrivalDistance)
% Safety, efficiency and computation metrics of one run of run_sweep.py
    p = double(run.p);                  % logged samples x agents x 3
    u = double(run.u);
    tLog = double(run.t_log(:));
    starts = double(run.starts);
    targets = double(run.targets);
    nAgents = size(p, 2);

    m.completed = double(run.completed) > 0;
    m.completionTime = NaN;
    if m.completed
        m.completionTime = double(run.time(end));
    end

    % Time after which each agent stays close to its target
    distanceToTarget = vecnorm(p - reshape(targets, 1, nAgents, 3), 2, 3);
    arrival = NaN(1, nAgents);
    for a = 1:nAgents
        last = find(distanceToTarget(:, a) > arrivalDistance, 1, 'last');
        if isempty(last)
            arrival(a) = tLog(1);
        elseif last < numel(tLog)
            arrival(a) = tLog(last + 1);
        end
    end
    m.meanArrivalTime = mean(arrival);

    % Travelled distance over the straight line between start and target
    pathLength = sum(vecnorm(diff(p, 1, 1), 2, 3), 1);
    straightLength = vecnorm(targets - starts, 2, 2)';
    m.pathRatio = mean(pathLength ./ straightLength);
    m.maxPathRatio = max(pathLength ./ straightLength);

    % Safety: the obstacles do not react, so a negative gap is a contact of the agent
    gap = double(run.gap_nn);
    m.minGap = min(gap(:));
    m.collision = m.minGap < 0;
    m.contactAgents = sum(min(gap, [], 1) < 0);
    m.minH = double(min(run.h_min(:), [], 'omitnan'));

    % Control. Agents much faster than the reference speed reveal a diverging
    % nominal acceleration, rather than the CBF
    m.maxSpeed = max(vecnorm(double(run.v), 2, 3), [], 'all');
    m.divergedAgents = sum(max(vecnorm(double(run.v), 2, 3), [], 1) > 5 * double(run.ref_speed));
    accelerationNorm = vecnorm(u, 2, 3);
    m.maxAcceleration = max(accelerationNorm(:));
    m.meanEffort = mean(sum(accelerationNorm.^2, 1) * mean(diff(tLog)));
    % Velocity change imposed by the CBF on each agent, integral of |u* - u_ref|
    m.meanModulation = mean(sum(double(run.du_norm), 1) * double(run.dt));
    m.activeFraction = mean(run.n_active(:) > 0);
    m.qpFailureRate = 1 - mean(double(run.qp_solved(:)));
    m.infeasibleRate = mean(double(run.qp_infeasible(:)));

    % Computation per agent and control step
    tCommand = 1e3 * double(run.t_command(:));
    tOsqp = 1e3 * double(run.t_osqp(:));
    m.tCommandMean = mean(tCommand);
    m.tCommandMedian = median(tCommand);
    m.tCommandP95 = prctile(tCommand, 95);
    m.tOsqpMean = mean(tOsqp);

    % Largest distance of an agent from the straight line between its start and target
    direction = (targets - starts) ./ vecnorm(targets - starts, 2, 2);
    offset = p - reshape(starts, 1, nAgents, 3);
    along = sum(offset .* reshape(direction, 1, nAgents, 3), 3);
    lateral = vecnorm(offset - along .* reshape(direction, 1, nAgents, 3), 2, 3);
    m.maxLateral = max(lateral(:));
end
