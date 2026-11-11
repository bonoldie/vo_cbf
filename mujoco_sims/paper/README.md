# Paper figures: sphere crossing, degree n against speed factor k

Run from `mujoco_sims/`. Simulation logs, metrics, snapshots and figures go to
the git-ignored `data/`.

1. Simulations of the speed study (300 runs, 10 layouts), raw logs in `data/sweep/speed/`:

       python run_sweep.py speed --workers 8

2. Metrics of every run, in MATLAB, saved in `data/sweep/speed_metrics.csv`:

       analysis/analyzeSpeed.m

3. Snapshots at t = 6 s of layout 3 at k = 2.5, in `data/crossing_video/`:

       python render_crossing.py --seed 3 --n 2 4 --tau 5 --speed 2.5 --snapshots 6

4. Figures and numbers of the text, in MATLAB:

       paper/paperFigures.m

   It exports `degree_speed.pdf`, `lateral_deviation.pdf`, `crossing_n{2,4}_t6s.png`
   and `velocity_n{2,4}_t6s.pdf` to `data/sweep/figures/`.

The crossing study (degree n against horizon tau) works the same way, with
`python run_sweep.py crossing` and `analysis/analyzeCrossing.m`.