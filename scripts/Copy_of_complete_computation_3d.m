clear all; close all; clc;

% robot state
syms x_r y_r z_r vx_r vy_r vz_r real 

% acceleration inputs
syms u_x u_y u_z real;

system_state = [x_r; y_r; z_r; vx_r; vy_r; vz_r];

u = [u_x; u_y; u_z];

% Robot is just a 3D double integrator
A = [ 0 0 0 1 0 0;...
      0 0 0 0 1 0;... 
      0 0 0 0 0 1;...
      0 0 0 0 0 0;...
      0 0 0 0 0 0;...
      0 0 0 0 0 0];

F = A*system_state;

G = [ 0 0 0;...
      0 0 0;...
      0 0 0;...
      1 0 0;...
      0 1 0;...
      0 0 1];

% obstacle state
syms x_ob y_ob z_ob vx_ob vy_ob vz_ob real;

obstacle_state = [x_ob; y_ob; z_ob; vx_ob; vy_ob; vz_ob];

% velocity to check
v_rel = [vx_r - vx_ob; vy_r - vy_ob; vz_r - vz_ob];

vx_rel = v_rel(1);
vy_rel = v_rel(2);
vz_rel = v_rel(3);

% super-hyperbola parameters

syms R tau n real;

delta_p = obstacle_state(1:3) - system_state(1:3);

d = sqrt(delta_p.'*delta_p); % distance between the obstacle and the robot


%% Obstacle-aligned local frame

eps_d = 1e-6;
eps_v = 1e-6;

d_squared = simplify(delta_p.' * delta_p);
d_safe_squared = d_squared + eps_d^2;
d_safe = sqrt(d_safe_squared);

% Smooth radial direction
e_y = simplify(delta_p / d_safe);

% Radial relative velocity
v_radial = simplify(v_rel.' * e_y);

% Tangential projection matrix
P_tangential = simplify( ...
    eye(3) - (delta_p * delta_p.') / d_safe_squared ...
);

% Tangential relative velocity
v_tangential_vector = simplify(P_tangential * v_rel);

% Smooth norm
v_tangential = simplify(sqrt( ...
    v_tangential_vector.' * v_tangential_vector + eps_v^2 ...
));

%% SH parameters

a = (d - R) / tau;

%% Gradients

syms b(x_r, y_r, z_r, vx_r, vy_r, vz_r);

h = a*(1.0 + (v_tangential / b)^n)^(1.0 / n) - v_radial;

grad_h = gradient(h, [x_r, y_r, z_r, vx_r, vy_r, vz_r]);
% Symfun to array of sym
grad_h = grad_h(x_r, y_r, z_r, vx_r, vy_r, vz_r);

% Now grad_h is the symbolic gradient that we need, it contains the partial
% derivatives of b wrt the system state

syms vy_tan vy_tan_func(x_r, y_r, z_r, vx_r, vy_r, vz_r);

% From this we will compute the partial derivatives of vy_tan (P in the
% paper)
vy_tan_eq = d*(vy_tan^n) - (d^2 - R^2) * vy_tan ^ (n - 1) - (a^n) * vy_tan + d*(a^n);

vx_tan = sqrt(R^2 - (vy_tan_func - d)^2); 

b_val = (a * vx_tan) / (vy_tan_func^n - a^n)^(1/n);

% The following gradient of b is built with the partial derivative of vy_tan
% wrt the system state (i.e. [x_r, y_r, z_r, vx_r, vy_r, vz_r])
grad_b_val = gradient(b_val,  [x_r, y_r, z_r, vx_r, vy_r, vz_r]);
grad_b_val = grad_b_val(x_r, y_r, z_r, vx_r, vy_r, vz_r);
% grad_b_val = simplify(grad_b_val);

dvy_tan_eq_dvy_tan = diff(vy_tan_eq, vy_tan);

dvy_tan_dx_r = - diff(vy_tan_eq, x_r)/dvy_tan_eq_dvy_tan;
dvy_tan_dy_r = - diff(vy_tan_eq, y_r)/dvy_tan_eq_dvy_tan;
dvy_tan_dz_r = - diff(vy_tan_eq, z_r)/dvy_tan_eq_dvy_tan;
dvy_tan_dvx_r = - diff(vy_tan_eq, vx_r)/dvy_tan_eq_dvy_tan;
dvy_tan_dvy_r = - diff(vy_tan_eq, vy_r)/dvy_tan_eq_dvy_tan;
dvy_tan_dvz_r = - diff(vy_tan_eq, vz_r)/dvy_tan_eq_dvy_tan;

grad_b_val = subs(grad_b_val,  ...
    gradient(vy_tan_func, [x_r, y_r, z_r, vx_r, vy_r, vz_r]),[ ...
    dvy_tan_dx_r; dvy_tan_dy_r;dvy_tan_dz_r; dvy_tan_dvx_r; dvy_tan_dvy_r; dvy_tan_dvz_r...
]);

% grad_b_val = simplify(grad_b_val);

% substitute back to the CBF gradient
grad_h = subs(grad_h, gradient(b, [x_r, y_r, z_r, vx_r, vy_r, vz_r]), grad_b_val);
% grad_h = simplify(grad_h);

syms b_computed;

final_grad_h = subs(grad_h, [ b(x_r, y_r, z_r, vx_r, vy_r, vz_r), vy_tan_func(x_r, y_r, z_r, vx_r, vy_r, vz_r)], [b_computed, vy_tan]);
% final_grad_h = simplify(final_grad_h)

% fprintMatPy('CBFGrad3D', {'x_r', 'y_r' , 'z_r', 'vx_r', 'vy_r', 'vz_r', 'x_ob', 'y_ob', 'z_ob','vx_ob', 'vy_ob', 'vz_ob', 'b_computed', 'vy_tan', 'R', 'tau', 'n'}, final_grad_h);

%% Grad validation
% barrier_val = final_grad_h' * F + final_grad_h' * G * u + 10 * U_cbf; 

%% Plotting possible configurations

configurations = [
    struct( ...
        "p_robot",    [0.0; 0.0], ...
        "v_robot",    [1.0; 0.3], ...
        "p_obstacle", [2.0; 1.0], ...
        "v_obstacle", [0.2; 0.6], ...
        "name",       "Diagonal obstacle" ...
    )

    struct( ...
        "p_robot",    [0.0; 0.0], ...
        "v_robot",    [1.0; 0.0], ...
        "p_obstacle", [2.0; 0.0], ...
        "v_obstacle", [-0.4; 0.0], ...
        "name",       "Head-on motion" ...
    )

    struct( ...
        "p_robot",    [0.0; 0.0], ...
        "v_robot",    [0.8; 0.8], ...
        "p_obstacle", [0.0; 2.0], ...
        "v_obstacle", [0.3; 0.0], ...
        "name",       "Obstacle above robot" ...
    )

    struct( ...
        "p_robot",    [-1.0; -0.5], ...
        "v_robot",    [0.4; 1.0], ...
        "p_obstacle", [1.5; 1.0], ...
        "v_obstacle", [0.9; 0.1], ...
        "name",       "Crossing motion" ...
    )
];

robot_radius = 0.20;
obstacle_radius = 0.35;

velocity_scale = 0.8;
axis_scale = 0.75;

figure("Name", "Global and obstacle-aligned relative velocities");
tiledlayout(2, 2, "TileSpacing", "compact", "Padding", "compact");

for k = 1:numel(configurations)

    cfg = configurations(k);

    p_r = cfg.p_robot;
    p_o = cfg.p_obstacle;

    v_r = cfg.v_robot;
    v_o = cfg.v_obstacle;

    % Relative position and distance
    delta_p_num = p_o - p_r;
    d_num = norm(delta_p_num);

    if d_num < 1e-10
        warning("Configuration %d skipped: robot and obstacle overlap.", k);
        continue;
    end

    % Local basis
    e_y_num = delta_p_num / d_num;

    e_x_num = [
         e_y_num(2);
        -e_y_num(1)
    ];

    R_wl = [
        e_x_num.';
        e_y_num.'
    ];

    R_lw = R_wl.';

    % Global relative velocity
    v_rel_global = v_r - v_o;

    % Relative velocity expressed in the local frame
    v_rel_local_num = R_wl * v_rel_global;

    vx_local = v_rel_local_num(1);
    vy_local = v_rel_local_num(2);

    % Local components expressed back in world coordinates.
    % These are useful for plotting both components in the global plot.
    v_tangential_world = R_lw * [vx_local; 0];
    v_radial_world     = R_lw * [0; vy_local];

    % Reconstruction check
    v_rel_check = v_tangential_world + v_radial_world;
    reconstruction_error = norm(v_rel_global - v_rel_check);

    fprintf("\nConfiguration %d: %s\n", k, cfg.name);
    fprintf("  Global relative velocity: [%.4f, %.4f]\n", ...
        v_rel_global(1), v_rel_global(2));
    fprintf("  Local relative velocity:  [%.4f, %.4f]\n", ...
        vx_local, vy_local);
    fprintf("  Tangential component:      %.4f\n", vx_local);
    fprintf("  Radial component:          %.4f\n", vy_local);
    fprintf("  Reconstruction error:      %.3e\n", ...
        reconstruction_error);

    nexttile;
    hold on;
    grid on;
    axis equal;

    %% Draw robot and obstacle

    drawCircle(p_r, robot_radius, [0.25, 0.55, 0.95]);
    drawCircle(p_o, obstacle_radius, [0.95, 0.35, 0.25]);

    plot(p_r(1), p_r(2), "ko", ...
        "MarkerFaceColor", "k", ...
        "MarkerSize", 4);

    plot(p_o(1), p_o(2), "ko", ...
        "MarkerFaceColor", "k", ...
        "MarkerSize", 4);

    text(p_r(1), p_r(2) - robot_radius - 0.12, ...
        "robot", ...
        "HorizontalAlignment", "center");

    text(p_o(1), p_o(2) - obstacle_radius - 0.12, ...
        "obstacle", ...
        "HorizontalAlignment", "center");

    %% Draw robot-to-obstacle line

    plot( ...
        [p_r(1), p_o(1)], ...
        [p_r(2), p_o(2)], ...
        "k:", ...
        "LineWidth", 1.2 ...
    );

    %% Draw global robot and obstacle velocities

    quiver( ...
        p_r(1), p_r(2), ...
        velocity_scale * v_r(1), ...
        velocity_scale * v_r(2), ...
        0, ...
        "LineWidth", 1.5, ...
        "DisplayName", "v_r" ...
    );

    quiver( ...
        p_o(1), p_o(2), ...
        velocity_scale * v_o(1), ...
        velocity_scale * v_o(2), ...
        0, ...
        "LineWidth", 1.5, ...
        "DisplayName", "v_o" ...
    );

    %% Draw global relative velocity at the robot

    quiver( ...
        p_r(1), p_r(2), ...
        velocity_scale * v_rel_global(1), ...
        velocity_scale * v_rel_global(2), ...
        0, ...
        "k", ...
        "LineWidth", 2.2, ...
        "MaxHeadSize", 0.5, ...
        "DisplayName", "v_{rel}" ...
    );

    %% Draw local coordinate frame at the robot

    quiver( ...
        p_r(1), p_r(2), ...
        axis_scale * e_x_num(1), ...
        axis_scale * e_x_num(2), ...
        0, ...
        "--", ...
        "LineWidth", 1.5, ...
        "MaxHeadSize", 0.5, ...
        "DisplayName", "local x" ...
    );

    quiver( ...
        p_r(1), p_r(2), ...
        axis_scale * e_y_num(1), ...
        axis_scale * e_y_num(2), ...
        0, ...
        "--", ...
        "LineWidth", 1.5, ...
        "MaxHeadSize", 0.5, ...
        "DisplayName", "local y" ...
    );

    text( ...
        p_r(1) + axis_scale * e_x_num(1), ...
        p_r(2) + axis_scale * e_x_num(2), ...
        "  x_L" ...
    );

    text( ...
        p_r(1) + axis_scale * e_y_num(1), ...
        p_r(2) + axis_scale * e_y_num(2), ...
        "  y_L" ...
    );

    %% Draw decomposition of relative velocity

    component_origin = p_r;

    % Tangential/local-x component
    quiver( ...
        component_origin(1), ...
        component_origin(2), ...
        velocity_scale * v_tangential_world(1), ...
        velocity_scale * v_tangential_world(2), ...
        0, ...
        "LineWidth", 2, ...
        "MaxHeadSize", 0.5, ...
        "DisplayName", "v_{rel,x_L}" ...
    );

    tangential_endpoint = component_origin + ...
        velocity_scale * v_tangential_world;

    % Radial/local-y component, starting at the end of x component
    quiver( ...
        tangential_endpoint(1), ...
        tangential_endpoint(2), ...
        velocity_scale * v_radial_world(1), ...
        velocity_scale * v_radial_world(2), ...
        0, ...
        "LineWidth", 2, ...
        "MaxHeadSize", 0.5, ...
        "DisplayName", "v_{rel,y_L}" ...
    );

    %% Labels and limits

    title({
        cfg.name
        sprintf( ...
            "v_{rel}^{W} = [%.2f, %.2f],  v_{rel}^{L} = [%.2f, %.2f]", ...
            v_rel_global(1), ...
            v_rel_global(2), ...
            vx_local, ...
            vy_local ...
        )
    });

    xlabel("World x");
    ylabel("World y");

    all_points = [
        p_r, ...
        p_o, ...
        p_r + velocity_scale * v_r, ...
        p_o + velocity_scale * v_o, ...
        p_r + velocity_scale * v_rel_global
    ];

    margin = 1.0;

    xlim([
        min(all_points(1, :)) - margin, ...
        max(all_points(1, :)) + margin
    ]);

    ylim([
        min(all_points(2, :)) - margin, ...
        max(all_points(2, :)) + margin
    ]);
end


%% Local function

function drawCircle(center, radius, faceColor)

    theta = linspace(0, 2*pi, 100);

    x = center(1) + radius * cos(theta);
    y = center(2) + radius * sin(theta);

    fill( ...
        x, ...
        y, ...
        faceColor, ...
        "FaceAlpha", 0.35, ...
        "EdgeColor", faceColor, ...
        "LineWidth", 1.5 ...
    );
end

% % these two symbols are the tangency coordinates
% syms vy_t(x_r, y_r) dvy_t_dx_r dvy_t_dy_r real;
% 
% % coord at which the curve is tangent to the circle
% vx_t = sqrt(R^2 - (vy_t - d)^2);
% 
% b = (a * vx_t) / (vy_t^n - a^n)^(1/n);
% 
% db_dx_r = gradient(b, [x_r]);
% 
% % equation to find dvy_t/dx_r and dvy_t/dy_r (from the tangency condition)
% eq_vy_t = d * vy_t^n -  a^n * vy_t - (d^2 - R^2) * vy_t ^ (n-1) + d * a^n == 0;
% eq_vy_t = simplify(eq_vy_t);
% 
% grad_eq_vy_t_x_r = simplify(gradient(lhs(eq_vy_t), [x_r]));
% grad_eq_vy_t_y_r = simplify(gradient(lhs(eq_vy_t), [y_r]));
% 
% grad_eq_vy_t_x_r = subs(grad_eq_vy_t_x_r,diff(vy_t(x_r, y_r), x_r),dvy_t_dx_r);
% 
% grad_eq_vy_t_y_r = subs(grad_eq_vy_t_y_r, diff(vy_t(x_r, y_r), y_r),dvy_t_dy_r);
% 
% % dvy_t/dx_r and dvy_t/dy_r
% dvy_t_dx_r_sol = solve(grad_eq_vy_t_x_r, dvy_t_dx_r, ReturnConditions=true);
% dvy_t_dy_r_sol = solve(grad_eq_vy_t_y_r, dvy_t_dy_r, ReturnConditions=true);
% 
% % Now we can subs into db_dx_r
% db_dx_r = simplify(subs(db_dx_r, diff(vy_t(x_r, y_r), x_r), dvy_t_dx_r_sol.dvy_t_dx_r));
% 
% % Candidate CBF
% h = a * (1 + (vx_rel / b)^n)^(1/n) - vy_rel;
% 
% % CBF gradient
% 
% dh_dx_r = simplify(gradient(h, [x_r]));
% dh_dx_r = simplify(subs(dh_dx_r,[diff(vy_t(x_r, y_r), x_r)], [dvy_t_dx_r_sol.dvy_t_dx_r]));
% 
% dh_dy_r = simplify(gradient(h, [y_r]));
% dh_dy_r = simplify(subs(dh_dy_r,[diff(vy_t(x_r, y_r), y_r)], [dvy_t_dy_r_sol.dvy_t_dy_r]));
% 
% dh_dvx_r = simplify(gradient(h, [vx_r]));
% dh_dvy_r = simplify(gradient(h, [vy_r]));
% 
% grad_h_T = [dh_dx_r, dh_dy_r, dh_dvx_r, dh_dvy_r];
% 
% % Computing gradient condition on CBF
% syms alpha real;
% 
% % Warning! The inputs have to be rotated to align with the relative
% % velocity frame
% 
% p_robot = system_state(1:2);
% p_obstacle = obstacle_state(1:2);
% 
% e_y = p_obstacle - p_robot;
% e_y = e_y / (norm(e_y) + 1e-9);
% e_x = [e_y(2); -e_y(1)];
% 
% R_world_to_local = [e_x, e_y];
% 
% u_aligned =  R_world_to_local * u;
% 
% U_cbf = grad_h_T*F + grad_h_T*G*u_aligned + alpha * h;
% U_cbf = simplify(U_cbf);
% 
% %% Example
% 
% robot_state1 = [0.0; 0.5; 1.0; 0.03];
% robot_r = 0.5;
% 
% obstacle_state1 = [2.0; 0.0; 0.3; -0.19];
% obstacle_r = 0.5;
% 
% d_val = double(subs(d, [x_r; y_r; x_ob; y_ob], [robot_state1(1);robot_state1(2);obstacle_state1(1);obstacle_state1(2)]));
% R_val = robot_r + obstacle_r;
% tau_val = 1.25;
% n_val = 6;
% u_val = [1.0; 5.0];
% 
% sh_out = SHVO(d_val, R_val, tau_val, n_val);
% 
% alpha_val = 2;
% 
% U_cbf1 = subs(U_cbf, [vy_t], [sh_out.superHyperbola.y_tan_super]);
% U_cbf2 = subs(U_cbf1, [tau; R; alpha; n], [tau_val;R_val; alpha_val;n_val]);
% U_cbf3 = subs(U_cbf2, [x_r; y_r; vx_r; vy_r; x_ob; y_ob; vx_ob; vy_ob], [robot_state1; obstacle_state1]);
% 
% U_cbf_final = subs(U_cbf3, u, u_val);
