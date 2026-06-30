import numpy as np
import matplotlib.pyplot as plt
from utils.fsh import FSH 

# Parameters 

r_robot = 0.5
r_obstacle = 0.5
r = r_robot + r_obstacle
tau = 1.25
n = 6

obstacle_pos = np.array([0, 2])
robot_pos = np.array([0, 0])

d = np.linalg.norm(obstacle_pos - robot_pos)

# ----------------------------
# Fit model
# ----------------------------
model = FSH(robot_radius=r_robot, tau=tau, degree=n)
params = model.fit(obstacle_pos, r_obstacle, robot_pos)

a = params["a"]
b = params["b"]

print(params)

# ----------------------------
# Recreate geometry
# ----------------------------
vx = np.linspace(-4, 4, 500)

# Obstacle circle
ang = np.linspace(0, 2*np.pi, 200)
obs_vx = r * np.cos(ang)
obs_vy = d + r * np.sin(ang)

# VO cone
theta = np.arcsin(r / d)
m_vo = 1 / np.tan(theta)
vy_vo = np.abs(vx) * m_vo

# FVO cap
y_cap = d / tau
R_cap = r / tau
cap_vx = R_cap * np.cos(ang)
cap_vy = y_cap + R_cap * np.sin(ang)

# Standard hyperbola (n=2)
b_n2 = b  # or recompute if you stored it separately
vy_hyperbola_n2 = a * (1 + np.abs(vx / b_n2)**2)**0.5

# Super hyperbola
vy_super = a * (1 + np.abs(vx / b)**n)**(1/n)

# Tangency point (from solver)
x_tan = params["x_tan"]
y_tan = params["y_tan"]

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8, 6))
plt.grid(True)

# Obstacle
plt.fill(obs_vx, obs_vy, color=(1, 0.6, 0.6), edgecolor='r',
         label="Obstacle")

# VO + cap
plt.plot(vx, vy_vo, '--k', label="VO cone")
plt.plot(cap_vx, cap_vy, '--b', label="Time cap")

# Hyperbola n=2
plt.plot(vx, vy_hyperbola_n2, '-b', linewidth=2, label="Hyperbola n=2")
plt.plot([x_tan, -x_tan], [y_tan, y_tan], 'ob')

# Super hyperbola
plt.plot(vx, vy_super, '-g', linewidth=3,
         label=f"Super-hyperbola n={n}")
plt.plot([x_tan, -x_tan], [y_tan, y_tan], 'og')

# Labels
plt.xlabel("v_x")
plt.ylabel("v_y")
plt.title("Super-Hyperbolic Velocity Obstacle")
plt.axis("equal")
plt.xlim([-4, 4])
plt.ylim([0, d + r + 1])
plt.legend()

plt.show()