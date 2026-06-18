import numpy as np
from scipy.optimize import minimize_scalar, root_scalar


class FSH:
    """
    Fit a Super-Hyperbolic Velocity Obstacle
    """

    def __init__(self, robot_radius, tau=1.5, degree=2):

        # Sanity check
        assert robot_radius > 0
        assert degree > 0 and degree % 2 == 0

        self.robot_radius = robot_radius
        self.tau = tau
        self.degree = degree

    def _get_tangency_error(self, b, a, n, d, r):

        def dist_sq(x):
            y = a * (1 + (x / b) ** n) ** (1 / n)
            return x**2 + (y - d) ** 2

        # Find closest point to the circle of the superhyperbola
        res = minimize_scalar(dist_sq, bounds=(0, r), method="bounded")

        return np.sqrt(res.fun) - r

    def fit(self, obstacle_pos, obstacle_radius, robot_pos=np.array([0.0, 0.0])):

        # distance must be scalar
        distance = np.linalg.norm(obstacle_pos - robot_pos)

        r = obstacle_radius + self.robot_radius

        if distance <= r:
            raise ValueError("Cannot compute VO while colliding with the obstacle (i.e. distance <= radius)")

        # cone angle
        theta = np.arcsin(r / distance)

        m_vo = 1 / np.tan(theta)

        # truncated VO
        y_cap = distance / self.tau
        R_cap = r / self.tau

        a = (distance - r) / self.tau

        # Computation of the upper bound of b term of the superhyperbola
        D_term = distance**2 - r**2 - a**2
        inner_term = max(0.0, D_term**2 - 4 * a**2 * r**2)
        b_n2 = np.sqrt(0.5 * (D_term - np.sqrt(inner_term)))

        # Find b s.t. the distance of the closest point to the circle of the syperhyperbola is at a minimum 
        err_func = lambda b: self._get_tangency_error(b, a, self.degree, distance, r)

        sol = root_scalar(err_func, x0=b_n2)

        b_super = sol.root

        # Tangency point 
        def dist_sq(x):
            y = a * (1 + (x / b_super) ** self.degree) ** (1 / self.degree)
            return x**2 + (y - distance) ** 2

        res = minimize_scalar(dist_sq, bounds=(0, r), method="bounded")

        x_tan = res.x
        y_tan = a * (1 + (x_tan / b_super) ** self.degree) ** (1 / self.degree)

        return {
            "a": a,
            "b": b_super,
            "x_tan": x_tan,
            "y_tan": y_tan,
            "m_vo": m_vo,
            "y_cap": y_cap,
            "R_cap": R_cap,
        }