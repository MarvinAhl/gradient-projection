import numpy as np
from gradient_projection import GradientProjection

class TrajectoryOptimization:
    """
    Vectors have to be numpy column vectors if not stated differently!
    """
    def __init__(self, N, J, a, d_a_x=None, d_a_u=None):
        """
        a is a list of functions of of the form x_dot = a(x, u)
        d_a_x and d_a_u are Jacobians of partial derivatives of a and can
        be analytically supplied (highly recommended).
        """
        self.N = N  # Number of discrete collocation points
        self.J = J  # Function to be minimized
        self.a = a  # System dynamics

        self._d_a_x = self._calc_d_a_x if d_a_x == None else d_a_x
        self._d_a_u = self._calc_d_a_u if d_a_u == None else d_a_u

    def solve(self, init_x_traj, init_u_traj):
        pass
    
    def _calc_A_B_c(self, x_traj, u_traj):
        """
        x_traj and u_traj have the form N x N_x and N x N_u where
        N_x is the x-Vector's dimension and N_u is the u-Vector's dimension.
        """
        As = np.zeros((self.N, len(x), len(x)), dtype=np.float32)
        Bs = np.zeros((self.N, len(x), len(u)), dtype=np.float32)
        cs = np.zeros((self.N, len(x)), dtype=np.float32)

        for i in range(self.N):
            x = np.atleast_2d(x_traj[i]).T
            u = np.atleast_2d(u_traj[i]).T
            a = self._calc_a(x, u)
            d_a_x = self._d_a_x(x, u)
            d_a_u = self._d_a_u(x, u)

            As[i] = d_a_x
            Bs[i] = d_a_u
            cs[i] = a - d_a_x @ x - d_a_u @ u

        return As, Bs, cs
    
    def _calc_a(self, x, u):
        """
        Computes Vector of system differential equations at point x, u.
        """
        a_res = np.zeros((len(x), 1), dtype=np.float32)

        for a, i in enumerate(self.a):
            a_res[i, 0] = a(x, u)
        
        return a_res

    def _calc_d_a_x(self, x, u):
        """
        Computes Jacobian of partial derivative da/dx at point x, u.
        """
        d_a_x = np.zeros((len(x), len(x)), dtype=np.float32)

        for a, i in enumerate(self.a):
            a_x = lambda x : a(x, u)
            a_x_grad = self._grad(a_x, x)
            d_a_x[i, :] = a_x_grad.squeeze()
        
        return d_a_x
    
    def _calc_d_a_u(self, x, u):
        """
        Computes Jacobian of partial derivative da/du at point x, u.
        """
        d_a_u = np.zeros((len(x), len(u)), dtype=np.float32)

        for a, i in enumerate(self.a):
            a_u = lambda u : a(x, u)
            a_u_grad = self._grad(a_u, u)
            d_a_u[i, :] = a_u_grad.squeeze()
        
        return d_a_u

    def _grad(self, f, y, eps=1e-5):
        """
        Calculates gradient of f at y numerically.
        f takes column vector y and returns scalar.
        """
        N = len(y)
        f_y = f(y)

        grad = np.zeros((N, 1), dtype=np.float32)

        for i in range(N):
            y_e = y.copy()
            y_e[i] += eps

            f_y_e = f(y_e)

            grad[i] = (f_y_e - f_y) / eps
        
        return grad