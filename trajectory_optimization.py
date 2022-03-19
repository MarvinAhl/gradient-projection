import numpy as np
from gradient_projection import GradientProjection

class TrajectoryOptimization:
    """
    Vectors have to be numpy column vectors if not stated differently!
    """
    def __init__(self, N, J, a, d_a_x=None, d_a_u=None):
        """
        a is a list of functions of of the form x(k+1) = a(x(k), u(k))
        = x(k) + a_c(x(k), u(k)) * dt
        derived from the continuous equation x_dot = a_c(x, u).
        d_a_x and d_a_u are Jacobians of partial derivatives of a and can
        be analytically supplied (highly recommended).
        """
        self.N = N  # Number of discrete collocation points (this N is always one more than N in book)
        self.J = J  # Function to be minimized
        self.a = a  # Discrete system dynamics

        self._d_a_x = self._calc_d_a_x if d_a_x == None else d_a_x
        self._d_a_u = self._calc_d_a_u if d_a_u == None else d_a_u

    def solve(self, init_x_traj, init_u_traj):
        """
        Solve Trajectory Optimization problem given an inital Trajectory.
        init_x_traj and init_u_traj are Arrays of numpy column Vectors,
        x as dimension N x N_x x 1 and u has dimension N-1 x N_u x 1
        """
        x_traj = init_x_traj
        u_traj = init_u_traj

        # Repeat until solution converges
        while True:
            As, Bs, cs = self._calc_A_B_c(x_traj, u_traj)
            x_H = self._calc_x_H(As, cs, x_traj[0])
            D = self._calc_D(As, Bs)

            # TODO: Rest
    
    def _calc_D(self, As, Bs):
        """
        Computes helper matrix D.
        """
        x_N = len(As[0, 0])  # State dimension
        u_N = len(Bs[0, 0])  # Control dimension
        D = np.zeros((x_N * self.N, u_N * (self.N-1)), dtype=np.float32)

        # Fill diagonal
        for l in range(self.N - 1):
            D[(l+1)*x_N:(l+2)*x_N, l*u_N:(l+1)*u_N] = Bs[l]

        # Fill lower triangel
        for j in range(2, self.N):
            for l in range(j-1):
                k = j - 1

                D_lk = Bs[l]  # Compute D's submatrix from A and B
                for i in range(k-l):
                    D_lk = As[l+i+1] @ D_lk

                D[j*x_N:(j+1)*x_N, l*u_N:(l+1)*u_N] = D_lk

        return D

    def _calc_x_H(self, As, cs, x_0):
        """
        Computes helper Vector x_H.
        x_0 is the initial state of the trajectory.
        """
        x_N = len(x_0)  # State dimension
        x_H = np.zeros((x_N * self.N, 1), dtype=np.float32)
        x_H[0:x_N] = x_0

        for i in range(self.N - 1):
            x_H[(i+1)*x_N:(i+2)*x_N] = As[i] @ x_H[i*x_N:(i+1)*x_N] + cs[i]
        
        return x_H
    
    def _calc_A_B_c(self, x_traj, u_traj):
        """
        x_traj and u_traj have the form N x N_x x 1 and N-1 x N_u x 1 where
        N_x is the x-Vector's dimension and N_u is the u-Vector's dimension.
        """
        As = np.zeros((self.N-1, len(x), len(x)), dtype=np.float32)
        Bs = np.zeros((self.N-1, len(x), len(u)), dtype=np.float32)
        cs = np.zeros((self.N-1, len(x), 1), dtype=np.float32)

        for i in range(self.N - 1):
            x = x_traj[i]
            u = u_traj[i]
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