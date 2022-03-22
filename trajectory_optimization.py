import numpy as np
from gradient_projection import GradientProjection

class TrajectoryOptimization:
    """
    Vectors have to be numpy column vectors if not stated differently!
    """
    def __init__(self, N, t_f, x_L, x_H, u_L, u_H, J, a, d_J_x=None, d_J_u=None, d_a_x=None, d_a_u=None):
        """
        N is the number of discrete collocation points. t_f is the final time.
        x_L, x_H, u_L, u_H are the upper and lower boundary constraints for state and control.
        They have to be big column vectors of all states at all collocation points concatenated with eachother
        of shape N*x_N x 1 for the state and (N-1)*x_U x 1 for the control (Like what J takes as arguments).
        a is a list of functions of of the form x_dot = a(x, u).
        d_a_x and d_a_u are Jacobians of partial derivatives of the continuous a and can
        be analytically supplied (Highly recommended!).
        J is the discrete function to be minimized. Takes two big column vectors of the form
        x = [x_1, x_2, ... , x_N]^T and u = [u_1, u_2, ... , u_(N-1)]^T
        """
        self.N = N  # Number of discrete collocation points (this N is always one more than N in book)
        self.dt = t_f / (N-1)  # Time interval between discrete collocation points

        # Upper and lower boundaries for x and u
        self.x_L = x_L
        self.x_H = x_H
        self.u_L = u_L
        self.u_H = u_H

        self.J = J  # Discrete Function to be minimized
        self._d_J_x = d_J_x  # dJ/dx
        self._d_J_u = d_J_u  # dJ/du

        self.a_D = self._discretize_dynamics(a)  # Discrete system dynamics
        # Either calculate derivatives numerically or discretize analytical derivatives
        self._d_a_x = self._calc_d_a_x if d_a_x == None else lambda x, u : np.eye(len(x)) + d_a_x(x, u) * self.dt
        self._d_a_u = self._calc_d_a_u if d_a_u == None else lambda x, u : d_a_u(x, u) * self.dt

    def solve(self, init_x_traj, init_u_traj, gamma=1e-3):
        """
        Solve Trajectory Optimization problem given an inital Trajectory.
        init_x_traj and init_u_traj are Arrays of numpy column Vectors,
        x as dimension N x N_x x 1 and u has dimension N-1 x N_u x 1
        gamma is the convergence margin.
        """
        x_traj = init_x_traj
        u_traj = init_u_traj

        # Repeat until solution converges
        while True:
            As, Bs, cs = self._calc_A_B_c(x_traj, u_traj)
            x_h = self._calc_x_h(As, cs, x_traj[0])
            D = self._calc_D(As, Bs)

            # Derive N_l and v_l from boundary conditions
            N_u_L = np.eye(len(self.u_L), dtype=np.float32)
            v_u_L = self.u_L
            N_u_H = -np.eye(len(self.u_L), dtype=np.float32)
            v_u_H = -self.u_H
            N_x_L = D
            v_x_L = self.x_L - x_h
            N_x_H = -D
            v_x_H = x_h - self.x_H

            N_l = np.concatenate((N_u_L, N_u_H, N_x_L, N_x_H), axis=0).T
            v_l = np.concatenate((v_u_L, v_u_H, v_x_L, v_x_H), axis=0)

            J_u = lambda u : self.J(D @ u + x_h, u)  # Find J of u
            if self._d_J_x == None or self._d_J_u == None:
                d_J_u_u = None
            else:
                # Total derivative dJ/du made up from dx/du * dJ/dx + dJ/du.
                d_J_u_u = lambda u : D.T @ self._d_J_x(D @ u + x_h, u) + self._d_J_u(D @ u + x_h, u)
            
            solver = GradientProjection(J_u, N_l, v_l, d_J_u_u)
            u_traj_r = np.reshape(u_traj, (len(u_traj)*len(u_traj[0]), 1))
            new_u_traj_r = solver.solve(u_traj_r)  # Reshape trajectory to be one big column vector
            
            if np.all(new_u_traj_r == None):
                return None, None, None  # Trajectory optimization failed
            
            new_u_traj = np.reshape(new_u_traj, (len(u_traj), len(u_traj[0]), 1))

            new_x_traj_r = D @ new_u_traj_r + x_h
            new_x_traj = np.reshape(new_x_traj_r, (len(x_traj), len(x_traj[0]), 1))

            # Return if trajectory converges
            if np.linalg.norm(new_u_traj_r - u_traj_r) <= gamma:
                J_min = self.J(new_x_traj_r, new_u_traj_r)  # Const value of optimal trajectory
                return new_x_traj, new_u_traj, J_min
            
            # Update trajectory and repeat
            x_traj = new_x_traj
            u_traj = new_u_traj

    def _discretize_dynamics(self, a):
        """
        Takes continuous state dynamics of the form x_dot = a(x, u) and converts
        them to a_D of the form x(k+1) = a_D(x(k), u(k)) = x(k) + a(x(k), u(k)) * dt.
        """
        a_D = []

        for constr in a:
            discr_constr = lambda x, u : x + constr(x, u) * self.dt
            a_D.append(discr_constr)

        return a_D
    
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

    def _calc_x_h(self, As, cs, x_0):
        """
        Computes helper Vector x_h.
        x_0 is the initial state of the trajectory.
        """
        x_N = len(x_0)  # State dimension
        x_h = np.zeros((x_N * self.N, 1), dtype=np.float32)
        x_h[0:x_N] = x_0

        for i in range(self.N - 1):
            x_h[(i+1)*x_N:(i+2)*x_N] = As[i] @ x_h[i*x_N:(i+1)*x_N] + cs[i]
        
        return x_h
    
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

        for a, i in enumerate(self.a_D):
            a_res[i, 0] = a(x, u)
        
        return a_res

    def _calc_d_a_x(self, x, u):
        """
        Computes Jacobian of partial derivative da/dx at point x, u.
        """
        d_a_x = np.zeros((len(x), len(x)), dtype=np.float32)

        for a, i in enumerate(self.a_D):
            a_x = lambda x : a(x, u)
            a_x_grad = self._grad(a_x, x)
            d_a_x[i, :] = a_x_grad.squeeze()
        
        return d_a_x
    
    def _calc_d_a_u(self, x, u):
        """
        Computes Jacobian of partial derivative da/du at point x, u.
        """
        d_a_u = np.zeros((len(x), len(u)), dtype=np.float32)

        for a, i in enumerate(self.a_D):
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