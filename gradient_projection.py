"""
Author: Marvin Ahlborn
Reference: "Optimal Control Theory: An Introduction" by Donald E. Kirk
"""

import numpy as np

class GradientProjection:
    """
    Gradient Projection Nonlinear Program Solver Algorithm.
    """
    def __init__(self, f, N_l, v_l, f_grad=None):
        """
        f is the function to be minimized.
        N_l and v_l are l linear inequality constraints.
        Constraints have the form: N_L_0i * y_0 + N_L_1i * y_1 ... + N_L_ki * y_k - v_i >= 0.
        Analytical gradient for f can be supplied via f_grad (highly recommended!).
        """
        self.f = f
        self._f_grad = f_grad if not f_grad == None else lambda y : self._grad(f, y)  # Choose analytical or numerical gradient

        self.N_l = N_l
        self.v_l = v_l
        self._norm_constr()

        self.active_constr = []  # Indices of active constraints
    
    def solve(self, init_y, eps=1e-3, delta=1e-3):
        """
        Solves NLP Problem for initial point y with gradient projection margin
        eps and linear interpolation margin delta.
        """
        self.y = init_y  # Solver starting point.
        self._calc_active_constr()  # Calc initially active constraints

        while True:
            N_qs_inv, r, grad_proj, grad_proj_norm = self._calc_utils()

            # Stop and return result if gradient is orthogonal
            # to constraint and pointing towards infeasable area
            if grad_proj_norm <= eps:
                if r == None or np.all(r <= 0.):
                    return self.y
            
            # Otherwise determin if an active constraint should be dropped to inactivity
            if len(self.active_constr) > 0:
                if grad_proj_norm <= eps:
                    drop_constr = r.argmax()
                    self._remove_ith_constr(drop_constr)

                    # Recalculate utilities
                    N_qs_inv, r, grad_proj, grad_proj_norm = self._calc_utils()
                else:
                    beta = np.absolute(N_qs_inv).sum(axis=1).max()
                    if r.max() > beta:
                        drop_constr = r.argmax()
                        self._remove_ith_constr(drop_constr)

                        # Recalculate utilities
                        N_qs_inv, r, grad_proj, grad_proj_norm = self._calc_utils()
            
            z = grad_proj / grad_proj_norm  # Normalized projection z

            N_j, v_j = self._calc_N_j()
            tau_j = (v_j - N_j.T @ self.y) / (N_j.T @ z + 1e-4)
            m = np.where(tau_j > 1e-4, tau_j, np.inf).argmin()
            tau_m = tau_j[m]  # Maximum possible (positive) steplength

            new_y = self.y + tau_m * z
            new_neg_f_grad = -self._f_grad(new_y)

            if z.T @ new_neg_f_grad >= 0:
                self.y = new_y

                constr_num = len(self.v_l)
                inactive_constr = [i for i in range(constr_num) if i not in self.active_constr]

                self._add_constr(inactive_constr[m])  # New constraint has joined the fight
            else:
                # Oops, went too far! Find optimal y by repeated linear interpolation.
                self.y = self._interpolate(self.y, new_y, z, delta)

    def _interpolate(self, y_a, y_b, z, delta):
        """
        Interpolates between points y_a and y_b given projection z and using self._f_grad()
        to determin optimal point on boundary hyperplane. Terminates if gradient projection
        is within delta from the desired value.
        """
        neg_a_f_grad = -self._f_grad(y_a)
        neg_b_f_grad = -self._f_grad(y_b)

        while True:
            tau_z = y_b - y_a
            theta = z.T @ neg_a_f_grad / (z.T @ neg_a_f_grad - z.T @ neg_b_f_grad)

            y_c = y_a + tau_z * theta  # Interpolate between points a and b
            neg_c_f_grad = -self._f_grad(y_c)

            if np.abs(z.T @ neg_c_f_grad) < delta:
                return y_c  # Return interpolated point if it's good enough

            # Check if optimal solution lies between c and b or a and c
            if z.T @ neg_c_f_grad > 0:
                y_a = y_c
                neg_a_f_grad = neg_c_f_grad
            else:
                y_b = y_c
                neg_b_f_grad = neg_c_f_grad

    def _calc_utils(self):
        """
        Calculates some useful matricies used by the solver.
        """
        N_q = self._calc_N_q()

        N_qs = N_q.T @ N_q
        N_qs_inv = np.linalg.inv(N_qs)

        P_q_t = N_q @ N_qs_inv @ N_q.T
        I = np.eye(len(self.y))
        P_q = I - P_q_t

        neg_f_grad = -self._f_grad(self.y)
        r = None if len(self.active_constr) == 0 else N_qs_inv @ N_q.T @ neg_f_grad

        grad_proj = P_q @ neg_f_grad
        grad_proj_norm = np.linalg.norm(grad_proj)

        return N_qs_inv, r, grad_proj, grad_proj_norm

    def _calc_r(self, y):
        N_q = self._calc_N_q()

        N_qs = N_q.T @ N_q
        N_qs_inv = np.linalg.inv(N_qs)

        neg_f_grad = -self._f_grad(y)

        return N_qs_inv @ N_q.T @ neg_f_grad

    def _calc_N_q(self):
        """
        Calculates Matrix of active constraints.
        """
        return self.N_l[:, self.active_constr]
    
    def _calc_N_j(self):
        """
        Calculates Matrix of inactive constraints.
        """
        constr_num = len(self.v_l)
        inactive_constr = [i for i in range(constr_num) if i not in self.active_constr]
        
        N_j = self.N_l[:, inactive_constr]
        v_j = self.v_l[inactive_constr]

        return N_j, v_j
    
    def _calc_P_q(self):
        """
        Calculates projection matrix P_q.
        """
        N_q = self._calc_N_q()

        N_qs = N_q.T @ N_q
        N_qs_inv = np.linalg.inv(N_qs)
        P_q_t = N_q @ N_qs_inv @ N_q.T

        I = np.eye(len(self.y))
        return I - P_q_t

    def _add_constr(self, i):
        """
        Adds constraint to active constraints list. N_q and P_q have to be
        recalculated manually still.
        """
        self.active_constr.append(i)
        self.active_constr.sort()
    
    def _remove_constr(self, i):
        """
        Same as _add_constr() but removes ith constraint overall from the active set
        (counting all constraints, active and inactive ones).
        """
        self.active_constr.remove(i)
    
    def _remove_ith_constr(self, i):
        """
        Same as _remove_constr() but removes the ith active constraint from
        the active constraints list.
        """
        self.active_constr.pop(i)
    
    def _norm_constr(self):
        """
        Normalizes constraint equations.
        """
        l = len(self.v_l)

        for i in range(l):
            norm = np.linalg.norm(self.N_l[:, i])
            self.N_l[:, i] /= norm
            self.v_l[i] /= norm
    
    def _calc_active_constr(self):
        """
        Add all constraints that are active to list of active constraints.
        """
        constr = self.N_l.T @ self.y - self.v_l
        self.active_constr = np.atleast_1d(np.argwhere(np.isclose(constr.squeeze(), 0)).squeeze()).tolist()

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


if __name__ == '__main__':
    f = lambda y : (y[0, 0]-2.)**2 + (y[1, 0]-2.)**2
    f_grad = lambda y : np.array([[2*y[0, 0]], [2*y[1, 0]]], dtype=np.float32)

    N_l = np.array([[3., 1/3, -1., 0.], [1., 1., 0., -1.]], dtype=np.float32)
    v_l = np.array([[3.], [1.], [-4.], [-4.]], dtype=np.float32)

    y = np.array([[3.], [0.7]])

    solver = GradientProjection(f, N_l, v_l)
    print(solver.solve(y))