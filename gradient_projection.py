"""
Author: Marvin Ahlborn
Reference: "Optimal Control Theory: An Introduction" by Donald E. Kirk
"""

import numpy as np

class GradientProjection:
    """
    Gradient Projection Nonlinear Program Solver Algorithm.
    """
    def __init__(self, f, init_y, N_l, v_l, f_grad=None):
        """
        f is the function to be minimized.
        N_l and v_l are l linear inequality constraints.
        Constraints have the form: N_L_0i * y_0 + N_L_1i * y_1 ... + N_L_ki * y_k - v_i >= 0.
        Analytical gradients can be given by f_grad.
        """
        self.f = f
        self.init_y = init_y
        self.y = init_y
        self._f_grad = f_grad if not f_grad == None else lambda y : self._grad(f, y)

        self.N_l = N_l
        self.v_l = v_l
        self._norm_constr()

        self.active_constr = []  # Indices of active constraints
    
    def solve(self, eps, delta):
        while True:
            N_qs_inv, r, grad_proj, grad_proj_norm = self._calc_utils()

            # Stop and return result if gradient is orthogonal
            # to constraint and pointing towards infeasable area
            if grad_proj_norm <= eps and np.all(r <= 0.):
                return self.y
            
            # Otherwise determin if an active constraint should be dropped to inactivity
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
            tau_j = (v_j - N_j.T @ self.y) / N_j.T @ z
            tau_m = tau_j.min()  # Maximum possible steplength
            new_constr = tau_j.argmin()

            new_y = self.y + tau_m * z
            new_neg_f_grad = -self._f_grad(new_y)

            if z.T @ new_neg_f_grad >= 0:
                self.y = new_y

                constr_num = len(self.v_l)
                inactive_constr = [i for i in range(constr_num) if i not in self.active_constr]

                self._add_constr(inactive_constr[new_constr])  # New constraint has joined the fight
            else:
                # Oops, went too far! Find optimal y by linear interpolation.
                self.y = self._interpolate(self.y, new_y, z, delta)

    def _interpolate(self, y_a, y_b, z, delta):
        neg_a_f_grad = -self._f_grad(y_a)
        neg_b_f_grad = -self._f_grad(y_b)

        while True:
            # TODO: Interpolate between y_a, y_b on line y_b - y_a, result is called y_c
            tau_z = y_b - y_a
            theta = z.T @ neg_a_f_grad / (z.T @ neg_a_f_grad - z.T @ neg_b_f_grad)

            y_c = y_a + tau_z * theta
            neg_c_f_grad = -self._f_grad(y_c)

            if np.abs(z.T @ neg_c_f_grad) < delta:
                return y_c

            # TODO: Check if next interpolation should be done with a and c or b and c; repeat
            if z.T @ neg_c_f_grad > 0:
                y_a = y_c
                neg_a_f_grad = neg_c_f_grad
            else:
                y_b = y_c
                neg_b_f_grad = neg_c_f_grad
            # TODO: Make GitHub

    def _calc_utils(self):
        N_q = self._calc_N_q()

        N_qs = N_q.T @ N_q
        N_qs_inv = np.linalg.inv(N_qs)

        P_q_t = N_q @ N_qs_inv @ N_q.T
        I = np.eye(len(self.y))
        P_q = I - P_q_t

        neg_f_grad = -self._f_grad(self.y)
        r = N_qs_inv @ N_q.T @ neg_f_grad

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
        Calc Matrix of active constraints.
        """
        return self.N_l[:, self.active_constr]
    
    def _calc_N_j(self):
        """
        Calc Matrix of inactive constraints.
        """
        constr_num = len(self.v_l)
        inactive_constr = [i for i in range(constr_num) if i not in self.active_constr]
        
        N_j = self.N_l[:, inactive_constr]
        v_j = self.v_l[inactive_constr]

        return N_j, v_j
    
    def _calc_P_q(self):
        N_q = self._calc_N_q()

        N_qs = N_q.T @ N_q
        N_qs_inv = np.linalg.inv(N_qs)
        P_q_t = N_q @ N_qs_inv @ N_q.T

        I = np.eye(len(self.y))
        return I - P_q_t

    def _add_constr(self, i):
        self.active_constr.append(i)
        self.active_constr.sort()
    
    def _remove_constr(self, i):
        self.active_constr.remove(i)
    
    def _remove_ith_constr(self, i):
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