# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:06:40 2016

@author: roobb
"""
from __future__ import division
import numpy as np
import scipy.integrate
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Cursor, CheckButtons
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# UserWarning: This figure includes Axes that are not compatible with
# tight_layout, so its results might be incorrect.


class ADRMDP(object):
    def __init__(self, **kwargs):
        """This is the Advection-Diffusion-Reaction Model for Depth Profiling.

        kwargs
        ------
        samp_len: float
            Sample length (nm) (default 20)
        n_x_steps: int
            Number of space steps (default 500)
        time: float
            Real time requested (s) (default 4.5)
        bound_cond: str
            Boundary condition 'dirichlet' | 'neumann' | 'robin'
            (default 'neumann')
        alpha_u: list of floats
            If bound_cond='dirichlet', this parameter denotes the constant
            concentrations at the surface and at the bottom for component u
            (default [0, 0])
        alpha_v: list of floats
            If bound_cond='dirichlet', this parameter denotes the constant
            concentrations at the surface and at the bottom for component v
            (default [0, 0])
        interf_slope: float
            Slope of the interface sigmoid (1/nm) (default 20)
        sampl_depth: str
            Shape of the sampling depth depth dependence 'surf' | 'sigm'
            (default 'sigmoid')
        sampl_d_ampl: float
            If sampl_depth='sigm', this parameter denotes the magnitude of the
            sampling depth sigmoid (n_at, n_mol) (default 1)
        sampl_d_slope: float
            If sampl_depth='sigm', this parameter denotes the slope of the
            sampling depth sigmoid (1/nm) (default 1.7)
        sampl_d_infl: float
            If sampl_depth='sigm', this parameter denotes the inflection point
            of the sampling depth sigmoid (nm) (default 2.2)
        l_depth_u: list of floats
            Layers of component u (nm) (default [14])
        l_width_u: list of floats
            Widths of the layers of component u (nm) (default [1])
        vel_u: float
            Velocity for component u (nm/s) (default 9)
        diff_ampl_u: float
            Magnitude of the diffusivity sigmoid for component u (nm^2/s)
            (default 18)
        diff_slope_u: float
            Slope of the diffusivity sigmoid for component u (1/nm)
            (default 0.7)
        diff_x_infl_u: float
            Inflection point of the diffusivity sigmoid for component u (nm)
            (default 3.5)
        diff_const_u: float
            Constant component of the total diffusivity for component u
            (nm^2/s) (default 0)
        l_depth_v: list of floats
            Layers of component v (nm) (default [])
        l_width_v: list of floats
            Widths of the layers of component v (nm) (default [])
        vel_v: float
            Velocity for component v (nm/s) (default vel_u)
        diff_ampl_v: float
            Magnitude of the diffusivity sigmoid for component v (nm^2/s)
            (default diff_ampl_u)
        diff_slope_v: float
            Slope of the diffusivity sigmoid for component v (1/nm)
            (default diff_slope_u)
        diff_x_infl_v: float
            Inflection point of the diffusivity sigmoid for component v (nm)
            (default diff_x_infl_u)
        diff_const_v: float
            Constant component of the total diffusivity for component v
            (nm^2/s) (default diff_const_u)
        vel_m: float
            Velocity for component m (nm/s) (default vel_u)
        reac_type: int
            Include reactions? If so, which type? (default False)
        reac_const: list of floats
            Reaction constants k_1, k_2,... (1/s) (default [5])
        reac_slope: float
            Slope of the reaction term sigmoid (1/nm) (default diff_slope_u)
        reac_x_infl: float
            Inflection point of the reaction term sigmoid
            (default diff_x_infl_u)
        model_calc: bool
            Perform calculations on instance creation? If False, the method
            calc_model() has to be invoked manually (default True)

        Example
        -------
        >>> m1 = ADRMDP()
        calculating... . . . done
        >>> m1.plot_depth_prof()
        """
        samp_len = kwargs.get('samp_len', 20)  # (nm)
        n_x_steps = kwargs.get('n_x_steps', 500)

        time = kwargs.get('time', 4.5)  # (s)

        self._bound_cond = kwargs.get('bound_cond', 'neumann')
        if self._bound_cond == 'dirichlet':
            self._alpha_u = kwargs.get('alpha_u', [0, 0])
            self._alpha_v = kwargs.get('alpha_v', [0, 0])

        interf_slope = kwargs.get('interf_slope', 20)  # (1/nm)

        sampl_depth = kwargs.get('sampl_depth', 'sigm')
        if sampl_depth == 'sigm':
            sampl_d_ampl = kwargs.get('sampl_d_ampl', 1)  # (n_at, n_mol)
            sampl_d_slope = kwargs.get('sampl_d_slope', 1.7)  # (1/nm)
            sampl_d_infl = kwargs.get('sampl_d_infl', 2.2)  # (nm)

        l_depth_u = kwargs.get('l_depth_u', [14])  # (nm)
        l_width_u = kwargs.get('l_width_u', [1])  # (nm)
        vel_u = kwargs.get('vel_u', 9)  # (nm/s)
        diff_ampl_u = kwargs.get('diff_ampl_u', 18)  # (nm^2/s)
        diff_slope_u = kwargs.get('diff_slope_u', 0.7)  # (1/nm)
        diff_x_infl_u = kwargs.get('diff_x_infl_u', 3.5)  # (nm)

        diff_const_u = kwargs.get('diff_const_u', 0)  # (nm^2/s)

        l_depth_v = kwargs.get('l_depth_v', [])  # (nm)
        l_width_v = kwargs.get('l_width_v', [])  # (nm)
        vel_v = kwargs.get('vel_v', vel_u)  # (nm/s)
        diff_ampl_v = kwargs.get('diff_ampl_v', diff_ampl_u)  # (nm^2/s)
        diff_slope_v = kwargs.get('diff_slope_v', diff_slope_u)  # (1/nm)
        diff_x_infl_v = kwargs.get('diff_x_infl_v', diff_x_infl_u)  # (nm)
        diff_const_v = kwargs.get('diff_const_v', diff_const_u)  # (nm^2/s)

        vel_m = kwargs.get('vel_m', vel_u)  # (nm/s)

        reac_type = kwargs.get('reac_type', False)
        reac_const = kwargs.get('reac_const', [5])  # (1/s)
        reac_slope = kwargs.get('reac_slope', diff_slope_u)  # (1/nm)
        reac_x_infl = kwargs.get('reac_x_infl', diff_x_infl_u)  # (nm)

        model_calc = kwargs.get('model_calc', True)
        # =====================================================================

        self._samp_len = samp_len

        L = 1  # samp_len/samp_len (1)
        self._J = n_x_steps  # dx = L/(J - 1)

        self._T = time  # (s)

        self._interf_slope = interf_slope/(1/self._samp_len)  # (1)

        if sampl_depth == 'sigm':
            sampl_d_slope = sampl_d_slope/(1/self._samp_len)  # (1)
            sampl_d_infl = sampl_d_infl/self._samp_len  # (1)

        self._l_depth_u = np.array(l_depth_u)/self._samp_len  # (1)
        self._l_width_u = np.array(l_width_u)/self._samp_len  # (1)
        self._l_depth_v = np.array(l_depth_v)/self._samp_len  # (1)
        self._l_width_v = np.array(l_width_v)/self._samp_len  # (1)

        self._a_u = -vel_u/self._samp_len  # (1/s)
        self._a_v = -vel_v/self._samp_len  # (1/s)
        self._a_m = -vel_m/self._samp_len  # (1/s)
        # nagative, the mass is traveling in the negative direction of x

        diff_ampl_u = diff_ampl_u/self._samp_len**2  # (1/s)
        diff_ampl_v = diff_ampl_v/self._samp_len**2  # (1/s)

        diff_slope_u = diff_slope_u/(1/self._samp_len)  # (1)
        diff_slope_v = diff_slope_v/(1/self._samp_len)  # (1)

        diff_x_infl_u = diff_x_infl_u/self._samp_len  # (1)
        diff_x_infl_v = diff_x_infl_v/self._samp_len  # (1)

        diff_u_const = diff_const_u/self._samp_len**2  # (1/s)
        diff_v_const = diff_const_v/self._samp_len**2  # (1/s)

        self._dx = L/(self._J - 1)  # (1)
        self._x_grid = np.array([j * self._dx for j in range(self._J)])

        self._t_grid = np.zeros(1)

        if sampl_depth == 'surf':
            self._sampl_d_x = np.zeros(self._J)
            self._sampl_d_x[0] = 1
        if sampl_depth == 'sigm':
            self._sampl_d_x = self._get_sigm_fun(sampl_d_infl, sampl_d_slope,
                                                 sampl_d_ampl)

        self._d_term_x_u = self._get_sigm_fun(diff_x_infl_u, diff_slope_u)
        self._d_term_x_v = self._get_sigm_fun(diff_x_infl_v, diff_slope_v)

        d_term_x_deriv_u = self._get_sigm_fun(diff_x_infl_u, diff_slope_u,
                                              deriv=True)
        d_term_x_deriv_v = self._get_sigm_fun(diff_x_infl_v, diff_slope_v,
                                              deriv=True)

        self._d_term_total_u = diff_ampl_u * self._d_term_x_u + diff_u_const
        self._d_term_total_v = diff_ampl_v * self._d_term_x_v + diff_v_const

        self._d_term_total_u_deriv = diff_ampl_u * d_term_x_deriv_u
        self._d_term_total_v_deriv = diff_ampl_v * d_term_x_deriv_v

        self._reac_type = reac_type
        self._reac_const = reac_const  # (1/s)
        reac_slope = reac_slope/(1/self._samp_len)  # (1)
        reac_x_infl = reac_x_infl/self._samp_len  # (1)

        self._r_term_x = self._get_sigm_fun(reac_x_infl, reac_slope)

        if model_calc is True:
            self.calc_model()
    # =========================================================================

    @staticmethod
    def sigm_fun(x, x_infl, slope, ampl, deriv=False):
        '''Get sigmoid s(x) or its derivative if deriv=True.

        Example
        -------
        >>> x = linspace(0, 10)
        >>> plot(x, ADRMDP.sigm_fun(x, 1, 1, 1))
        '''
        if deriv:
            return -ampl * slope * np.exp(slope * (x - x_infl)) / \
                (np.exp(slope * (x - x_infl)) + 1)**2
        else:
            return ampl/(np.exp(slope * (x - x_infl)) + 1)

    def _get_sigm_fun(self, x_infl, slope, ampl=1, deriv=False):
        '''Get sigmoid function.'''
        return self.sigm_fun(self._x_grid, x_infl, slope, ampl, deriv)
    # =========================================================================

    def _get_ini_cond(self):
        '''Get initial condition.'''
        U = np.zeros(self._J)
        for x1, x2 in zip(self._l_depth_u, self._l_depth_u + self._l_width_u):
            x_one_fourth = (x2 - x1)/4
            j1 = int((x1 - x_one_fourth)/self._dx)
            j2 = int((x1 + 2 * x_one_fourth)/self._dx)
            j3 = int((x2 - 2 * x_one_fourth)/self._dx)
            j4 = int((x2 + x_one_fourth)/self._dx)
            if j1 < 0:
                j1 = 0
            if j2 < 0:
                j2 = 0
            if j1 > self._J:
                j1 = self._J
            if j2 > self._J:
                j2 = self._J
            if j3 < 0:
                j3 = 0
            if j4 < 0:
                j4 = 0
            if j3 > self._J:
                j3 = self._J
            if j4 > self._J:
                j4 = self._J
            U[j1: j2] = (-self._get_sigm_fun(x1,
                                             self._interf_slope) + 1)[j1: j2]
            U[j3 - 1: j4] = self._get_sigm_fun(x2,
                                               self._interf_slope)[j3 - 1: j4]

        V = np.zeros(self._J)
        for x1, x2 in zip(self._l_depth_v, self._l_depth_v + self._l_width_v):
            x_one_fourth = (x2 - x1)/4
            j1 = int((x1 - x_one_fourth)/self._dx)
            j2 = int((x1 + 2 * x_one_fourth)/self._dx)
            j3 = int((x2 - 2 * x_one_fourth)/self._dx)
            j4 = int((x2 + x_one_fourth)/self._dx)
            if j1 < 0:
                j1 = 0
            if j2 < 0:
                j2 = 0
            if j1 > self._J:
                j1 = self._J
            if j2 > self._J:
                j2 = self._J
            if j3 < 0:
                j3 = 0
            if j4 < 0:
                j4 = 0
            if j3 > self._J:
                j3 = self._J
            if j4 > self._J:
                j4 = self._J
            V[j1: j2] = (-self._get_sigm_fun(x1,
                                             self._interf_slope) + 1)[j1: j2]
            V[j3 - 1: j4] = self._get_sigm_fun(x2,
                                               self._interf_slope)[j3 - 1: j4]

        return U, V
    # =========================================================================

    def _calc_matrices(self, U, V, M):
        '''Calculate matrices.'''
        u_surf = np.average(U, weights=self._sampl_d_x)
        v_surf = np.average(V, weights=self._sampl_d_x)
        m_surf = np.average(M, weights=self._sampl_d_x)

        if m_surf < 0:
            a = (self._a_u * u_surf + self._a_v * v_surf)/(u_surf + v_surf)
        else:
            a = self._a_u * u_surf + self._a_v * v_surf + self._a_m * m_surf

        self._dt = self._dx/abs(a)

        sigma_u = (self._d_term_total_u * self._dt)/(2 * self._dx**2)
        sigma_v = (self._d_term_total_v * self._dt)/(2 * self._dx**2)

        rho_u = (self._d_term_total_u_deriv * self._dt) / (4 * self._dx)
        rho_v = (self._d_term_total_v_deriv * self._dt) / (4 * self._dx)

        # Dirichlet boundary condition
        self._A_u = np.diagflat(-(sigma_u[: -1] + rho_u[: -1]), 1) + \
            np.diagflat(1 + 2 * sigma_u) + \
            np.diagflat(-sigma_u[1:] + rho_u[1:], -1)

        self._B_u = np.diagflat(sigma_u[: -1] + rho_u[: -1], 1) + \
            np.diagflat(1 - 2 * sigma_u) + \
            np.diagflat(sigma_u[1:] - rho_u[1:], -1)

        self._A_v = np.diagflat(-(sigma_v[: -1] + rho_v[: -1]), 1) + \
            np.diagflat(1 + 2 * sigma_v) + \
            np.diagflat(-sigma_v[1:] + rho_v[1:], -1)

        self._B_v = np.diagflat(sigma_v[: -1] + rho_v[: -1], 1) + \
            np.diagflat(1 - 2 * sigma_v) + \
            np.diagflat(sigma_v[1:] - rho_v[1:], -1)

        if self._bound_cond == 'dirichlet':
            self._dirich_term_u[0] = (2 * sigma_u[0] - 2 * rho_u[0]) * \
                self._alpha_u[0]
            self._dirich_term_u[-1] = (2 * sigma_u[-1] + 2 * rho_u[-1]) * \
                self._alpha_u[1]

            self._dirich_term_v[0] = (2 * sigma_v[0] - 2 * rho_v[0]) * \
                self._alpha_v[0]
            self._dirich_term_v[-1] = (2 * sigma_v[-1] + 2 * rho_v[-1]) * \
                self._alpha_v[1]

        if self._bound_cond == 'neumann' or self._bound_cond == 'robin':
            self._A_u[0, 1] = -2 * sigma_u[0]
            self._A_u[-1, -2] = -2 * sigma_u[-1]

            self._B_u[0, 1] = 2 * sigma_u[0]
            self._B_u[-1, -2] = 2 * sigma_u[-1]

            self._A_v[0, 1] = -2 * sigma_v[0]
            self._A_v[-1, -2] = -2 * sigma_v[-1]

            self._B_v[0, 1] = 2 * sigma_v[0]
            self._B_v[-1, -2] = 2 * sigma_v[-1]

        if self._bound_cond == 'robin':
            self._A_u[0, 0] = 1 + 2 * sigma_u[0] - 2 * self._dx * a * \
                (-sigma_u[0] + rho_u[0]) / self._d_term_total_u[0]
            self._A_u[-1, -1] = 1 + 2 * sigma_u[-1] - 2 * self._dx * a * \
                (sigma_u[-1] + rho_u[-1]) / self._d_term_total_u[-1]

            self._B_u[0, 0] = 1 - 2 * sigma_u[0] - 2 * self._dx * a * \
                (sigma_u[0] - rho_u[0]) / self._d_term_total_u[0]
            self._B_u[-1, -1] = 1 - 2 * sigma_u[-1] + 2 * self._dx * a * \
                (sigma_u[-1] + rho_u[-1]) / self._d_term_total_u[-1]

            self._A_v[0, 0] = 1 + 2 * sigma_v[0] - 2 * self._dx * a * \
                (-sigma_v[0] + rho_v[0]) / self._d_term_total_v[0]
            self._A_v[-1, -1] = 1 + 2 * sigma_v[-1] - 2 * self._dx * a * \
                (sigma_v[-1] + rho_v[-1]) / self._d_term_total_v[-1]

            self._B_v[0, 0] = 1 - 2 * sigma_v[0] - 2 * self._dx * a * \
                (sigma_v[0] - rho_v[0]) / self._d_term_total_v[0]
            self._B_v[-1, -1] = 1 - 2 * sigma_v[-1] + 2 * self._dx * a * \
                (sigma_v[-1] + rho_v[-1]) / self._d_term_total_v[-1]

        return a
    # =========================================================================

    def _calc_r_terms(self, U, V, M):
        '''Calculate reaction terms.'''
        if self._reac_type is False:
            self._r_term_u = 0
            self._r_term_v = 0

        if self._reac_type == 1:
            r_term_conc = -self._reac_const[0] * U  # -k_1 * u
            self._r_term_u = r_term_conc * self._r_term_x

            self._r_term_v = -self._r_term_u  # -(-k_1 * u)

        if self._reac_type == 2:
            r_term_conc = -self._reac_const[0] * U * V  # -k_1 * u * v
            self._r_term_u = r_term_conc * self._r_term_x

            self._r_term_v = self._r_term_u  # -k_1 * u * v

        if self._reac_type == 3:
            r_term_conc = -self._reac_const[0] * U * M  # -k_1 * u * m
            self._r_term_u = r_term_conc * self._r_term_x

            self._r_term_v = -self._r_term_u  # -(-k_1 * u * m)
    # =========================================================================

    def calc_model(self):
        '''Run calculations. Invoke if the ADRMDP instance was created with
        model_calc=False.'''
        print 'calculating...',

        U, V = self._get_ini_cond()

        self._U_xt = np.zeros((1, self._J))
        self._V_xt = np.zeros((1, self._J))
        self._M_xt = np.zeros((1, self._J))
        self._a_t = np.zeros(1)
        # a record of temporary a's needed to convert t to x

        self._dirich_term_u = np.zeros(self._J)
        self._dirich_term_v = np.zeros(self._J)

        self._U_xt[0] = U
        self._V_xt[0] = V
        M = 1 - U - V
        self._M_xt[0] = M
        a = self._calc_matrices(U, V, M)
        self._a_t[0] = a

        self._calc_r_terms(U, V, M)

        while(True):
            U_new = np.linalg.solve(self._A_u, self._B_u.dot(U) +
                                    self._r_term_u * self._dt +
                                    self._dirich_term_u)
            V_new = np.linalg.solve(self._A_v, self._B_v.dot(V) +
                                    self._r_term_v * self._dt +
                                    self._dirich_term_v)

            if a < 0:
                if self._bound_cond == 'dirichlet':
                    alpha_u = self._alpha_u[1]
                    alpha_v = self._alpha_v[1]
                if self._bound_cond == 'neumann':
                    alpha_u = U_new[-2]
                    alpha_v = V_new[-2]
                if self._bound_cond == 'robin':
                    alpha_u = U_new[-2] + 2 * self._dx * a * U_new[-1] / \
                        self._d_term_total_u[-1]
                    alpha_v = V_new[-2] + 2 * self._dx * a * V_new[-1] / \
                        self._d_term_total_v[-1]

                U_new = np.concatenate((U_new[1:], [alpha_u]))
                V_new = np.concatenate((V_new[1:], [alpha_v]))

            if a > 0:
                if self._bound_cond == 'dirichlet':
                    alpha_u = self._alpha_u[0]
                    alpha_v = self._alpha_v[0]
                if self._bound_cond == 'neumann':
                    alpha_u = U_new[1]
                    alpha_v = V_new[1]
                if self._bound_cond == 'robin':
                    alpha_u = U_new[1] - 2 * self._dx * a * U_new[0] / \
                        self._d_term_total_u[0]
                    alpha_v = V_new[1] - 2 * self._dx * a * V_new[0] / \
                        self._d_term_total_v[0]
                U_new = np.concatenate(([alpha_u], U_new[: -1]))
                V_new = np.concatenate(([alpha_v], V_new[: -1]))

            if self._t_grid[-1] >= self._T:
                break

            self._t_grid = np.concatenate((self._t_grid,
                                           [self._t_grid[-1] + self._dt]))

            U = U_new
            V = V_new

            self._U_xt = np.concatenate((self._U_xt, [U]))
            self._V_xt = np.concatenate((self._V_xt, [V]))
            M = 1 - U - V
            self._M_xt = np.concatenate((self._M_xt, [M]))
            a = self._calc_matrices(U, V, M)
            self._a_t = np.concatenate((self._a_t, [a]))

            self._calc_r_terms(U, V, M)

            if len(self._t_grid) % 500 == 0:
                print '%i%%' % (self._t_grid[-1]/self._T * 100),

        print 'done'
    # =========================================================================

    def plot_vel(self, comps=['u']):
        '''Plot velocities for specified componets.

        comps=['u', 'v', 'm'] (default ['u'])
        '''
        x_pos = []
        x_labs = []
        vels = []
        for i, comp in enumerate(comps, 1):
            x_pos.append(i)
            if comp == 'u':
                x_labs.append('u')
                vels.append(abs(self._a_u * self._samp_len))
            if comp == 'v':
                x_labs.append('v')
                vels.append(abs(self._a_v * self._samp_len))
            if comp == 'm':
                x_labs.append('m')
                vels.append(abs(self._a_m * self._samp_len))
        plt.bar(x_pos, vels, align='center')
        plt.xticks(x_pos, x_labs)
        plt.ylabel('vel (nm/s)')
        plt.show()

    def plot_sampl_d(self):
        '''Plot sampling depth sigmoid.'''
        plt.plot(self._x_grid * self._samp_len, self._sampl_d_x)
        plt.xlabel('x (nm)')
        plt.ylabel('S(x) (n$_{at}$, n$_{mol}$)')
        plt.show()

    def plot_diff(self, comps=['u']):
        '''Plot total diffusivity for specified components.

        comps=['u', 'v', 'm'] (default ['u'])
        '''
        x = self._x_grid * self._samp_len
        for comp in comps:
            if comp == 'u':
                y = self._d_term_total_u * self._samp_len**2
            if comp == 'v':
                y = self._d_term_total_v * self._samp_len**2
            plt.plot(x, y, label=comp)
        plt.xlabel('x (nm)')
        plt.ylabel('D(x) (nm$^2$/s)')
        plt.legend(loc='best')
        plt.show()

    def plot_reac(self):
        '''Plot the sigmoid of reaction terms.'''
        plt.plot(self._x_grid * self._samp_len, self._r_term_x)
        plt.xlabel('x')
        plt.ylabel('s$_R$(x)')
        plt.show()

    def plot_conc(self, comp='u'):
        '''Plot concentration surface c(x, t) for the specified component.

        comp='u' | 'v' | 'm' (default 'u')
        '''
        x = self._x_grid * self._samp_len
        t = self._t_grid
        if comp == 'u':
            y = self._U_xt
        if comp == 'v':
            y = self._V_xt
        if comp == 'm':
            y = self._M_xt
        heatmap = plt.pcolormesh(x, t, y, vmin=0, vmax=1)
        plt.xlabel('x (nm)')
        plt.ylabel('t (s)')
        cbar = plt.colorbar(heatmap)
        cbar.set_label(comp)
        plt.show()

        if comp == 'm' and np.any(self._M_xt < 0):
            y = self._M_xt
            heatmap = plt.pcolormesh(x, t, y, vmin=-1, vmax=0,
                                     cmap=cm.Greys_r)
            plt.xlabel('x (nm)')
            plt.ylabel('t (s)')
            cbar = plt.colorbar(heatmap)
            cbar.set_label(comp)
            plt.show()

    def plot_conc_sect(self, comps=['u'], times=[0]):
        '''Plot sections of the concentration surface c(x, t) for specified
        times.

        comps=['u', 'v', 'm'] (default ['u'])

        e.g. times=[0, 2, 5] (default [0])
        '''
        x = self._x_grid * self._samp_len

        for time in times:
            for comp in comps:
                if comp == 'u':
                    y = self._U_xt[np.argmax(self._t_grid >= time), :]
                if comp == 'v':
                    y = self._V_xt[np.argmax(self._t_grid >= time), :]
                if comp == 'm':
                    y = self._M_xt[np.argmax(self._t_grid >= time), :]
                plt.plot(x, y, label='%s, %s' % (comp, time))
        plt.ylim(-0.1, 1.1)
        plt.xlabel('x (nm)')
        plt.ylabel('c')
        plt.legend(loc='best')
        plt.show()

    def plot_depth_prof(self, comps=['u'], var='depth', comp_max=0.2):
        '''Plot depth profiles for specified components.

        comps=['u', 'v', 'm'] (default ['u'])

        var='depth' | 'time' (default 'depth')

        e.g. comp_max=1.1 (default 0.2)
        '''
        if var == 'depth':
            x = np.array([i * self._dx * self._samp_len for
                          i in range(self._t_grid.shape[0])])  # t -> x (nm)
        if var == 'time':
            x = self._t_grid

        for comp in comps:
            if comp == 'u':
                # u(t, x ~ 0)
                y = np.average(self._U_xt, axis=1, weights=self._sampl_d_x)
            if comp == 'v':
                # v(t, x ~ 0)
                y = np.average(self._V_xt, axis=1, weights=self._sampl_d_x)
            if comp == 'm':
                # m(t, x ~ 0)
                y = np.average(self._M_xt, axis=1, weights=self._sampl_d_x)
            col = plt.plot(x, y, label=comp)[0].get_color()

            if var == 'depth':
                if comp == 'u':
                    for d, w in zip(self._l_depth_u, self._l_width_u):
                        plt.axvspan(d * self._samp_len,
                                    (d + w) * self._samp_len, color=col,
                                    alpha=0.2)
                if comp == 'v':
                    for d, w in zip(self._l_depth_v, self._l_width_v):
                        plt.axvspan(d * self._samp_len,
                                    (d + w) * self._samp_len, color=col,
                                    alpha=0.2)
                plt.xlabel('x (nm)')

            if var == 'time':
                depth = np.array([i * self._dx for i in
                                 range(self._t_grid.shape[0])]) \
                    # t -> x (1)
                if comp == 'u':
                    for d, w in zip(self._l_depth_u, self._l_width_u):
                        t_d = self._t_grid[np.argmax(depth >= d)]
                        t_dw = self._t_grid[np.argmax(depth >= d + w)]

                        plt.axvspan(t_d, t_dw, color=col, alpha=0.2)
                if comp == 'v':
                    for d, w in zip(self._l_depth_v, self._l_width_v):
                        t_d = self._t_grid[np.argmax(depth >= d)]
                        t_dw = self._t_grid[np.argmax(depth >= d + w)]

                        plt.axvspan(t_d, t_dw, color=col, alpha=0.2)

                plt.xlabel('t (s)')
        plt.ylabel('c')
        if 'm' in comps:
            plt.ylim(0, 1.1)
        else:
            plt.ylim(0, comp_max)
        plt.legend(loc='best')
        plt.show()

    def calc_dp_prop(self, comp='u'):
        ''' Calculate delta-layer depth profile properties for the specified
        component (default 'u').'''
        print 'comp:', comp

        x = self._t_grid * abs(self._a_t) * self._samp_len  # t -> x (nm)
        if comp == 'u':
            # u(t, x ~ 0)
            y = np.average(self._U_xt, axis=1, weights=self._sampl_d_x)
        if comp == 'v':
            # v(t, x ~ 0)
            y = np.average(self._V_xt, axis=1, weights=self._sampl_d_x)
        if comp == 'm':
            # m(t, x ~ 0)
            y = np.average(self._M_xt, axis=1, weights=self._sampl_d_x)

        integral = scipy.integrate.simps(y, x=x)
        print 'integral: %g nm' % integral

        height = y[y.argmax()]
        print 'height: %g' % height

        peak_pos = x[y.argmax()]
        print 'peak_pos: %g nm' % peak_pos

        y_max = y[y.argmax()]
        n = y.shape[0]
        x1 = 0
        x2 = 0
        for k in range(n):
            thL = y[n - k - 1]
            if thL > y_max * 0.8413:
                x2 = x[n - k - 1]
            if thL > y_max * 0.1587:
                x1 = x[n - k - 1]
        e1684 = x2 - x1
        print 'e1684: %g nm' % e1684

        y_max = y[y.argmax()]
        n = y.shape[0]
        x1 = 0
        x2 = 0
        for k in range(n):
            thR = y[k]
            thL = y[n - k - 1]
            if thR > y_max * 0.5:
                x2 = x[k]
            if thL > y_max * 0.5:
                x1 = x[n - k - 1]
        fwhm50 = x2 - x1
        print 'fwhm50: %g nm' % fwhm50

        y_max = y[y.argmax()]
        n = y.shape[0]
        x1 = 0
        x2 = 0
        for k in range(n):
            thR = y[k]
            thL = y[n - k - 1]
            if thR > y_max * 0.6067:
                x2 = x[k]
            if thL > y_max * 0.6067:
                x1 = x[n - k - 1]
        fwhm61 = x2 - x1
        print 'fwhm61: %g nm' % fwhm61

        y_max = y[y.argmax()]
        n = y.shape[0]
        x1 = 0
        x2 = 0
        for k in range(n):
            thR = y[k]
            if thR > y_max * 0.8413:
                x1 = x[k]
            if thR > y_max * 0.1587:
                x2 = x[k]
        e8416 = x2 - x1
        print 'e8416: %g nm' % e8416

        y_sum = y.sum()
        n = y.shape[0]
        xThSum = 0
        for k in range(n):
            xThSum += x[k] * y[k]
        mean = xThSum/y_sum
        print 'mean: %g nm' % mean

        y_integr = scipy.integrate.simps(y, x=x)
        n = y.shape[0]
        thIntegr = 0
        for k in range(n):
            thIntegr += y[k] * (x[k + 1] - x[k])
            if thIntegr >= 0.5 * y_integr:
                median = x[k]
                break
        print 'median: %g nm' % median

    def plot_interact_pres(self):
        '''Plot interactive presentation of the results of the model.
        '''
        fig = plt.figure('ADRM interactive presentation', figsize=(10, 8))
        plt.subplots_adjust(left=0.07, bottom=0.3, right=0.95, top=0.95,
                            hspace=0.39)

        ax = fig.add_subplot(211)
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('c')
        ax.set_title('Concentration vs. depth for given time')
        ax.grid(True)
        plt.axis([0, self._samp_len, -0.05, 1.05])

        x = self._x_grid * self._samp_len
        y = self._U_xt[0, :]
        l0, = ax.plot(x, y, 'b', lw=2)
        y = self._V_xt[0, :]
        l1, = ax.plot(x, y, 'g', visible=False, lw=2)
        y = self._M_xt[0, :]
        l2, = ax.plot(x, y, 'r', visible=False, lw=2)

        ax.plot(x, self._sampl_d_x, 'k--', lw=1.5, label='sampl_d')
        l3, = ax.plot(x, self._d_term_x_u, 'b', lw=1, label='diff_u')
        l4, = ax.plot(x, self._d_term_x_v, 'g', lw=1, label='diff_v')
        if self._reac_type is not False:
            ax.plot(x, self._r_term_x, 'c-.', lw=1, label='reac')
        ax.legend(loc='best')
        l4.set_visible(False)

        for d, w in zip(self._l_depth_u, self._l_width_u):
            plt.axvspan(d * self._samp_len, (d + w) * self._samp_len,
                        color='blue', alpha=0.2)
        for d, w in zip(self._l_depth_v, self._l_width_v):
            plt.axvspan(d * self._samp_len, (d + w) * self._samp_len,
                        color='green', alpha=0.2)

        ax2 = fig.add_subplot(212)
        ax2.set_xlabel('t (s)')
        ax2.set_ylabel('c')
        ax2.set_title('Concentration vs. time at the surface')
        ax2.grid(True)
        plt.axis([0, self._T, -0.05, 1.05])

        x = self._t_grid
        y = np.average(self._U_xt, axis=1, weights=self._sampl_d_x)
        l5, = ax2.plot(x, y, 'b', lw=2, label='u')
        y = np.average(self._V_xt, axis=1, weights=self._sampl_d_x)
        l6, = ax2.plot(x, y, 'g', lw=2, label='v')
        y = np.average(self._M_xt, axis=1, weights=self._sampl_d_x)
        l7, = ax2.plot(x, y, 'r', lw=2, label='m')
        ax2.legend(loc='best')
        l6.set_visible(False)
        l7.set_visible(False)

        l8, = ax2.plot([0, 0], [0, 1.1], 'k--', lw=1.5)

        for d, w in zip(self._l_depth_u, self._l_width_u):
            plt.axvspan(d/abs(self._a_t.mean()),
                        (d + w)/abs(self._a_t.mean()), color='blue',
                        alpha=0.2)
        for d, w in zip(self._l_depth_v, self._l_width_v):
            plt.axvspan(d/abs(self._a_t.mean()),
                        (d + w)/abs(self._a_t.mean()), color='green',
                        alpha=0.2)

        axS = plt.axes([0.13, 0.2, 0.77, 0.03])
        self._sT = Slider(axS, 'time (s)', 0, self._T, valinit=0)

        def update(val):
            time = self._sT.val

            time_ind = np.argmax(self._t_grid >= time)
            y = self._U_xt[time_ind, :]
            l0.set_ydata(y)
            y = self._V_xt[time_ind, :]
            l1.set_ydata(y)
            y = self._M_xt[time_ind, :]
            l2.set_ydata(y)

            l8.set_xdata([time, time])

            plt.draw()
        self._sT.on_changed(update)

        self._cursor = Cursor(ax, useblit=True, linestyle='--',
                              linewidth=1)
        self._cursor2 = Cursor(ax2, useblit=True, linestyle='--',
                               linewidth=1)

        axC = plt.axes([0.13, 0.012, 0.1, 0.15])
        self._checkB = CheckButtons(axC, ('u', 'v', 'm'),
                                    (True, False, False))

        def func(label):
            if label == 'u':
                l0.set_visible(not l0.get_visible())
                l3.set_visible(not l3.get_visible())
                l5.set_visible(not l5.get_visible())
            elif label == 'v':
                l1.set_visible(not l1.get_visible())
                l4.set_visible(not l4.get_visible())
                l6.set_visible(not l6.get_visible())
            elif label == 'm':
                l2.set_visible(not l2.get_visible())
                l7.set_visible(not l7.get_visible())
            plt.draw()
        self._checkB.on_clicked(func)

        plt.show()
# =============================================================================


def comp_vel(*args, **kwargs):
    '''Compare velocities.

    e.g. args=m1, m2,...

    kwargs
    ------
    comp:
        'u' | 'v' | 'm' (default 'u')
    '''
    comp = kwargs.get('comp', 'u')
    x_pos = []
    vels = []
    for i, m in enumerate(args, 1):
        x_pos.append(i)
        if comp == 'u':
            vels.append(abs(m._a_u * m._samp_len))
        if comp == 'v':
            vels.append(abs(m._a_v * m._samp_len))
        if comp == 'm':
            vels.append(abs(m._a_m * m._samp_len))
    plt.bar(x_pos, vels, align='center')
    plt.xticks(x_pos)
    if comp == 'u':
        plt.ylabel('vel$_u$ (nm/s)')
    if comp == 'v':
        plt.ylabel('vel$_v$ (nm/s)')
    if comp == 'm':
        plt.ylabel('vel$_m$ (nm/s)')
    plt.show()


def comp_sampl_d(*args):
    '''Compare sampling depth sigmoids.

    e.g. args=m1, m2,...
    '''
    for i, m in enumerate(args):
        x = m._x_grid * m._samp_len
        plt.plot(x, m._sampl_d_x, label=i + 1)
    plt.xlabel('x (nm)')
    plt.ylabel('S(x) (n$_{at}$, n$_{mol}$)')
    plt.legend(loc='best')
    plt.show()


def comp_diff(*args, **kwargs):
    '''Compare total diffusivities.

    e.g. args=m1, m2,...

    kwargs
    ------
    comp:
        'u' | 'v' | 'm' (default 'u')
    '''
    comp = kwargs.get('comp', 'u')
    for i, m in enumerate(args):
        x = m._x_grid * m._samp_len
        if comp == 'u':
            y = m._d_term_total_u * m._samp_len**2
        if comp == 'v':
            y = m._d_term_total_v * m._samp_len**2
        plt.plot(x, y, label=i + 1)
    plt.xlabel('x (nm)')
    if comp == 'u':
        plt.ylabel('D$_u$(x) (nm$^2$/s)')
    if comp == 'v':
        plt.ylabel('D$_v$(x) (nm$^2$/s)')
    plt.legend(loc='best')
    plt.show()


def comp_depth_prof(*args, **kwargs):
    '''Compare depth profiles.

    e.g. args=m1, m2,...

    kwargs
    ------
    comp:
        'u' | 'v' | 'm' (default 'u')
    comp_max:float
        (default 0.2)
    '''
    comp = kwargs.get('comp', 'u')
    comp_max = kwargs.get('comp_max', 0.2)
    for n, m in enumerate(args):
        x = np.array([i * m._dx * m._samp_len for
                      i in range(m._t_grid.shape[0])])  # t -> x (nm)

        if comp == 'u':
            # u(t, x ~ 0)
            y = np.average(m._U_xt, axis=1, weights=m._sampl_d_x)
        if comp == 'v':
            # v(t, x ~ 0)
            y = np.average(m._V_xt, axis=1, weights=m._sampl_d_x)
        if comp == 'm':
            # m(t, x ~ 0)
            y = np.average(m._M_xt, axis=1, weights=m._sampl_d_x)

        col = plt.plot(x, y, label=n + 1)[0].get_color()
        if n == 0:
            if comp == 'u':
                for d, w in zip(m._l_depth_u, m._l_width_u):
                    plt.axvspan(d * m._samp_len, (d + w) * m._samp_len,
                                color=col, alpha=0.2)
            if comp == 'v':
                for d, w in zip(m._l_depth_v, m._l_width_v):
                    plt.axvspan(d * m._samp_len, (d + w) * m._samp_len,
                                color=col, alpha=0.2)
    if comp == 'm':
        plt.ylim(0, 1.1)
    else:
        plt.ylim(0, comp_max)
    plt.xlabel('x (nm)')
    if comp == 'u':
        plt.ylabel('u')
    if comp == 'v':
        plt.ylabel('v')
    if comp == 'm':
        plt.ylabel('m')
    plt.legend(loc='best')
    plt.show()
