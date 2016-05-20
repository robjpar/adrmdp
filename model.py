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
        n_t_steps: int
            Number of time steps (default 500)
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
        samp_d_slope: float
            Slope of the sampling depth sigmoid (1/nm) (default 1.7)
        samp_d_infl: float
            Inflection point of the sampling depth sigmoid (nm) (default 2.2)
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
            (nm^2/s) (default 0.08)
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
            Perform calculations? (default True)

        Example
        -------
        >>> m1 = ADRMDP()
        calculating... . . . done
        >>> m1.plot_depth_prof()
        """
        samp_len = kwargs.get('samp_len', 20)  # (nm)
        n_x_steps = kwargs.get('n_x_steps', 500)

        time = kwargs.get('time', 4.5)  # (s)
        n_t_steps = kwargs.get('n_t_steps', 500)

        self._bound_cond = kwargs.get('bound_cond', 'neumann')
        if self._bound_cond == 'dirichlet':
            self._alpha_u = kwargs.get('alpha_u', [0, 0])
            self._alpha_v = kwargs.get('alpha_v', [0, 0])

        interf_slope = kwargs.get('interf_slope', 20)  # (1/nm)

        samp_d_slope = kwargs.get('samp_d_slope', 1.7)  # (1/nm)
        samp_d_infl = kwargs.get('samp_d_infl', 2.2)  # (nm)

        l_depth_u = kwargs.get('l_depth_u', [14])  # (nm)
        l_width_u = kwargs.get('l_width_u', [1])  # (nm)
        vel_u = kwargs.get('vel_u', 9)  # (nm/s)
        diff_ampl_u = kwargs.get('diff_ampl_u', 18)  # (nm^2/s)
        diff_slope_u = kwargs.get('diff_slope_u', 0.7)  # (1/nm)
        diff_x_infl_u = kwargs.get('diff_x_infl_u', 3.5)  # (nm)
        diff_const_u = kwargs.get('diff_const_u', 0.08)  # (nm^2/s)
        # small positive value needed for numerical stability

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

        T = time  # (s)
        self._N = n_t_steps  # dt = T/(N - 1)

        self._interf_slope = interf_slope/(1/self._samp_len)  # (1)

        samp_d_slope = samp_d_slope/(1/self._samp_len)  # (1)
        samp_d_infl = samp_d_infl/self._samp_len  # (1)

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

        self._dx = L/(self._J - 1)
        self._x_grid = np.array([j * self._dx for j in range(self._J)])

        self._dt = T/(self._N - 1)
        self._t_grid = np.array([n * self._dt for n in range(self._N)])

        self._samp_d_x = self._get_sigm_fun(samp_d_infl, samp_d_slope)

        d_term_x_u = self._get_sigm_fun(diff_x_infl_u, diff_slope_u)
        d_term_x_v = self._get_sigm_fun(diff_x_infl_v, diff_slope_v)

        d_term_x_deriv_u = self._get_sigm_fun(diff_x_infl_u, diff_slope_u,
                                              deriv=True)
        d_term_x_deriv_v = self._get_sigm_fun(diff_x_infl_v, diff_slope_v,
                                              deriv=True)

        self._d_term_total_u = diff_ampl_u * d_term_x_u + diff_u_const
        self._d_term_total_v = diff_ampl_v * d_term_x_v + diff_v_const

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
            return ampl * slope * np.exp(slope * (x - x_infl)) / \
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
        sigma_u = (self._d_term_total_u * self._dt)/(2 * self._dx**2)
        sigma_v = (self._d_term_total_v * self._dt)/(2 * self._dx**2)

        u_surf = np.average(U, weights=self._samp_d_x)
        v_surf = np.average(V, weights=self._samp_d_x)
        m_surf = np.average(M, weights=self._samp_d_x)

        if m_surf < 0:
            a = (self._a_u * u_surf + self._a_v * v_surf)/(u_surf + v_surf)
        else:
            a = self._a_u * u_surf + self._a_v * v_surf + self._a_m * m_surf

        rho_u = ((self._d_term_total_u_deriv - a) * self._dt) / (4 * self._dx)
        rho_v = ((self._d_term_total_v_deriv - a) * self._dt) / (4 * self._dx)

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

        self._U_xt = np.zeros((self._N, self._J))
        self._V_xt = np.zeros((self._N, self._J))
        self._M_xt = np.zeros((self._N, self._J))
        self._a_t = np.zeros(self._N)
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

        one_fourth = self._N/4
        for ti in range(1, self._N):
            U_new = np.linalg.solve(self._A_u, self._B_u.dot(U) +
                                    self._r_term_u * self._dt +
                                    self._dirich_term_u)
            V_new = np.linalg.solve(self._A_v, self._B_v.dot(V) +
                                    self._r_term_v * self._dt +
                                    self._dirich_term_v)

            U = U_new
            V = V_new

            self._U_xt[ti] = U
            self._V_xt[ti] = V
            M = 1 - U - V
            self._M_xt[ti] = M
            a = self._calc_matrices(U, V, M)
            self._a_t[ti] = a

            self._calc_r_terms(U, V, M)

            if ti % one_fourth == 0:
                print '.',

        print 'done'
    # =========================================================================

    def plot_samp_d(self):
        '''Plot sampling depth sigmoid.'''
        plt.plot(self._x_grid * self._samp_len, self._samp_d_x)
        plt.xlabel('x (nm)')
        plt.ylabel('s$_S$(x)')
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
            heatmap = plt.pcolormesh(x, t, y, vmin=-0.5, vmax=0,
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
                    y = self._U_xt[int(time/self._dt), :]
                if comp == 'v':
                    y = self._V_xt[int(time/self._dt), :]
                if comp == 'm':
                    y = self._M_xt[int(time/self._dt), :]
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
            x = self._t_grid * abs(self._a_t) * self._samp_len  # t -> x (nm)
        if var == 'time':
            x = self._t_grid

        for comp in comps:
            if comp == 'u':
                # u(t, x ~ 0)
                y = np.average(self._U_xt, axis=1, weights=self._samp_d_x)
            if comp == 'v':
                # v(t, x ~ 0)
                y = np.average(self._V_xt, axis=1, weights=self._samp_d_x)
            if comp == 'm':
                # m(t, x ~ 0)
                y = np.average(self._M_xt, axis=1, weights=self._samp_d_x)
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
                if comp == 'u':
                    for d, w in zip(self._l_depth_u, self._l_width_u):
                        plt.axvspan(d/abs(self._a_t.mean()),
                                    (d + w)/abs(self._a_t.mean()),
                                    color=col, alpha=0.2)
                if comp == 'v':
                    for d, w in zip(self._l_depth_v, self._l_width_v):
                        plt.axvspan(d/abs(self._a_t.mean()),
                                    (d + w)/abs(self._a_t.mean()),
                                    color=col, alpha=0.2)
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
            y = np.average(self._U_xt, axis=1, weights=self._samp_d_x)
        if comp == 'v':
            # v(t, x ~ 0)
            y = np.average(self._V_xt, axis=1, weights=self._samp_d_x)
        if comp == 'm':
            # m(t, x ~ 0)
            y = np.average(self._M_xt, axis=1, weights=self._samp_d_x)

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
# =============================================================================


def comp_samp_d(*args):
    '''Compare sampling depth sigmoids.

    e.g. args=m1, m2,...
    '''
    m1 = args[0]
    x = m1._x_grid * m1._samp_len
    for i, m in enumerate(args):
        plt.plot(x, m._samp_d_x, label=i + 1)
    plt.xlabel('x (nm)')
    plt.ylabel('s$_S$(x)')
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
    m1 = args[0]
    x = m1._x_grid * m1._samp_len
    for i, m in enumerate(args):
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
    m1 = args[0]
    x = m1._t_grid * abs(m1._a_t) * m1._samp_len  # t -> x (nm)
    for i, m in enumerate(args):
        if comp == 'u':
            # u(t, x ~ 0)
            y = np.average(m._U_xt, axis=1, weights=m._samp_d_x)
        if comp == 'v':
            # v(t, x ~ 0)
            y = np.average(m._V_xt, axis=1, weights=m._samp_d_x)
        if comp == 'm':
            # m(t, x ~ 0)
            y = np.average(m._M_xt, axis=1, weights=m._samp_d_x)

        col = plt.plot(x, y, label=i + 1)[0].get_color()
        if i == 0:
            if comp == 'u':
                for d, w in zip(m1._l_depth_u, m1._l_width_u):
                    plt.axvspan(d * m1._samp_len, (d + w) * m1._samp_len,
                                color=col, alpha=0.2)
            if comp == 'v':
                for d, w in zip(m1._l_depth_v, m1._l_width_v):
                    plt.axvspan(d * m1._samp_len, (d + w) * m1._samp_len,
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
