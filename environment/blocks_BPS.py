import numpy as np
import mpmath as mp
import scipy.special as sc
from numpy import linalg as LA
import scipy.integrate as si

class BPS:
    def __init__(self, params, z_data, block_list, integral_list_1, integral_list_2):
        self.multiplet_index = params.multiplet_index
        self.action_space_N = params.action_space_N
        self.chi = z_data.z
        self.delta_max = params.delta_max
        self.g = params.g
        self.verbose = params.verbose
        self.reward_scale = params.reward_scale
        self.integral_mode = params.integral_mode
        self.w1 = params.w1
        self.w2 = params.w2
        
        self.Curvature = params.Curvature
        self.C_BPS = self.get_C_BPS(self.g)
        self.F = self.get_F(self.g)
        self.B = self.get_B(self.g)
        self.RHS_1 = self.get_RHS_1()
        self.RHS_2 = self.get_RHS_2()
        self.f_BPS_chi = self.f_BPS(self.chi)
        self.f_BPS_inv = self.f_BPS(1-self.chi)
        
        self.delta_sep = params.delta_sep
        self.delta_start = params.delta_start
        self.delta_end_increment = params.delta_end_increment
        
        self.block_list = np.array(block_list)
        self.integral_list_1 = np.array(integral_list_1).reshape((-1))
        self.integral_list_2 = np.array(integral_list_2).reshape((-1))

        self.env_shape = z_data.env_shape
    
    def get_C_BPS(self, g):
        if g==0.:
            return 1.
        elif g=='inf':
            return 2.
        else:
            factor1 = (3*sc.iv(1, 4*np.pi*g)/(2*(np.pi)**2 * g**2 * (sc.iv(2, 4*np.pi*g))**2))
            factor2 = ((2*(np.pi)**2 * g**2 + 1)*sc.iv(1, 4*np.pi*g) - 2*np.pi*g*sc.iv(0, 4*np.pi*g))
            return factor1*factor2 - 1
        
    def get_F(self, g):
        return 1. + self.C_BPS
    
    def get_B(self, g):
        B = g/np.pi * (sc.iv(2, 4*np.pi*g)/sc.iv(1, 4*np.pi*g))
        return B
    
    def get_RHS_1(self):
        RHS_1 = (self.B - 3*self.Curvature)/(8*self.B**2) + (7*np.log(2)-41/8)*(self.F-1) + np.log(2)
        return RHS_1
    
    def get_RHS_2(self):
        RHS_2 = (1-self.F)/6 + (2-self.F)*np.log(2) + 1 - self.Curvature/(4*self.B**2)
        return RHS_2

    def get_teor_lambdas(self, delta):
        if self.g==0.:
            return (sc.gamma(delta+3)*sc.gamma(delta+1)*(delta-1))/(2*sc.gamma(2*delta+2))
        elif self.g=='inf':
            return (sc.gamma(delta+3)*sc.gamma(delta+1)*(delta-1))/(sc.gamma(2*delta+2))
        else:    
            return -1
    
    def compute_g0_function(self):
        vector = self.chi**2 * (1/self.chi - 1/(1-self.chi))
        return vector
    
    def compute_ginf_function(self):
        vector = self.chi + self.chi**2/(self.chi-1)
        return vector
    
    def f_BPS(self, chi):
        return chi * (1 - sc.hyp2f1(1, 2, 4, chi))

    def f_delta(seld, delta: np.array, chi):
        return (chi**(delta+1)/(1-delta)) * sc.hyp2f1(delta+1, delta+2, 2*delta+4, chi)

    def BPS_blocks(self, delta: np.array, lambdads: np.array, chi):
        blocks_chi = lambdads * self.f_delta(delta, chi)
        blocks_inv = lambdads * self.f_delta(delta, 1-chi)      
        return (1-chi)**2 * (chi + self.C_BPS*self.f_BPS(chi) + np.sum(blocks_chi)) + (chi)**2 * ((1-chi) + self.C_BPS*self.f_BPS(1-chi) + np.sum(blocks_inv))
    
    def compute_BPS_vector(self, delta: np.array, lambdads: np.array, chi: np.array):
        vector = []
        for chiel in chi:
            vector.append(self.BPS_blocks(delta, lambdads, chiel).real)
        return np.array(vector)
    
    def compute_test_vector(self, delta: np.array, chi: np.array):
        vector = []
        lambdas = self.get_teor_lambdas(delta=delta)
        for chiel in chi:
            vec = chiel + self.get_C_BPS(self.g)*self.f_BPS(chiel)
            for i, delta_el in enumerate(delta):
                if delta_el==1.:
                    vec = vec - (chiel**2) * sc.hyp2f1(2, 3, 6, chiel) * (sc.gamma(4)*sc.gamma(2)/(2*sc.gamma(4)))
                else:
                    vec = vec + lambdas[i]*self.f_delta(delta_el, chiel)
            vector.append(vec)
        return vector
    
    def compute_single_reward(self, delta: np.array, lambdads: np.array):
        vector = self.compute_BPS_vector(delta, lambdads, self.chi)
        return 1/LA.norm(vector)
    
    def integrand_1(self, x, delta):
        integrand = (x - 1 - x**2) * self.f_delta(delta, x)/(x**2) * (1 - 2*x)/(x - x**2)
        return -integrand
    
    def integrand_2(self, x, delta):
        integrand = (self.f_delta(delta, x)*(2*x - 1))/(x**2)
        return integrand
    
    def integral_1(self, delta):
        integral, err = si.quad(func=self.integrand_1, a=0, b=0.5, args=(delta,))
        return integral
    
    def integral_2(self, delta):
        integral, err = si.quad(func=self.integrand_2, a=0, b=0.5, args=(delta,))
        return integral
    
    def calc_constraint_1(self, delta, lambdads):
        vector = []
        for el in delta:
            vector.append(self.integral_1(el))
        return abs(np.sum(lambdads * vector) + self.RHS_1)
    
    def calc_constraint_2(self, delta, lambdads):
        vector = []
        for el in delta:
            vector.append(self.integral_2(el))
        return abs(np.sum(lambdads * vector) + self.RHS_2)
    
    def precalc_block(self, delta):
        block_chi = self.f_delta(delta, self.chi)
        block_inv = self.f_delta(delta, 1-self.chi)
        return (1 - self.chi)**2 * block_chi + self.chi**2 * block_inv
    
    def get_precalc_vector(self, lambdads):
        blocks = self.block_list * lambdads[:, np.newaxis]
        res = np.sum(blocks, axis=0) + (1 - self.chi)**2 * (self.chi + self.C_BPS*self.f_BPS(self.chi)) + (self.chi)**2 * ((1-self.chi) + self.C_BPS * self.f_BPS(1-self.chi))
        return res.real
    
    def get_precalc_constraint_1(self, lambdads):
        constr = self.integral_list_1 * lambdads
        return abs(np.sum(constr) + self.RHS_1)
    
    def get_precalc_constraint_2(self, lambdads):
        constr = self.integral_list_2 * lambdads
        return abs(np.sum(constr) + self.RHS_2)
    
class BPS_SAC(BPS):

    def __init__(self, params, z_data, block_list, integral_list_1, integral_list_2):
        super().__init__(params, z_data, block_list, integral_list_1, integral_list_2)

        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy  # impose weight separation flag
        self.dyn_shift = params.dyn_shift  # the weight separation value
        self.dup_list = np.ones(params.num_of_operators)  # which long spins are degenerate
        self.best_rew = 0.
    def split_cft_data(self, cft_data):
        """
        Sets up dictionaries to decompose the search space data into easily identifiable pieces.

        Parameters
        ----------
        cft_data : ndarray
            The array to be split.

        Returns
        -------
        delta_dict : dict
            A dictionary containing the keys ("short_d", "short_b", "long") and values of the conformal weights.
        ope_dict : dict
            A dictionary containing the keys ("short_d", "short_b", "long") and values of the OPE-squared coefficients.

        """
        delta_dict = {
            "all": cft_data[self.multiplet_index[0]]
        }
        ope_dict = {
            "all": cft_data[self.multiplet_index[0] + self.action_space_N // 2]
        }
        return delta_dict, ope_dict

    def impose_weight_separation(self, delta_dict):
        """
        Enforces a minimum conformal dimension separation between long multiplets of the same spin by
        overwriting values of delta_dict.

        Parameters
        ----------
        delta_dict : dict
            A dictionary of multiplet types and their conformal weights.
        Returns
        -------
        delta_dict : dict
            Dictionary with modified values for 'long' key.
        """
        deltas = delta_dict['all']
        flag_current = False
        flag_next = False
        for i in range(self.dup_list.size):
            flag_current = self.dup_list[i]
            flag_next_tmp = False

            if flag_next and not flag_current:
                deltas[i] = np.clip(deltas[i], a_min=(deltas[i - 1] + self.dyn_shift), a_max=None)

            if flag_current and not flag_next:
                flag_next_tmp = True

            if flag_current and flag_next:
                deltas[i] = np.clip(deltas[i], a_min=(deltas[i - 1] + self.dyn_shift), a_max=None)
                flag_next_tmp = True

            flag_next = flag_next_tmp

        return delta_dict

    def crossing(self, cft_data):
        """
        Evaluates the truncated crossing equations for the given CFT data at all points in the z-sample simultaneously.

        Parameters
        ----------
        cft_data : ndarray
            An array containing the conformal weights and OPE-squared coefficients of all the multiplets.

        Returns
        -------
        constraints : ndarray
            Array of values of the truncated crossing equation.
        reward : float
            The reward determined from the constraints.
        cft_data : ndarray
            A list of possibly modified CFT data.

        """
        # get some dictionaries
        delta_dict, ope_dict = self.split_cft_data(cft_data)

        if self.same_spin_hierarchy_deltas:
            # impose the mimimum conformal weight separations between operators
            delta_dict = self.impose_weight_separation(delta_dict)
            # since we've altered some data we update the long multiplet weights in cft_data
            cft_data[self.multiplet_index[0]] = delta_dict['all']

        # broadcast the reshaped long multiplet ope coefficients over their crossing contributions
        #spin_cons = ope_dict['all'].reshape(-1, 1) * self.compute_ising2d_vector(delta_dict['all'])
        
        constraints = self.compute_BPS_vector(delta_dict['all'], ope_dict['all'], self.chi)
        # long_cons.shape = (num_of_long, env_shape)

        # add up all the components
        if self.integral_mode == 0:
            reward = 1 / LA.norm(constraints)
        elif self.integral_mode == 1:
            const_1 = self.calc_constraint_1(delta_dict['all'], ope_dict['all'])
            reward = 1/ LA.norm(constraints) + self.w1 / const_1
        elif self.integral_mode == 2:
            const_1 = self.calc_constraint_1(delta_dict['all'], ope_dict['all'])
            const_2 = self.calc_constraint_2(delta_dict['all'], ope_dict['all'])
            reward = 1/ LA.norm(constraints) + self.w1 / const_1 + self.w2 / const_2
        return constraints, reward, cft_data
    
    def crossing_precalc(self, cft_data):
        """
        Evaluates the truncated crossing equations for the given CFT data at all points in the z-sample simultaneously.

        Parameters
        ----------
        cft_data : ndarray
            An array containing the conformal weights and OPE-squared coefficients of all the multiplets.

        Returns
        -------
        constraints : ndarray
            Array of values of the truncated crossing equation.
        reward : float
            The reward determined from the constraints.
        cft_data : ndarray
            A list of possibly modified CFT data.

        """
        # get some dictionaries
        delta_dict, ope_dict = self.split_cft_data(cft_data)

        if self.same_spin_hierarchy_deltas:
            # impose the mimimum conformal weight separations between operators
            delta_dict = self.impose_weight_separation(delta_dict)
            # since we've altered some data we update the long multiplet weights in cft_data
            cft_data[self.multiplet_index[0]] = delta_dict['all']

        # broadcast the reshaped long multiplet ope coefficients over their crossing contributions
        #spin_cons = ope_dict['all'].reshape(-1, 1) * self.compute_ising2d_vector(delta_dict['all'])
        
        constraints = self.get_precalc_vector(ope_dict['all'])
        # long_cons.shape = (num_of_long, env_shape)
        cross = LA.norm(constraints)
        const_1 = 0.
        const_2 = 0.

        # add up all the components
        if self.integral_mode == 0:
            reward = 1 / cross
        elif self.integral_mode == 1:
            const_1 = self.get_precalc_constraint_1(ope_dict['all'])
            #reward = 1/ LA.norm(constraints) + self.w1 / const_1
            reward = 1/(cross + self.w1*const_1)
            if reward > self.best_rew:
                self.best_rew = reward
                #print(f'Base reward: {1/LA.norm(constraints)}, reward from constraint_1: {1/const_1}')
                print(f'Base norm: {cross}, constraint_1: {const_1}')
        elif self.integral_mode == 2:
            const_1 = self.get_precalc_constraint_1(ope_dict['all'])
            const_2 = self.get_precalc_constraint_2(ope_dict['all'])
            reward = 1/(cross + self.w1 * const_1 + self.w2 * const_2)
            if reward > self.best_rew:
                self.best_rew = reward
                #print(f'Base reward: {1/LA.norm(constraints)}, reward from constraint_1: {1/const_1}')
                print(f'Base norm: {cross}, constraint_1: {const_1}, constraint_2: {const_2}')
        return constraints, reward, cft_data, cross, const_1, const_2 
