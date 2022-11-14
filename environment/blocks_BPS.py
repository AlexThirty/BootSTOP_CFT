import numpy as np
import mpmath as mp
import scipy.special as sc
from numpy import linalg as LA

def get_C_BPS(g):
    if g==0.:
        return 1.
    elif g=='inf':
        return 2.
    else:
        return (3*sc.iv(1, 4*np.pi*g)/(2*(np.pi)**2 * g**2 * (sc.iv(2, 4*np.pi*g))**2)) * ((2*(np.pi)**2 * g**2 + 1)*sc.iv(1, 4*np.pi*g) - 2*np.pi*g*sc.iv(0, 4*np.pi*g)) - 1

def zero_lambdas(delta):
    return (sc.gamma(delta+3)*sc.gamma(delta+1)*(delta-1))/(2*sc.gamma(2*delta+2))

class BPS:
    def __init__(self, params, z_data):
        self.multiplet_index = params.multiplet_index
        self.action_space_N = params.action_space_N
        self.chi = z_data.z
        self.delta_max = params.delta_max
        self.g = params.g
        self.verbose = params.verbose
        self.reward_scale = params.reward_scale
        
        if self.g == 0.:
            self.best_theoretical_reward = self.compute_g0_reward()
        elif self.g == 'inf':
            self.best_theoretical_reward = self.compute_ginf_reward()
        else: 
            self.best_theoretical_reward = 0.

        self.C_BPS = get_C_BPS(self.g)

        self.delta_sep = params.delta_sep
        self.delta_start = params.delta_start
        self.delta_end_increment = params.delta_end_increment

        self.env_shape = z_data.env_shape
        
    def compute_g0_reward(self):
        vector = []
        for chiel in self.chi:
            vector.append((1-chiel)**2 * chiel**2 * (1/chiel - 1/(1-chiel)) + chiel**2 * (1-chiel)**2 * (1/(1-chiel) - 1(chiel)))
        return 1/LA.norm(vector)
    
    def compute_ginf_reward(self):
        vector = []
        for chiel in self.chi:
            vector.append(((1-chiel)**2 * (chiel + chiel**2/(chiel-1)) + chiel**2 * (1-chiel + (1-chiel)**2/(-chiel))))
        return 1/LA.norm(vector)
        
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
        return vector
    
    def compute_single_reward(self, delta: np.array, lambdads: np.array):
        vector = self.compute_BPS_vector(delta, lambdads, self.chi)
        return 1/LA.norm(vector)
    
    
    
class BPS_SAC(BPS):

    def __init__(self, params, z_data):
        super().__init__(params, z_data)

        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy  # impose weight separation flag
        self.dyn_shift = params.dyn_shift  # the weight separation value
        self.dup_list = np.ones(params.num_of_operators)  # which long spins are degenerate

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
        
        spin_cons = self.compute_BPS_vector(delta_dict['all'], ope_dict['all'], self.chi)
        # long_cons.shape = (num_of_long, env_shape)

        # add up all the components
        constraints = spin_cons  # the .sum implements summation over multiplet spins
        reward = 1 / LA.norm(constraints)

        return constraints, reward, cft_data
