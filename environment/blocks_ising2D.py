import numpy as np
import mpmath as mp
import scipy.special as sc
from numpy import linalg as LA

class Ising2D:
    def __init__(self, block_list, params, z_data):
        self.use_scipy_for_hypers = True
        self.delta_model = params.delta_model
        self.s = params.spin_list
        self.multiplet_index = params.multiplet_index
        self.action_space_N = params.action_space_N
        self.delta_start = params.delta_start
        self.delta_end_increment = params.delta_end_increment
        self.delta_sep = params.delta_sep
        self.z = z_data.z
        self.block_list = block_list
        self.z_conj = z_data.z_conj
        self.delta_max = params.delta_max
        self.delta_teor = params.delta_teor
        self.lambda_teor = params.lambda_teor
        self.verbose = params.verbose
        self.reward_scale = params.reward_scale
        self.delta_sep = params.delta_sep
        self.delta_start = params.delta_start
        self.delta_end_increment = params.delta_end_increment

        self.env_shape = z_data.env_shape

        self.use_scipy_for_hypers = True  # compute the hypergeometrics using scipy package or mpmath
        
    def g1d(self, a: np.array, x):
        if self.use_scipy_for_hypers:
            return (x**a) * sc.hyp2f1(a, a, 2*a, x)
        else:
            return (x**a) * mp.hyp2f1(a, a, 2*a, x)

    def g2d(self, delta: np.array, s: np.array, z, zbar):
        return (1 / (1 + (s == 0).astype(float))) * ((self.g1d((delta+s)/2, z) * self.g1d((delta-s)/2, zbar)) +
                                                     (self.g1d((delta+s)/2, zbar) * self.g1d((delta-s)/2, z)))

    def ising2d(self, delta: np.array, s: np.array, lambdads: np.array, z, zbar, delta_model: float):
        blocks = lambdads.reshape(-1,1) * self.precalc_val_array(delta)
        return ((1-z) * (1-zbar))**(delta_model) - (z*zbar)**(delta_model) + np.sum(blocks, axis=0)
    
    def precalc_val(self, delta, spin):
        # Get the spin index
        spin_index = spin // 2
        # Get the index for delta in the precalculated data
        delta = np.clip(delta, a_min=None, a_max=self.delta_start[spin_index] + self.delta_end_increment - self.delta_sep)
        n = np.rint((delta - self.delta_start[spin_index]) / self.delta_sep)
        val = self.block_list[spin_index][int(n)]
        return np.transpose(val)
    
    def precalc_val_array(self, deltas):
        """
        Aggregates all the long multiplet contributions together into a single array.

        Returns
        -------
        long_c : ndarray(num_of_long, env_shape)
        """
        vals = self.precalc_val(deltas[0], self.s[0])
        for x in range(1, deltas.size):
            vals = np.vstack((vals, self.precalc_val(deltas[x], self.s[x])))
        return vals
    
    def ising2d_precalc(self, delta: np.array, s: np.array, z, zbar, delta_model: float):
        val = (((1-z) * (1-zbar))**(delta_model) * self.g2d(delta, s, z, zbar) -
                         (z*zbar)**(delta_model) * self.g2d(delta, s, 1-z, 1-zbar))
        return val
    
    def ising2d_old(self, delta: np.array, s: np.array, lambdads: np.array, z, zbar, delta_model: float):
        blocks = lambdads * (((1-z) * (1-zbar))**(delta_model) * self.g2d(delta, s, z, zbar) -
                         (z*zbar)**(delta_model) * self.g2d(delta, s, 1-z, 1-zbar))

        return ((1-z) * (1-zbar))**(delta_model) - (z*zbar)**(delta_model) + np.sum(blocks)
    
    def compute_ising2d_vector(self, delta: np.array, s: np.array, lambdads: np.array, z: np.array, zbar: np.array, delta_model: float):
        vector = self.ising2d(delta, s, lambdads, z, zbar, delta_model).real
        return vector
    
    def compute_ising2d_vector_recalc(self, delta: np.array, s: np.array, lambdads: np.array, z: np.array, zbar: np.array, delta_model: float):
        vector = []
        for zel, zbarel in zip(z, zbar):
            vector.append(self.ising2d_old(delta, s, lambdads, zel, zbarel, delta_model).real)
        return vector
    
    def compute_single_reward(self, delta: np.array, lambdads: np.array):
        vector = self.compute_ising2d_vector(delta, self.s, lambdads, self.z, self.z_conj, self.delta_model)
        return 1/LA.norm(vector)
    
    def best_theoretical(self):
        vector = self.compute_ising2d_vector(self.delta_teor, self.s, self.lambda_teor, self.z, self.z_conj, self.delta_model)
        return 1/LA.norm(vector)
    
    def best_theoretical_recalc(self):
        vector = self.compute_ising2d_vector_recalc(self.delta_teor, self.s, self.lambda_teor, self.z, self.z_conj, self.delta_model)
        return 1/LA.norm(vector)
    
    
class Ising2D_SAC(Ising2D):

    def __init__(self, block_list, params, z_data):
        super().__init__(block_list, params, z_data)

        self.same_spin_hierarchy_deltas = params.same_spin_hierarchy  # impose weight separation flag
        self.dyn_shift = params.dyn_shift  # the weight separation value
        self.dup_list = self.s == np.roll(self.s, -1)  # which long spins are degenerate

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
        
        spin_cons = self.compute_ising2d_vector(delta_dict['all'], self.s, ope_dict['all'], self.z, self.z_conj, self.delta_model)
        # long_cons.shape = (num_of_long, env_shape)

        # add up all the components
        constraints = spin_cons  # the .sum implements summation over multiplet spins
        cross = LA.norm(constraints)
        reward = 1 / LA.norm(constraints)

        return constraints, reward, cft_data, cross
    
    def crossing_recalc(self, cft_data):
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
        
        spin_cons = self.compute_ising2d_vector_recalc(delta_dict['all'], self.s, ope_dict['all'], self.z, self.z_conj, self.delta_model)
        # long_cons.shape = (num_of_long, env_shape)

        # add up all the components
        constraints = spin_cons  # the .sum implements summation over multiplet spins
        reward = 1 / LA.norm(constraints)

        return constraints, reward, cft_data
