from environment.blocks_ising2D import Ising2D
import numpy as np


class ParametersIsing2D:
    """
    Class used to hold parameters needed to initialise Ising2D located in blocks_ising2D.py

    Attributes
    ----------
    inv_c_charge : float
        The inverse central charge of the CFT. It can be set to 0 corresponding to the supergravity limit.
    spin_list_short_d : ndarray
        A NumPy array containing either [0] if the D[0,4] multiplet is present or [] if it isn't.
        No other values should be used.
    spin_list_short_b : ndarray
        A NumPy array containing a list of the B[0,2] multiplet spins. These must be even and given in increasing
        order without duplication.
    spin_list_long : ndarray
        A NumPy array containing a list of the L[0,0] long multiplet spins. These must be even and given in increasing
        order. Degeneracy of spins is allowed.
    ell_max : int
        Spin cutoff for the a_chi function in blocks.py.

    Notes
    -----
    No validation of the inputs is done.
    """

    def __init__(self):
        # ---Delta Model---
        self.delta_model = 1.

        # ---Spin partition---
        # Note: spins HAVE to be given in ascending order
        self.spin_list = np.array([0, 0, 2, 2, 2, 4, 4, 6, 6, 8, 10])
        
        self.delta_teor = np.array([4., 8., 2., 6., 10., 4., 8., 6., 10., 8., 10.])
        self.lambda_teor = np.array([1., 1/100., 1., 1/10., 1/1260., 1/10., 1/126., 1/126., 1/1716., 1/1716., 1/24310.])
        
        #self.delta_teor = np.array([2.7016236744005675,5.050856944348188,2.0213466735433836,4.2856177631124215,4.909299202733042,5.6414889781221405,8.328514525342179,7.605468108807205,8.542512893393454,8.315081625434248,10.22088919725358])
        #self.lambda_teor = np.array([0.9530206983000907,0.4039839938189997,0.6166258049859034,0.17948634587633205,0.04899791188878733,0.04378130084964199,0.011652024319895985,0.0021118376035766362,0.0026691862468548704,0.0026228588405235584,0.0005427505644773896])
        
        self.ell_max = 30  # Spin cutoff for the a_chi function in blocks.py. This MUST be an even number

        # ---Pre-generated conformal block lattice parameters---
        self.delta_start = np.array([0., 0., 1.8, 1.8, 1.8, 3.8, 3.8, 5.8, 5.8, 7.8, 9.8])

        self.delta_sep = 0.0005  # jump in weights between each lattice point
        self.delta_end_increment = 10.5 - self.delta_start  # maximum deltas are delta_start + delta_end_increment - delta_sep eg 35.7995

        # This is a list of the original 180 columns to delete from the '6d_blocks_spin*.csv' files
        self.z_kill_list = []
        # An example of a non-empty z_kill_list
        # self.z_kill_list = [3, 5, 7, 9, 10, 11, 12, 13, 15, 18, 19, 22, 26,
        #                     27, 28, 29, 30, 31, 32, 35, 37, 40, 43, 44, 46, 48,
        #                     49, 53, 57, 58, 60, 61, 63, 65, 68, 70, 73, 74, 75,
        #                     79, 80, 81, 82, 83, 86, 87, 91, 95, 96, 102, 104, 107,
        #                     113, 114, 115, 116, 119, 121, 122, 124, 129, 131, 132, 134, 136,
        #                     139, 141, 144, 145, 146, 147, 149, 151, 153, 154, 155, 156, 157,
        #                     158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
        #                     171, 172, 173, 174, 175, 176, 177, 178, 179]

        # DO NOT CHANGE ANYTHING BEYOND THIS POINT IN THIS CLASS
        # ---Non-User Adjustable Parameters---        
        self.num_of_operators = self.spin_list.size
        self.multiplet_index = [np.arange(self.num_of_operators)]


class ParametersIsing2D_SAC(ParametersIsing2D):
    """
    A subclass of ParametersSixD. It holds the parameters required to configure the soft-Actor-Critic algorithm.

    Attributes
    ----------
    filename_stem: str
        Stem of the filename where output is to be saved (.csv is appended automatically in code).
        Can be used to distinguish output from runs with different central charges e.g. sac_c25 or sac_cSUGRA.
    verbose : {'', 'e', 'o'}
        When the soft-Actor-Critic algorithm should print to the console:
        - ``: no output.
        - `e`: print everytime reward is recalculated.
        - `o`: only when faff_max is reached and a re-initialisation occurs.
    faff_max : int
        Maximum number of steps spent searching for an improved reward without success.
        When faff_max is reached a re-initialisation occurs.
        Higher value means algorithm spends more time searching for a better solution.
    pc_max : int
        Maximum number of re-initialisations before window size decrease.
        Higher value means algorithm spends more time searching for a better solution.
    window_rate : float
        Search window size decrease rate. Range is (0, 1).
        The window_rate multiplies the search window sizes so small values focus the search quickly.
    max_window_exp : int
        Maximum number of search window size decreases.
        The final window sizes will be equal to ( window_rate ** max_window_exp ) * guess_sizes.
    same_spin_hierarchy : bool
        This flag determines whether a minimum separation in scaling dimension of long operators of the same spin
        is enforced.
    dyn_shift : float
         The minimum separation in scaling dimension between long operators degenerate in spin.
    guessing_run_list_deltas : ndarray
        Controls the guessing mode status for each conformal weight datum.
        0 = non-guessing mode
        1 = guessing mode.
    guessing_run_list_opes : ndarray
        Controls the guessing mode status for each OPE-squared coefficient datum.
        0 = non-guessing mode
        1 = guessing mode
    guess_sizes_deltas : ndarray
        Initial size of search windows for the conformal weights. They need not all be the same value.
        The guess_sizes of short D and B multiplets should be set to 0 as their weights are fixed.
        There is an implicit upper bound set by the highest weight in the pregenerated conformal block csv files.
        They need not all be the same value.
    guess_sizes_opes : ndarray
        Initial size of search windows for the OPE-squared coefficients. They need not all be the same value.
    shifts_deltas : ndarray
        Lower bounds for the conformal weights. They need not all be the same value.
    shifts_opecoeffs : ndarray
        Lower bounds for the OPE-squared coefficients. They need not all be the same value.
    global_best : ndarray
        The CFT data to start the soft-Actor-Critic with.
        For a 'from scratch' run the values should be the same as guess_sizes_deltas and guess_sizes_opes.
    global_reward_start : float
        The initial reward to start with.
    action_space_N : ndarray
        The dimension of the search space, equal to twice the total number of operators.
    shifts : ndarray
        The concatenation of shifts_deltas and shifts_opes.
    guessing_run_list : ndarray
        The concatenation of guessing_run_list_deltas and guessing_run_list_opes.
    guess_sizes : ndarray
        The concatenation of guess_sizes_deltas and guess_sizes_opes.
    Notes
    -----
    The user should not modify the attributes action_space_N, shifts, guessing_run_list and guess_sizes.
    This subclass inherits the spin partition which must be defined in the class ParametersSixD.
    No validation of the inputs is done.
    """

    def __init__(self, config):
        super().__init__()
        
        
        # ---Output Parameters---
        self.filename_stem = 'sac'
        self.verbose = ''  # When the SAC algorithm should print to the console:
        # e - print at every step
        # o - only after a re-initialisation
        # default is '' which produces no output

        # ---Learn Loop Paramaters---
        self.faff_max = config['faff_max']  # maximum time spent not improving

        # ---Automation Run Parameters---
        self.pc_max = config['pc_max']  # max number of re-initialisations before window decrease
        self.window_rate = config['window_rate']  # window decrease rate (between 1 and 0)
        self.max_window_exp = config['max_window_exp']  # maximum number of window changes

        # ---Spin Hierachy Parameters---
        self.same_spin_hierarchy = config['same_spin_hierarchy']  # same long multiplet operators with the same spin should be ordered
        self.dyn_shift = config['dyn_shift']  # set the gap between long multiplet same spin deltas

        # ---Environment Parameters---
        # set guessing run list for conformal weights
        self.guessing_run_list_deltas = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        
        # set guessing run list for ope coefficients        
        self.guessing_run_list_opes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        
        self.reward_scale = config['reward_scale']
        self.delta_max = 10.5
        
        # initial search window size for conformal weights
        # windows for D and B multiplets should be set to zero as they are fixed
        
        # !!! Modified this to set the initial window to respect delta <= delta_max
        self.guess_sizes_deltas = np.array([10.5, 10.5, 8.5, 8.5, 8.5, 6.5, 6.5, 4.5, 4.5, 2.5, 0.5])
        
        # initial search window size for OPE coeffs        
        self.guess_sizes_opes = np.ones(self.num_of_operators)
        
        # set minimum values for conformal weights
        # minimums for D and B multiplets are fixed as weights are known
        self.shifts_deltas = np.array([0., 0., 2., 2., 2., 4., 4., 6., 6., 8., 10.])
    
        # set minimum values for OPE coeffs
        self.shifts_opecoeffs = np.zeros(self.num_of_operators)

        # ---Starting Point Parameters---
        # initial configuration to explore around
        # set equal to combination of shifts_deltas and shifts_opecoeffs to effectively start from a zero solution
        
        #print(self.guess_sizes_deltas)
        
        delta_start = self.shifts_deltas
        ope_start = self.shifts_opecoeffs
        #print(delta_start)
        #print(ope_start)
        self.global_best = np.concatenate((delta_start, ope_start))
        # initial reward to start with
        # set equal to 0.0 to start from a zero solution.
        self.global_reward_start = 0.0

        # ------------------------------------------------------
        # DO NOT CHANGE ANYTHING BEYOND THIS POINT IN THIS CLASS
        # ------------------------------------------------------
        # ---Non-User Adjustable Parameters---        
        self.action_space_N = 2 * self.num_of_operators                      
        self.shifts = np.concatenate((self.shifts_deltas, self.shifts_opecoeffs))
        self.guessing_run_list = np.concatenate((self.guessing_run_list_deltas,
                                                 self.guessing_run_list_opes))
        self.guess_sizes = np.concatenate((self.guess_sizes_deltas, self.guess_sizes_opes))
        self.output_order = ''  # The order to arrange the data saved to csv file
        # default order is (index_array, reward, cft data)
        # 'Reversed' - order is (cft data, reward, index_array)
