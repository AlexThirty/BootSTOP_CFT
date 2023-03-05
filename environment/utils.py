import numpy as np
from csv import writer


def generate_block_list(max_spin, z_kill_list):
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    block_list = []
    for i in range(0, max_spin + 2, 2):
        tmp_name = 'block_lattices/6d_blocks_spin' + str(i) + '.csv'
        tmp = np.genfromtxt(tmp_name, delimiter=',')
        # if the kill list is empty append the whole array
        if len(z_kill_list) == 0:
            block_list.append(tmp)
        # otherwise delete the columns which appear in z_kill_list and then append
        else:
            block_list.append(np.delete(tmp, z_kill_list, axis=1))

    print('Done loading pregenerated conformal block data.')
    return block_list


def output_to_file(file_name, output, mode='a'):
    """
    Appends row of output to a file.

    Parameters
    ----------
    file_name : str
        Filename of a writer object.
    output : iterable of strings or numbers
        The parameter passed to writer.writerow.

    """
    with open(file_name, mode, newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(output)
        f_object.close()


def output_to_console(output):
    """
    Print to the console.

    Parameters
    ----------
    output : str
        String printed to the console.
    """
    print(output)


def generate_Ising2D_block_list(max_spin, z_kill_list, sigma=False):
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    block_list = []
    for i in range(0, max_spin + 2, 2):
        if sigma:
            tmp_name = 'block_lattices/ising2D_blocks_spin_sigma' + str(i) + '.csv'
        else:
            tmp_name = 'block_lattices/ising2D_blocks_spin' + str(i) + '.csv'
        tmp = np.genfromtxt(tmp_name, delimiter=',')
        # if the kill list is empty append the whole array
        if len(z_kill_list) == 0:
            block_list.append(tmp)
        # otherwise delete the columns which appear in z_kill_list and then append
        else:
            block_list.append(np.delete(tmp, z_kill_list, axis=1))

    print('Done loading pregenerated conformal block data.')
    return block_list

def generate_BPS_block_list(g_index):
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    block_list = []
    for delta in range(1, 11, 1):
        tmp_name = 'BPS_precalc/blocks_delta' + str(delta) + '.csv'
        tmp = np.genfromtxt(tmp_name, delimiter=',')
        # if the kill list is empty append the whole array
        block_list.append(tmp[g_index].reshape((-1)))

    print('Done loading pregenerated conformal block data.')
    return block_list

def generate_BPS_int1_list(g_index):
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    int1_list = []
    for delta in range(1, 11, 1):
        tmp_name = 'BPS_precalc/integral1_delta' + str(delta) + '.csv'
        tmp = np.genfromtxt(tmp_name, delimiter=',')
        # if the kill list is empty append the whole array
        int1_list.append(tmp[g_index])

    print('Done loading pregenerated conformal block data.')
    return int1_list

def generate_BPS_int2_list(g_index):
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    int2_list = []
    for delta in range(1, 11, 1):
        tmp_name = 'BPS_precalc/integral2_delta' + str(delta) + '.csv'
        tmp = np.genfromtxt(tmp_name, delimiter=',')
        # if the kill list is empty append the whole array
        int2_list.append(tmp[g_index])

    print('Done loading pregenerated conformal block data.')
    return int2_list



def generate_BPS_block_list_free():
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    tmp_name = 'BPS_precalc_free/blocks.csv'
    block_list = np.genfromtxt(tmp_name, delimiter=',')

    print('Done loading pregenerated conformal block data.')
    return block_list

def generate_BPS_int1_list_free():
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    tmp_name = 'BPS_precalc_free/integral1.csv'
    integral = np.genfromtxt(tmp_name, delimiter=',')

    print('Done loading pregenerated conformal block data.')
    return integral

def generate_BPS_int2_list_free():
    """
    Reads the pregenerated conformal blocks data from csv files into a list.

    Parameters
    ----------
    max_spin : int
        The upper bound of spin for loading conformal blocks.
    z_kill_list : list
        A list of z-point positions to remove.
    Returns
    -------
    block_list : list
        A list of ndarrays containing pregenerated conformal block data.
    """
    # since this takes a long time give the user some feedback
    print('Loading pregenerated conformal block data.')
    tmp_name = 'BPS_precalc_free/integral2.csv'
    integral = np.genfromtxt(tmp_name, delimiter=',')

    print('Done loading pregenerated conformal block data.')
    return integral