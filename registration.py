import numpy as np
from skimage.registration import phase_cross_correlation

def generate_slices(displacement: tuple[int, int]) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """
    Generates the slices that return the common area according to the displacement supplied.

    Parameters
    ----------
    displacement : np.ndarray
        Displacement of `dataset_2` with respect to `dataset_1`.
        
    Returns
    -------
    slice_tiple : tuple
        (slice of `dataset_1`, slice of `dataset_2`)
    """
    y_disp = int(displacement[1])
    x_disp = int(displacement[0])

    init_slice = (np.s_[0:y_disp] if y_disp < 0 else np.s_[y_disp:], 
                  np.s_[0:x_disp] if x_disp < 0 else np.s_[x_disp:])
    aftr_slice = (np.s_[0:-y_disp] if y_disp > 0 else np.s_[-y_disp:],
                  np.s_[0:-x_disp] if x_disp > 0 else np.s_[-x_disp:])
 
    return (init_slice, aftr_slice)


def match_topography_cost_function(displacement: np.ndarray, dataset_1: np.ndarray, dataset_2: np.ndarray) -> np.ndarray:
    """
    Returns the mean squared error (MSE) of the common area between the datasets provided for the given displacement.

    Parameters
    ----------
    displacement: np.ndarray
        Displacement of `dataset_2` with respect to `dataset_1`.
    dataset_1: np.ndarray
        Topography of the reference image.
    dataset_2: np.ndarray
        Topography of the displaced image.      
      
    Returns
    -------
    err : np.ndarray
        MSE cost. `np.ndarray` with a singular value.
    """
    dataset_1_slice, dataset_2_slice = generate_slices(displacement)
    return np.mean((dataset_1[dataset_1_slice] - dataset_2[dataset_2_slice])**2)


def match_image_by_hill_descent(
        diplacement_guess: tuple[int, int], 
        init_data: np.ndarray,
        aftr_data: np.ndarray, 
        iterations: int
    ) -> dict:
    """
    Uses greedy hill descent to find the translation required for registering `init_data` and `aftr_data`.

    Returns a dict with :
     - the predicted 'displacement' of the `aftr_data` with respect to `init_data`,
     - the 'path' followed to get to the minimum, and
     - the minimum 'cost'.

    Parameters
    ----------
    displacement_guess : tuple[int, int]
        A initial guess for the displacement, from where to start the hill descent.
    init_data : np.ndarray
        A ndarray of values representing the reference image.
    aftr_data : np.ndarray
        A ndarray of values representing the displaced image.
    iterations : int
        Maximum number of steps to take to find the minimum.

    Returns
    -------
    registration_result : dict
        See the description above.
    """

    descent_path = []
    disp_pos = np.array(diplacement_guess)
    descent_path.append(np.copy(disp_pos))
    kernel = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    
    for _ in range(iterations):
        err = np.array([
            match_topography_cost_function(disp_pos + step, init_data, aftr_data) 
            for step in kernel
        ])
        if err.argmin() == 0:
            break

        disp_pos += kernel[err.argmin(), :]
        descent_path.append(np.copy(disp_pos))

    final_cost = match_topography_cost_function(disp_pos, init_data, aftr_data)

    return {"displacement": disp_pos, "path": np.array(descent_path), "cost": final_cost}