import numpy as np
import pandas as pd

from scipy.special import erf
from scipy.special import wofz

def clean_data(x, y):
    """
    Removes NaN and negative values from 'y' and corresponding elements in 'x'.

    Parameters:
        x (np.array): Array of x values.
        y (np.array): Array of y values.

    Returns:
        np.array: Cleaned x values.
        np.array: Cleaned y values.
    """
    # Remove NaN values from 'y'
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]

    # Remove negative values from 'y'
    mask = y >= 0
    x, y = x[mask], y[mask]

    return x, y
    
def convert_to_json_serializable(data):
    if isinstance(data, dict):
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int64, np.int32, np.float64, np.float32)):
        return float(data)
    return data


# Function to handle None values for calculations and formatting
def format_val(value): 
    return f"{np.round(value, 2):>10}" if value is not None else '         -'


def find_closest_fwhm(filename, target_vp_ratio, target_gamma):
    # Load the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Failed to read the CSV file: {e}")
        return None
    
    # First, find the Gamma value closest to the target
    df['Gamma Diff'] = np.abs(df['FWHM_l (meV)'] - target_gamma)
    closest_gamma = df.loc[df['Gamma Diff'].idxmin(), 'FWHM_l (meV)']
    
    # Filter the DataFrame to only include rows with the closest Gamma
    # Here, we make an explicit copy to avoid SettingWithCopyWarning when modifying the DataFrame
    gamma_filtered_df = df[df['FWHM_l (meV)'] == closest_gamma].copy()
    
    if gamma_filtered_df.empty:
        print(f"No data found for the closest Gamma = {closest_gamma}.")
        return None
    
    # Now find the row with the closest VP Ratio within the filtered DataFrame
    gamma_filtered_df['VP Ratio Diff'] = np.abs(gamma_filtered_df['3P1V Ratio'] - target_vp_ratio)
    closest_row = gamma_filtered_df.loc[gamma_filtered_df['VP Ratio Diff'].idxmin()]
    
    # Return the FWHM_g value from the closest row
    return closest_row['FWHM_g (meV)']

def remove_neg_values(x,y):
    """
    Remove negative values from y and corresponding values from x

    Parameters
    ----------
    x : numpy.array
    y : numpy.array

    Return
    --------
    x,y : np.array
        two arrays without negative values
    """
    ind = np.where(y < 0)
    x = np.delete(x,ind[0])
    y = np.delete(y,ind[0])
    return x,y

def calculate_rms(residuals):
    """
    Calculate the Root Mean Square (RMS) of residuals.

    Parameters:
    - residuals (np.array): Array of residuals (observed - predicted values).

    Returns:
    - float: The RMS of the residuals.
    """
    return np.sqrt(np.mean(np.square(residuals)))

def find_nearest_idx(array, value):
    """
    Find the index of the values in array closer to value

    Parameters
    ----------
    array : numpy.array
    value : int or float     


    Return
    --------
    idx : int
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def evaluate_line(x, slope, intercept):
    """
    Evaluates the y-value of a linear equation given x, slope, and intercept.

    Parameters:
    - x (float): The x-coordinate at which to evaluate the line.
    - slope (float): The slope of the line.
    - intercept (float): The y-intercept of the line.

    Returns:
    - y (float): The computed y-value corresponding to the given x.

    Example:
    Given the line equation y = 2x + 1, calling evaluate_line(2, 2, 1) will return 5.
    """
    y = slope * x + intercept
    return y

def estimate_line_parameters(x_points, y_points):
    """
    Estimates the slope and y-intercept of a line given two points.

    Parameters:
    - x_points (list or array): A list or array containing two x-coordinates.
    - y_points (list or array): A list or array containing two y-coordinates.

    Returns:
    - slope (float): The estimated slope of the line.
    - intercept (float): The estimated y-intercept of the line.
    
    Raises:
    - ValueError: If the lists or arrays do not contain exactly two elements.
    """
    if len(x_points) != 2 or len(y_points) != 2:
        raise ValueError("Exactly two x and two y points are required to estimate line parameters.")
    
    # Calculate the slope (m = (y2 - y1) / (x2 - x1))
    delta_x = x_points[1] - x_points[0]
    delta_y = y_points[1] - y_points[0]
    if delta_x == 0:
        raise ValueError("Delta x cannot be zero for slope calculation.")
    slope = delta_y / delta_x
    
    # Calculate the intercept (b = y - mx)
    intercept = y_points[0] - slope * x_points[0]
    
    return slope, intercept


def calculate_skewed_voigt_amplitude(center, sigma, gamma, skew, desired_peak_intensity):
    """
    Estimate the amplitude for a Skewed Voigt peak to achieve a desired peak intensity.
    
    Parameters:
    - center (float): The center position of the peak.
    - sigma (float): The Gaussian width (standard deviation) of the peak.
    - gamma (float): The Lorentzian width (half-width at half-maximum) of the peak.
    - skew (float): The skewness parameter, affecting the asymmetry of the peak.
    - desired_peak_intensity (float): The desired intensity at the peak position.
    
    Returns:
    - float: The estimated amplitude necessary to achieve the desired peak intensity.
    """
    # Calculate the skewed Voigt profile at the center
    s2 = np.sqrt(2)
    z = (center - center + 1j * gamma) / (sigma * s2)
    voigt_profile = wofz(z).real / (sigma * np.sqrt(2 * np.pi))
    
    # Apply skewness
    beta = skew / (sigma * s2)
    skew_factor = 1 + erf(beta * (center - center))
    peak_intensity = voigt_profile * skew_factor
    
    # Calculate the required amplitude to achieve the desired peak intensity
    required_amplitude = desired_peak_intensity / peak_intensity

    return required_amplitude

def guess_amp(x,y,vc):
    """
    Guess the amplitude for to input in skewd voigt model starting from
    real data x and y, at the value of x closer to vc

    Parameters
    ----------
    x : numpy.array
    y : numpy.array
    vc : int or float     


    Return
    --------
    amp : float
    """
    idx = find_nearest_idx(x,vc)
    amp = y[idx]
    return amp

def find_max_around_theoretical(x, y, theoretical_center, fwhm):
    """
    Finds the maximum value in a window defined by the FWHM around a theoretical center value in a dataset.

    Parameters
    ----------
    x : numpy.array
        Array of x values, typically energy or wavelength.
    y : numpy.array
        Array of y values, typically intensity or counts.
    theoretical_center : float
        The theoretical (expected) center value of the peak.
    fwhm : float
        The Full Width at Half Maximum used to define the search window.

    Return
    --------
    max_x : float
        The x value at which the maximum occurs.
    max_y : float
        The intensity at the maximum.
    max_idx : int
        The index of the maximum value.
    """
    # Determine the window range around the theoretical center
    half_width = fwhm / 2
    min_val = theoretical_center - half_width
    max_val = theoretical_center + half_width

    # Find indices within this range
    indices = (x >= min_val) & (x <= max_val)

    # Extract the portion of y within this range and find the maximum
    if np.any(indices):
        windowed_y = y[indices]
        max_idx_within_window = np.argmax(windowed_y)
        # Convert local index within window to global index
        max_idx_global = np.where(indices)[0][max_idx_within_window]
        max_x = x[max_idx_global]
        max_y = y[max_idx_global]
        return max_x, max_y, max_idx_global
    else:
        raise ValueError("No data points found within the specified FWHM range around the theoretical center.")

def find_first_max(x,y,fwhm):
    """
    Finds the first max in a Nitrogen spectra. The routine performs a window scan 
    of the array starting from left of the array until two conditions are met:
        the new max values < old max value
        old max value > 0.8

    Parameters
    ----------
    x : numpy.array
    y : numpy.array
    fwhm : float
            the fwhm of the nitrogen peak, used to decide how
            wide the scan window is

    Return
    --------
    idy : int
            the position of the first max in x
    """
    step    = x[1]-x[0]
    ind     = int(fwhm/step/1) if int(fwhm/step/1)>0 else 1
    n_steps = int(x.shape[0]/ind)
    for i in range(n_steps):
        if i == 0:
            amax    = np.max(y[i*ind:i*ind+ind])
            argmax  = np.argmax(y[i*ind:i*ind+ind])
        else:
            tmax    = np.max(y[i*ind:i*ind+ind])
            targmax = np.argmax(y[i*ind:i*ind+ind]) +i*ind
            if tmax <= amax and amax > 0.8:
                break
            if tmax >= amax:
                amax = tmax
                argmax = targmax           
    return x[argmax], y[argmax], argmax

def extract_bandwidth_and_rp(out, first_peak=400.76, n_peaks=3):
    sigma = 0
    for index in range(1,n_peaks+1):
        sigma += out.params[f'v{index}_sigma']
    sigma /= n_peaks
    fwhm = 2.35*sigma
    rp = first_peak/fwhm
    return np.round(fwhm*1000,2), int(rp)


def extract_RP_ratio(x, y, fit):
    """
    Calculate the 3rd valley to 1st peak ratio using the fitted parameters.

    Parameters:
    - x (np.array): Array of x values used in the fit.
    - y (np.array): Array of y values used in the fit.
    - fit (ModelResult): The result from an lmfit fitting process, or a Parameters object from a fit.

    Returns:
    - vp_ratio (float): The valley-to-peak ratio calculated from the fit.
    """

    # Check if fit is ModelResult or Parameters and extract parameters accordingly
    if hasattr(fit, 'params'):
        params = fit.params  # If fit is a ModelResult, use .params
    else:
        params = fit  # If fit is already Parameters object

    # Extract centers of peaks from parameters
    cen_v1 = params['v1_center'].value
    cen_v2 = params['v2_center'].value
    cen_v3 = params['v3_center'].value

    # Generate a finely spaced x array for evaluating the fit results more precisely
    energy_fine = np.arange(x[0], x[-1], 0.001)

    # Evaluate the model intensity at the fine energy points
    if hasattr(fit, 'eval'):
        intensity_fine = fit.eval(x=energy_fine)  # If fit is a ModelResult
    else:
        # If you only have Parameters without the model, you would need the model to evaluate
        raise AttributeError("Need a ModelResult object to evaluate the fit, not just Parameters.")

    # Find indices for cen_v1, cen_v2, cen_v3 in the fine energy array
    idx_v1 = np.argmin(np.abs(energy_fine - cen_v1))
    idx_v2 = np.argmin(np.abs(energy_fine - cen_v2))
    idx_v3 = np.argmin(np.abs(energy_fine - cen_v3))

    # Find the minimum intensity (valley) between cen_v1 and cen_v2
    first_valley_intensity = np.min(intensity_fine[idx_v1:idx_v2])
    first_valley_intensity_arg = idx_v1+np.argmin(intensity_fine[idx_v1:idx_v2])
    v1_x = energy_fine[first_valley_intensity_arg]

    # Get the intensity (peak) at cen_v3
    peak_v3_intensity = np.max(intensity_fine[idx_v3- 10: idx_v3+10])
    peak_v3_intensity_arg = np.argmax(intensity_fine[idx_v3- 50: idx_v3+50])
    p3_x = energy_fine[idx_v3- 50+peak_v3_intensity_arg]

    slope = params['lin_slope']
    intercept = params['lin_intercept']
    valley_line = evaluate_line(first_valley_intensity_arg, slope, intercept)
    peak_line = evaluate_line(idx_v3, slope, intercept)
    # print(f'energy_fine {energy_fine[0]}-{energy_fine[-1]}')
    # print(f'first_valley_intensity_arg {first_valley_intensity_arg}')
    # print('slope and intercept line', slope, intercept)
    # print('valley and peak line', valley_line, peak_line)
    


    # Calculate the valley-to-peak ratio
    vp_ratio = (first_valley_intensity-valley_line) / (peak_v3_intensity-peak_line)
    vp = {'valley':(v1_x, first_valley_intensity), 
          'peak':(p3_x, peak_v3_intensity)}

    return vp_ratio, vp

def extract_RP_ratio_for_table(x, y, params):
    """
    Calculate the 3rd valley to 1st peak ratio using the fitted parameters.

    Parameters:
    - x (np.array): Array of x values used in the fit.
    - y (np.array): Array of y values used in the fit.
    - fit (ModelResult): The result from an lmfit fitting process, or a Parameters object from a fit.

    Returns:
    - vp_ratio (float): The valley-to-peak ratio calculated from the fit.
    """


    # Extract centers of peaks from parameters
    cen_v1 = params['v1_center'].value
    cen_v2 = params['v2_center'].value
    cen_v3 = params['v3_center'].value

    # Generate a finely spaced x array for evaluating the fit results more precisely
    energy_fine = np.arange(x[0], x[-1], 0.01)


    # Find indices for cen_v1, cen_v2, cen_v3 in the fine energy array
    idx_v1 = np.argmin(np.abs(energy_fine - cen_v1))
    idx_v2 = np.argmin(np.abs(energy_fine - cen_v2))
    idx_v3 = np.argmin(np.abs(energy_fine - cen_v3))

    # Find the minimum intensity (valley) between cen_v1 and cen_v2
    first_valley_intensity = np.min(y[idx_v1:idx_v2])

    # Get the intensity (peak) at cen_v3
    peak_v3_intensity = y[idx_v3]

    # Calculate the valley-to-peak ratio
    vp_ratio = first_valley_intensity / peak_v3_intensity

    return vp_ratio