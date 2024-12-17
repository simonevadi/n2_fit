import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .models import N2SkewedVoigtModel
from .helper_functions import extract_RP_ratio_for_table

from .n2_peaks_parameters import theoretical_centers, voigt_intensities, first_peak, theoretical_intensities

class CreateTable:
    """
    A class dedicated to creating tables for analyzing the 3rd-peak to 1st-valley ratio
    in spectral data using various fitting models.

    Attributes:
        theoretical_centers (np.array): Theoretical center positions for spectral peaks.
        first_peak (float): Position of the first peak in the spectrum, used as a reference.
        theoretical_intensities (np.array): Theoretical intensities of the peaks based on Voigt profiles.
        voigt_intensities (np.array): Modeled intensities of the peaks using the Skewed Voigt model.
        models (Models): An instance of a modeling class to access various spectral fitting functions.
    """
    def __init__(self, model='SkewedVoigt'):
        """
        Initializes the CreateTable instance with predetermined spectral characteristics and the specified model.

        Parameters:
            model (str): Specifies the model to use for spectral fitting. Currently, only 'SkewedVoigt' is implemented.

        Raises:
            NotImplementedError: If a model other than 'SkewedVoigt' is specified.
        """
        if model == 'SkewedVoigt':
            self.model = N2SkewedVoigtModel()
        else:
            raise NotImplemented('No other Model is Implemented')
        
    def create_table(self, savepath:str=None, fwhm_g:np.array=None, fwhm_l:np.array=None):
        """
        Generates a table of resolving power (RP) and 3rd-peak to 1st-valley (3P1V) ratios across a range of
        Full Width at Half Maximum (FWHM) for Gaussian and Lorentzian widths in spectral data analysis.

        Parameters:
            savepath (str, optional): The path where the resulting table will be saved as a CSV file. If None, the table
                                      is not saved to disk.
            fwhm_g (np.array, optional): Array of Gaussian widths (in eV) to simulate. Defaults to a range of 0.01 to 0.15 eV
                                         with a step of 0.001 eV if not provided.
            fwhm_l (np.array, optional): Array of Lorentzian widths (in meV) to simulate. Defaults to a range of 55 to 60.1 meV
                                         with a step of 0.5 meV if not provided.

        Returns:
            pd.DataFrame: A DataFrame containing the FWHM Lorentzian (in meV), FWHM Gaussian (in meV), resolving power,
                          and 3P1V ratios for the simulated conditions.
        """
        if fwhm_g is None:
            fwhm_g = np.arange(0.01, .15, 0.001)  
        else:
            fwhm_g = fwhm_g.astype(float) / 1000.0  

        if fwhm_l is None:
            gamma = np.arange(0.055, 0.0601, 0.0005)  
        else:
            gamma = fwhm_l.astype(float) / 2 /1000.0  

        
        results = []
        total_iterations = len(gamma) * len(fwhm_g)
        progress_bar = tqdm(total=total_iterations, desc="Creating Table")

        for g in gamma:
            for f in fwhm_g:
                dict_fit = self.model.prepare_param_for_table(fwhm_g=f, gamma=g)
                mod, pars = self.model.make_model(dict_fit)
                energy = np.arange(theoretical_centers[0]-1, theoretical_centers[-1]+1, 0.01)
                intensity = mod.eval(pars, x=energy)
                vp_ratio = extract_RP_ratio_for_table(energy, intensity, pars)
                rp = int(first_peak / f)
                results.append({'FWHM_l (meV)': np.round(2*g*1000,2),
                                'FWHM_g (meV)': np.round(f*1000, 2),
                                'RP': rp, '3P1V Ratio': vp_ratio})
                progress_bar.update(1)  # Update the progress bar per iteration

        progress_bar.close()

        # Convert results to DataFrame and save as CSV
        df_results = pd.DataFrame(results)
        if savepath is not None:
            df_results.to_csv(savepath, index=False)
        return df_results    

    def plot_table(self, table_to_plot='tables/table.csv', save_path=False, show_plot=False):
        """
        Plots the relationship between the resolving power (RP) and the third peak to first valley ratio (3P1V Ratio),
        categorized by different Lorentzian Full Width at Half Maximum (FWHM) values from a specified table.

        Parameters:
            table_to_plot (str): Path to the CSV file that contains the simulation data for plotting.
                                Default is 'tables/table.csv'.
            save_path (str, optional): The file path where the plot image will be saved. If None, the plot is not saved to disk.
                                    Providing a path will automatically save the plot to that location.
            show_plot (bool): A flag to determine whether to display the plot in the UI. Default is False.

        Returns:
            None: The function directly plots the graph or saves it to a file depending on the input parameters.

        Raises:
            FileNotFoundError: If the specified table_to_plot does not exist.
            Exception: General exceptions related to plotting errors or data issues, which might need further investigation.

        This function reads the provided CSV file to extract data, then plots the resolving power against the 3P1V Ratio
        for each unique Lorentzian width defined in the 'FWHM_l (meV)' column of the DataFrame. Each Lorentzian width is
        represented as a separate line in the plot to visually discern the impact of Lorentzian width on the spectral analysis.
        """
        # Read the data from the CSV file
        df = pd.read_csv(table_to_plot)

        # Create a figure and axis for the plot
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        # Iterate over each unique gamma value and plot
        for gamma in df['FWHM_l (meV)'].unique():
            # Filter the DataFrame by gamma
            gamma_data = df[df['FWHM_l (meV)'] == gamma]
            
            # Plot FWHM vs RP for each gamma
            ax.plot(gamma_data['3P1V Ratio'], gamma_data['RP'],
                    linestyle='-',label=f'FWHM_l: {gamma} meV')

        # Setting plot labels and titles
        ax.set_xlabel('3rdPeak/1st valley ratio', fontsize=12)
        ax.set_ylabel('Resolving Power', fontsize=12)
        ax.set_title('3rdPeak/1st valley ratio vs Resolving Power', fontsize=14)
        ax.legend(title='Lorentian FWHM', fontsize=10, title_fontsize=12)

        # Display the plot
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show_plot:
            plt.show()

        return ax  # Optional, in case you need to further manipulate the Axes object outside the function



















