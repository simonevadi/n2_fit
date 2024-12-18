import numpy as np
from .fit import N2_fit

class N2FitBluesky:
    """
    A class that integrates N2 spectrum fitting capabilities into a Bluesky framework,
    allowing for spectroscopic analysis of data retrieved via a Bluesky database.

    Attributes:
        db (database connection): Connection to the database used to retrieve experimental data.
        n2fit (N2_fit): An instance of the N2_fit class for performing spectrum fitting.
    """

    def __init__(self, database):
        """
        Initializes the N2FitBluesky class with a database connection.

        Args:
            database: A database connection to retrieve scan data.
        """
        self.db = database
        self.n2fit = N2_fit()


    def _retrieve_spectra(self, identifier, motor=None, detector=None):
        """
        Retrieves spectral data from a database based on the given identifier, with optional motor and detector specifics.

        Args:
            identifier (str or int): The identifier for the data set, can be a scan ID or other unique identifier.
            motor (str, optional): Specifies the motor to retrieve data from. Defaults to the primary motor if not provided.
            detector (str, optional): Specifies the detector to retrieve data from. Defaults to the primary detector if not provided.

        Returns:
            tuple: A tuple containing two numpy arrays; the first for motor positions (x) and the second for detector counts (y).
        """       

        run = self.db[identifier]
        if detector == None:
            detector  = run.metadata['start']['detectors'][0]
        if motor == None:
            motor = run.metadata['start']['motors'][0]
        spectra   = run.primary.read()
        x    = np.array(spectra[motor])
        y = np.array(spectra[detector])

        return x, y
    
    def fit_n2(self, scan_id, n_peaks=5, motor='pgm.en', detector='kth01', title=None, 
               print_fit_results=False, fwhm_l=114):
        """
        Performs an N2 fit on the data retrieved for a given scan ID, motor, and detector.

        Args:
            scan_id (str or int): The scan identifier to retrieve data for fitting.
            n_peaks (int, optional): The number of peaks to fit. Defaults to 5.
            motor (str, optional): The motor to retrieve data from. Defaults to 'pgm.en'.
            detector (str, optional): The detector to retrieve data from. Defaults to 'kth01'.
            title (str, optional): The title for the fit plot and reports. Defaults to the scan ID if not provided.
            print_fit_results (bool, optional): If True, prints the fitting results. Defaults to False.
            fwhm_l (float, optional): Full width at half maximum for Lorentzian fits. Defaults to 114.

        Returns:
            None: This method does not return any value but outputs fitting results.
        """
        if '.' in motor:
            motor.replace('.', '_')
        if '.' in detector:
            detector.replace('.', '_')

        title = title if title is not None else scan_id

        energy, intensity = self._retrieve_spectra(scan_id)
        self.n2fit.fit_n2(energy, intensity, n_peaks=n_peaks, title=title, 
                          print_fit_results=print_fit_results, save_results=False, 
                          show_results=True, fwhm_l=fwhm_l,
                          motor=motor, detector=motor)
        
    def fit_n2_3peaks(self, scan_id, motor='pgm.en', detector='kth01', title=None, 
               print_fit_results=False, fwhm_l=114):
        """
        Performs an N2 fit with 3 peaks on the data retrieved for a given scan ID, using specified motor and detector.

        Args:
            scan_id (str or int): The scan identifier to retrieve data for fitting.
            motor (str, optional): The motor to retrieve data from. Defaults to 'pgm.en'.
            detector (str, optional): The detector to retrieve data from. Defaults to 'kth01'.
            title (str, optional): The title for the fit plot and reports. Defaults to the scan ID if not provided.
            print_fit_results (bool, optional): If True, prints the fitting results. Defaults to False.
            fwhm_l (float, optional): Full width at half maximum for Lorentzian fits. Defaults to 114.

        Returns:
            None: This method does not return any value but outputs fitting results for three peaks specifically.
        """
        if '.' in motor:
            motor.replace('.', '_')
        if '.' in detector:
            detector.replace('.', '_')

        title = title if title is not None else scan_id

        energy, intensity = self._retrieve_spectra(scan_id)
        
        self.n2fit.fit_n2_3peaks(energy, intensity,  
                                 print_fit_results=print_fit_results, 
                                 save_results=False, show_results=True, 
                                 fwhm_l=fwhm_l, motor=motor, detector=detector)