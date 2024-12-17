# N2 Fitting and Resolving Power Evaluation

## Overview
This repository provides a Python-based tool for analyzing and fitting nitrogen spectra data, estimating the resolving power (RP), and calculating the third-peak to first-valley (3P1V) ratio. The package uses the **Skewed Voigt model** for fitting and is built on top of the `lmfit` library.

## Features
- Automatic fitting of nitrogen spectra.
- Resolving power (RP) estimation based on Gaussian contributions.
- Calculation of the 3P1V ratio.
- Customizable initial parameters for fitting.
- Supports fixed or free fit parameters.
- Interactive visualization of spectral data and fits.

## Installation
To use the package, clone the repository and install the necessary dependencies:

```bash
# Clone the repository
git clone https://github.com/simonevadi/n2_fit.git
cd n2_fit

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- numpy
- pandas
- matplotlib
- lmfit
- tqdm

## Usage
### Basic Fit Example
The `fit_n2` method accepts spectral data and fits it using the **Skewed Voigt model**:

```python
from fit import N2_fit

# Example usage
n2fit = N2_fit()
energy = [list_of_energy_values]
intensity = [list_of_intensity_values]

n2fit.fit_n2(energy, intensity, title="MySpectrum", print_fit_results=True)
```

### Options for `fit_n2`
- **energy**: List or array of energy values.
- **intensity**: List or array of intensity values.
- **title** *(str)*: Title for the plot.
- **print_fit_results** *(bool)*: Prints the lmfit fit report.
- **n_peaks** *(int)*: Number of peaks to fit (default: 5).
- **save_results** *(bool/str)*: Saves results to a file if a path is provided.
- **show_results** *(bool)*: Displays the fit plot interactively.
- **fwhm_l** *(float)*: Lorentzian FWHM value (default: 114).

### Resolving Power Calculation
The RP is computed using the average Gaussian FWHM of the first three peaks:

```math
RP = \frac{\mu_2}{FWHM_g}
```

Where the Gaussian FWHM is:

```math
FWHM_g = 2 \sqrt{2 \ln(2)} \sigma_2
```

### 3rd-Peak to 1st-Valley Ratio
The 3P1V ratio is calculated and returned alongside the fit results, providing insight into spectral properties.The resolving power is also looked up in table, and this vaue is more reliable that the one from the fit. 

## Visualizations
The package includes tools for plotting:
- **Initial fit guesses**: Visualize the initial parameters used for fitting.
- **Fitting results**: Displays the data, fitted curve, and individual components.

Example:
```python
n2fit.fit_n2(energy, intensity, plot_initial_guess=True)
```

## Table Generation
Use the `CreateTable` class to generate tables of RP and 3P1V ratios across a range of Gaussian and Lorentzian FWHM values:

```python
from create_table import CreateTable

# Generate and save the table
table_generator = CreateTable()
df_results = table_generator.create_table(savepath="results_table.csv")
```

### Plot Table Results
```python
table_generator.plot_table(table_to_plot="results_table.csv", show_plot=True)
```

## Directory Structure
```
N2_Fit/
├── fit.py               # Main fitting class
├── models.py            # Skewed Voigt and background models
├── create_table.py      # Resolving power table generation
├── helper_functions.py  # Utility functions
├── n2_peaks_parameters.py # Default theoretical values for peaks
├── tables/              # Default tables (e.g., skewedVoigt.csv)
├── examples/            # Example scripts and usage
└── requirements.txt     # Dependencies
```

## Contributing
Contributions are welcome! If you'd like to improve the package, submit a pull request or open an issue.

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your description here"
   ```
4. Push the changes and open a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


