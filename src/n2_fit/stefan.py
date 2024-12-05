import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import wofz

# Voigt-Profil-Funktion mit festen Lorentz-Breiten
def voigt_fixed_gamma(x, center, sigma, height, gamma=0.06):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return height * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

# ASCII-Datei einlesen
file_path = "belchem_n2.txt"  # Pfad zur ASCII-Datei anpassen
data = np.loadtxt(file_path)
energy = data[:, 0]
intensity = data[:, 1]

# Peaks automatisch finden
peaks, _ = find_peaks(
    intensity,
    height=0.04 * np.max(intensity),  # Mindesthöhe des Peaks
    distance=5  # Mindestabstand zwischen Peaks
)
peak_positions = energy[peaks]

# Initiale Parameter für den Gesamt-Fit
initial_params = [0.05]  # Gemeinsame Gauß-Breite als erster Parameter
for peak in peak_positions:
    height_guess = intensity[np.abs(energy - peak).argmin()]  # Höhe an der Peak-Position
    initial_params.extend([peak, height_guess])

# Fit der Summe aller Voigt-Profile mit gleicher Gauß-Breite
def voigt_sum_shared_sigma(x, sigma, *params):
    num_peaks = len(params) // 2
    total = np.zeros_like(x)
    for i in range(num_peaks):
        center = params[2 * i]
        height = params[2 * i + 1]
        total += voigt_fixed_gamma(x, center, sigma, height, gamma=0.060)
    return total

try:
    popt, _ = curve_fit(
        lambda x, *params: voigt_sum_shared_sigma(x, *params),
        energy,
        intensity,
        p0=initial_params
    )
except RuntimeError:
    print("Der globale Fit der Summe ist fehlgeschlagen.")
    popt = initial_params

# Feiner Energiebereich für glattere Kurven
fine_energy = np.linspace(energy.min(), energy.max(), 5000)

# Einzelne Voigt-Profile aus den Fit-Parametern rekonstruieren
sigma_shared = popt[0]  # Gemeinsame Gauß-Breite
fitted_profiles = []
gamma_fixed = 0.06  # Lorentz-Breite ist konstant
for i in range(len(popt[1:]) // 2):
    center = popt[2 * i + 1]
    height = popt[2 * i + 2]
    profile = voigt_fixed_gamma(fine_energy, center, sigma_shared, height, gamma=gamma_fixed)
    fitted_profiles.append((fine_energy, profile, (center, sigma_shared, height)))

# Berechnung der Auflösung deltaE
mean_gauss_fwhm = 2.35482 * sigma_shared  # Mittelwert der Gauß-FWHM-Werte (nur ein Wert in diesem Fall)
deltaE = 400 / mean_gauss_fwhm  # Auflösung
# Berechnung des Verhältnisses r
# 1. Erstes Minimum: suchen zwischen ersten beiden Peaks
if len(peaks) >= 2:
    peak1_pos = peaks[0]
    peak2_pos = peaks[1]
    region_between_peaks = (energy > energy[peak1_pos]) & (energy < energy[peak2_pos])
    local_min_index = np.argmin(intensity[region_between_peaks])  # Index des Minimums im Bereich
    first_min = np.where(region_between_peaks)[0][local_min_index]
    height_first_min = intensity[first_min]
else:
    height_first_min = np.nan

# 2. Drittes Maximum: suchen nach drittem Peak
if len(peaks) >= 3:
    third_max = peaks[2]
    height_third_max = intensity[third_max]
else:
    height_third_max = np.nan

# Verhältnis r
if not np.isnan(height_first_min) and not np.isnan(height_third_max):
    r = height_first_min / height_third_max
else:
    r = np.nan


# Plot der Messdaten, Voigt-Profile und Summe
plt.figure(figsize=(12, 8))
plt.plot(energy, intensity, label="Originaldaten", color="black")

# Einzelprofile und Summe explizit berechnen
total_fit = np.zeros_like(fine_energy)
colors = plt.cm.tab10.colors
for i, (energy_profile, profile, params) in enumerate(fitted_profiles):
    center, sigma, height = params

    # FWHM-Berechnungen
    fwhm_gauss = 2.35482 * sigma
    fwhm_lorentz = 2 * gamma_fixed

    plt.plot(
        energy_profile, profile, linestyle="--",
        label=f"Voigt-Profil {i+1} (Peak {center:.2f} eV)", 
        color=colors[i % len(colors)]
    )
    # Texte mit FWHM-Werten hinzufügen
    plt.text(
        center, height * 0.6, 
        f"FWHM (Gauß): {fwhm_gauss:.2f} eV\nFWHM (Lorentz): {fwhm_lorentz:.2f} eV", 
        color=colors[i % len(colors)],
        fontsize=9,
        horizontalalignment="center"
    )
    # Summe der Einzelprofile berechnen
    total_fit += profile

# Plotte die Summe aller Einzelprofile
plt.plot(fine_energy, total_fit, label="Summe aller Voigt-Profile", color="red", linewidth=2)

# Text mit der Auflösung deltaE hinzufügen
print(f'FWHM_G: {np.round(2.35*sigma_shared*1000,2)} meV')
print(f'FWHM_L: {np.round(fwhm_lorentz*1000,2)} meV')
print(f'RP: {deltaE}')
plt.text(
    0.70, 0.62, 
    f"E/$\\Delta$E: ~{deltaE:.0f}", 
    color="green", 
    fontsize=12,
    transform=plt.gca().transAxes
)
# Text mit dem Verhältnis r hinzufügen
plt.text(
    0.70, 0.67, 
    f"ratio r (1. Min / 3. Max): {r:.2f}", 
    color="blue", 
    fontsize=12,
    transform=plt.gca().transAxes
)
# Plot-Titel und Achsenbeschriftungen
plt.title("Messwerte, einzelne Voigt-Profile und ihre Summe")
plt.xlabel("Energie (eV)")
plt.ylabel("Intensität (a.u.)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('stefan.pdf')
plt.show()
