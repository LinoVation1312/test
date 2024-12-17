# -*- coding: utf-8 -*-
"""
Outil interactif d'analyse acoustique amélioré
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import streamlit as st

# Configuration de l'application Streamlit
st.title("Outil interactif d'analyse acoustique")
st.sidebar.title("Configuration des paramètres")

# Paramètres personnalisés via l'interface
thicknesses = np.array(st.sidebar.multiselect(
    "Choisissez les épaisseurs (mm)",
    options=[10, 20, 30, 40, 50],
    default=[10, 20, 30]
))

densities = np.array(st.sidebar.multiselect(
    "Choisissez les densités (kg/m³)",
    options=[50, 75, 110, 150, 200],
    default=[75, 110, 150, 200]
))

frequencies = np.array([100, 500, 1000, 2000])  # Gardé fixe pour simplification

# Exemple de données d'absorption (réajustées si nécessaire)
absorption_data = np.array([
    [[0.2, 0.4, 0.6, 0.8],
     [0.25, 0.45, 0.65, 0.85],
     [0.3, 0.5, 0.7, 0.9]],
    [[0.4, 0.6, 0.8, 0.9],
     [0.45, 0.65, 0.85, 0.95],
     [0.5, 0.7, 0.9, 0.95]],
    [[0.6, 0.8, 0.9, 0.95],
     [0.65, 0.85, 0.95, 0.98],
     [0.7, 0.9, 0.97, 0.99]]
])

def update_interpolator(thicknesses, densities, frequencies, absorption_data):
    """
    Met à jour l'interpolateur en fonction des épaisseurs et densités sélectionnées
    """
    # Ajuste la forme des données d'absorption pour correspondre aux sélections
    absorption_data_selected = absorption_data[:len(thicknesses), :len(densities), :len(frequencies)]

    # Crée un nouvel interpolateur avec les dimensions mises à jour
    interpolator = RegularGridInterpolator(
        (thicknesses, densities, frequencies),
        absorption_data_selected,
        bounds_error=False,
        fill_value=None
    )
    return interpolator

# Mise à jour de l'interpolateur après chaque modification
interpolator = update_interpolator(thicknesses, densities, frequencies, absorption_data)

def generate_contour_optimized(f_target, thicknesses, densities):
    """
    Génère l'abaque avec des calculs vectorisés pour optimiser les performances.
    """
    Ep_grid, Density_grid = np.meshgrid(
        np.linspace(thicknesses.min(), thicknesses.max(), 100),
        np.linspace(densities.min(), densities.max(), 100),
        indexing="ij"
    )
    
    alpha_values = np.arange(0.0, 1.05, 0.05)
    Z = np.zeros((len(alpha_values), *Ep_grid.shape))
    
    for idx, alpha_target in enumerate(alpha_values):
        interp_values = interpolator((Ep_grid, Density_grid, f_target))
        Z[idx] = np.abs(interp_values - alpha_target)
    
    return Ep_grid[:, 0], Density_grid[0], Z, alpha_values

def plot_contour_dynamic(Ep_grid, Density_grid, Z, alpha_values, f_target):
    """
    Affichage des abaques dynamiques dans Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, alpha_target in enumerate(alpha_values):
        tolerance = 0.003  # Tolérance pour les courbes d'isovaleurs
        contour = ax.contour(
            Density_grid, Ep_grid, Z[idx], levels=[tolerance], linewidths=1.5
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt=f'Alpha ≈ {alpha_target:.2f}')
    
    ax.set_title(f"Abaque des épaisseurs et densités pour {f_target} Hz")
    ax.set_xlabel("Densité (kg/m³)")
    ax.set_ylabel("Épaisseur (mm)")
    ax.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(fig)

# Curseur pour la fréquence
f_target = st.slider("Fréquence cible (Hz)", min_value=int(frequencies.min()), max_value=int(frequencies.max()), value=1000, step=100)

# Vérification des plages
if len(thicknesses) < 2 or len(densities) < 2:
    st.warning("Veuillez sélectionner au moins 2 valeurs d'épaisseurs et densités pour générer l'abaque.")
else:
    # Mise à jour de l'interpolateur après modification des paramètres
    interpolator = update_interpolator(thicknesses, densities, frequencies, absorption_data)

    # Génération et affichage
    Ep_grid, Density_grid, Z, alpha_values = generate_contour_optimized(f_target, thicknesses, densities)
    plot_contour_dynamic(Ep_grid, Density_grid, Z, alpha_values, f_target)
