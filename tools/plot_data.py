import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import argparse

# Configuration de l'argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--inputfile_input", type=str, required=True, help="Chemin vers le fichier CSV d'entrée")
parser.add_argument("--inputfile_output", type=str, required=True, help="Chemin vers le fichier CSV de sortie")
parser.add_argument("--scale", type=str, default="linear", choices=["linear", "log"], help="Échelle des axes (linear ou log)")
parser.add_argument("--xtitle", type=str, default="", help="Titre de l'axe X")
parser.add_argument("--ytitle", type=str, default="", help="Titre de l'axe Y")


# Lecture des arguments
args = parser.parse_args()


# Définir le chemin de base
base_path = "../build_change_internal_structure_06_2024/tests/functional_tests/hmatrix/hmatrix_factorization/"

# Construire les chemins complets pour les fichiers
inputfile_input_path = os.path.join(base_path, args.inputfile_input)
inputfile_output_path = os.path.join(base_path, args.inputfile_output)

# Lecture des fichiers CSV
df_input = pd.read_csv(args.inputfile_input)
df_output = pd.read_csv(args.inputfile_output)

# Supposons que les colonnes x et y soient présentes dans les deux fichiers
x = df_input['x'].values
y = df_output['y'].values

# Création de la figure
plt.figure()

# Configuration de l'échelle
if args.scale == "log":
    plt.xscale('log')
    plt.yscale('log')

# Tracé des données
plt.plot(x, y)

# Ajout des titres des axes
plt.xlabel(args.xtitle)
plt.ylabel(args.ytitle)

# Affichage du graphique
plt.show()