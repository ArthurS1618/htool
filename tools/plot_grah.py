import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import argparse
import math

# Input
parser = argparse.ArgumentParser()
parser.add_argument("--inputfile", type=str)
parser.add_argument("--title", type=str, default="Tracé des données du fichier texte", help="Titre du graphique")
parser.add_argument("--xlabel", type=str, default="size", help="Libellé de l'axe des x")
parser.add_argument("--ylabel", type=str, default="Time", help="Libellé de l'axe des y")
parser.add_argument("--scale", type=str, default="N", help="echelle log ou pas")


args = parser.parse_args()
inputfile = args.inputfile
title = args.title
xlabel = args.xlabel
ylabel = args.ylabel
scale = args.scale


X = []
Y = []
for k in range (1,  50):
    X.append ( 200*(k)) 
    Y.append ((200*k)*math.log(200*k)**2/1000)

# Data
df = pd.read_csv(inputfile, header=None)
if ( scale== 'N'):
    plt.plot(X, df.values[0, :])  # Utilise la première ligne comme données à tracer

else :
    plt.loglog(X, df.values[0, :]) 

# plt.loglog(X, Y)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.show()