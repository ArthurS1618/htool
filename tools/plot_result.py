import matplotlib.pyplot as plt
import csv
import argparse
import numpy as np

def lire_csv(nom_fichier):
    with open(nom_fichier, mode='r') as fichier:
        lecteur = csv.reader(fichier)
        return [float(valeur) for ligne in lecteur for valeur in ligne]

parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", type=int, required=True)
parser.add_argument("--eta", type=int, required=True)

args = parser.parse_args()

epsilon = args.epsilon
eta = args.eta

path = "../build_change_internal_structure_06_2024/tests/functional_tests/hmatrix/hmatrix_factorization/"

# Lire les vecteurs depuis les fichiers CSV
vecteur1 = lire_csv(path + 'size_test_hlu_eps' + str(epsilon) + '_eta_' + str(eta) + '.csv')
vecteur2 = lire_csv(path + 'err_test_hlu_eps' + str(epsilon) + '_eta_' + str(eta) + '.csv')
vecteur3 = lire_csv(path + 'time_test_hlu_eps' + str(epsilon) + '_eta_' + str(eta) + '.csv')
vecteur4 = lire_csv(path + 'comprl_test_hlu_eps' + str(epsilon) + '_eta_' + str(eta) + '.csv')
vecteur5 = lire_csv(path + 'compru_test_hlu_eps' + str(epsilon) + '_eta_' + str(eta) + '.csv')

nlogn = vecteur1*(np.log(vecteur1)**3)/100000
print(len(vecteur2))
# Tracer vecteur 2 en fonction de vecteur 1
plt.figure()
plt.plot(vecteur1, vecteur2, label='error(size)')
plt.xlabel('size')
plt.ylabel('error')
plt.title('Error solve $L_\mathcal{H}U_\mathcal{H}x = y$')
plt.yscale('log') 
plt.xscale('log') 

plt.legend()
plt.grid(True)
plt.show()

# Tracer vecteur 3 en fonction de vecteur 1
plt.figure()
plt.plot(vecteur1, vecteur3, label='$time_{HLU}$(size)')
plt.plot(vecteur1, nlogn,  label='$n\log(n)^3$')
plt.xlabel('size')
plt.ylabel('time(s)')
plt.title('Time HLU factorization')
plt.yscale('log') 
plt.xscale('log') 


plt.legend()
plt.grid(True)
plt.show()

# Tracer vecteur 4 en fonction de vecteur 1
plt.figure()
plt.plot(vecteur1, vecteur4, label='compr$_{L_\mathcal{H}}$(size)')
plt.xlabel('size')
plt.ylabel('compr$_{L_\mathcal{H}}$')
plt.title('compression $L_\mathcal{H}$')
plt.xscale('log') 

plt.legend()
plt.grid(True)
plt.show()

# Tracer vecteur 5 en fonction de vecteur 1
plt.figure()
plt.plot(vecteur1, vecteur5, label='compr$_{U_\mathcal{H}}$(size)')
plt.xlabel('size')
plt.ylabel('compr$_{U_\mathcal{H}}$')
plt.title('compression $U_\mathcal{H}$')
plt.xscale('log') 

plt.legend()
plt.grid(True)
plt.show()