// #include "/work/sauniear/Documents/MCGproject/mcgproject-master/src/MCGS/BCSR/BCSRMatrix.h"
#include <cmath>
#include <fstream>
#include <htool/basic_types/vector.hpp>
#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/interface.hpp>
#include <htool/hmatrix/lrmat/fullACA.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <htool/wrappers/wrapper_lapack.hpp>
#include <iostream>
#include <mpi.h>
#include <regex>
#include <string>

using namespace std;
using namespace htool;

std::vector<std::vector<double>> get_csr(const std::string &matfile) {
    std::ifstream file(matfile);
    std::vector<double> data_csr;
    std::vector<double> col_csr;
    std::vector<double> row_csr;
    std::vector<std::vector<double>> res;

    if (!file.is_open()) {
        std::cerr << "Error: Can't open the file" << std::endl;
        return res;
    }

    std::string line;
    std::string word;
    char delimiter = ',';

    // Read first line
    if (getline(file, line)) {
        std::stringstream ss(line);
        while (getline(ss, word, delimiter)) {
            double d1 = std::stod(word);
            data_csr.push_back(d1);
        }
    }

    // Read second line
    if (getline(file, line)) {
        std::stringstream ss(line);
        while (getline(ss, word, delimiter)) {
            double d1 = std::stod(word);
            col_csr.push_back(d1);
        }
    }

    // Read third line
    if (getline(file, line)) {
        std::stringstream ss(line);
        while (getline(ss, word, delimiter)) {
            double d1 = std::stod(word);
            row_csr.push_back(d1);
        }
    }

    file.close();
    res.push_back(data_csr);
    res.push_back(row_csr);
    res.push_back(col_csr);
    return res;
}

template <class CoefficientPrecision, class CoordinatePrecision>
class CSR_generator : public VirtualGenerator<CoefficientPrecision>, public Matrix<CoefficientPrecision> {
  private:
    std::vector<double> data;
    std::vector<double> row_csr;
    std::vector<double> col_csr;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    CSR_generator(const std::vector<double> &row_csr0,
                  const std::vector<double> &col_csr0,
                  const std::vector<double> &val_csr0,
                  const Cluster<CoordinatePrecision> &target0,
                  const Cluster<CoordinatePrecision> &source0) : data(val_csr0), row_csr(row_csr0), col_csr(col_csr0), target(target0), source(source0) {}

    // CSR_generator(const Cluster<CoordinatePrecision> &target0,
    //               const Cluster<CoordinatePrecision> &source0,
    //               const std::vector<double> &row_csr0,
    //               const std::vector<double> &col_csr0,
    //               const std::vector<double> &val_csr0) : target(target0), source(source0) {

    //     int M = target0.get_size(); // nombre de lignes après permutation
    //     int N = source0.get_size(); // nombre de colonnes après permutation

    //     // Initialisation des nouvelles structures CSR
    //     row_csr.resize(M + 1, 0);
    //     int rep = 0;
    //     // Boucle sur les lignes permutées
    //     std::cout << "[____________________]" << std::flush;
    //     int repp = 0;
    //     for (int i = 0; i < M; i++) {
    //         if (i > repp * M / 20) {
    //             repp += 1;
    //             std::cout << "\r[";
    //             for (int kk = 0; kk < repp; ++kk) {
    //                 std::cout << "#";
    //             }
    //             for (int kk = repp; kk < 20; ++kk) {
    //                 std::cout << "_";
    //             }
    //             std::cout << "]" << std::flush;
    //         }
    //         int row_perm = target0.get_permutation()[i];
    //         int r1       = row_csr0[row_perm];     // début de la ligne dans row_csr
    //         int r2       = row_csr0[row_perm + 1]; // fin de la ligne dans row_csr
    //         // Boucle sur les colonnes permutées pour la ligne courante
    //         int flag = 0;
    //         for (int j = 0; j < N; j++) {
    //             int col_perm = source0.get_permutation()[j];
    //             // Extraire les colonnes et les valeurs de la ligne permutée
    //             std::vector<int> row_iperm(r2 - r1);
    //             std::copy(col_csr0.begin() + r1, col_csr0.begin() + r2, row_iperm.begin());
    //             // Rechercher l'indice de la colonne permutée
    //             auto it = std::find(row_iperm.begin(), row_iperm.end(), col_perm);
    //             if (it != row_iperm.end()) {
    //                 int idx = std::distance(row_iperm.begin(), it);
    //                 col_csr.push_back(j);               // colonne après permutation
    //                 data.push_back(val_csr0[r1 + idx]); // valeur associée
    //                 flag += 1;
    //             }
    //         }
    //         // Mettre à jour row_csr_out pour la nouvelle ligne
    //         row_csr[i + 1] = row_csr[i] + flag;
    //     }
    //     std::cout << '\n';
    // }

    // CSR_generator(const Cluster<CoordinatePrecision> &target0,
    //               const Cluster<CoordinatePrecision> &source0,
    //               const std::vector<double> &row_csr0,
    //               const std::vector<double> &col_csr0,
    //               const std::vector<double> &val_csr0) : target(target0), source(source0) {
    //     int M = target0.get_size(); // nombre de lignes après permutation
    //     int N = source0.get_size(); // nombre de colonnes après permutation

    //     // Initialisation des nouvelles structures CSR
    //     row_csr.resize(M + 1, 0);
    //     col_csr.reserve(M); // Estimation de la taille des colonnes après permutation
    //     data.reserve(M);    // Estimation de la taille des valeurs après permutation

    //     // Stocker les permutations localement pour éviter les appels répétés
    //     auto target_perm = target0.get_permutation();
    //     auto source_perm = source0.get_permutation();

    //     // Boucle sur les lignes permutées
    //     int repp = 0;
    //     // #pragma omp parallel for
    //     std::cout << "[____________________]" << std::flush;

    //     for (int i = 0; i < M; i++) {
    //         if (i > repp * M / 20) {
    //             repp += 1;
    //             std::cout << "\r[";
    //             for (int kk = 0; kk < repp; ++kk) {
    //                 std::cout << "#";
    //             }
    //             for (int kk = repp; kk < 20; ++kk) {
    //                 std::cout << "_";
    //             }
    //             std::cout << "]" << std::flush;
    //         }
    //         int row_perm = target_perm[i];
    //         int r1       = row_csr0[row_perm];
    //         int r2       = row_csr0[row_perm + 1];

    //         int flag             = 0;
    //         auto row_iperm_begin = col_csr0.begin() + r1;
    //         auto row_iperm_end   = col_csr0.begin() + r2;

    //         for (int j = 0; j < N; j++) {
    //             int col_perm = source_perm[j];

    //             auto it = std::lower_bound(row_iperm_begin, row_iperm_end, col_perm);
    //             if (it != row_iperm_end && *it == col_perm) {
    //                 int idx = std::distance(row_iperm_begin, it);
    //                 // #pragma omp critical
    //                 {
    //                     col_csr.push_back(j);               // colonne après permutation
    //                     data.push_back(val_csr0[r1 + idx]); // valeur associée
    //                 }
    //                 flag += 1;
    //             }
    //         }
    //         row_csr[i + 1] = row_csr[i] + flag;
    //     }
    //     std::cout << '\n';
    // }

    CSR_generator(const Cluster<CoordinatePrecision> &target0,
                  const Cluster<CoordinatePrecision> &source0,
                  const std::vector<double> &row_csr0,
                  const std::vector<double> &col_csr0,
                  const std::vector<double> &val_csr0) : target(target0), source(source0) {
        int M = target0.get_size(); // nombre de lignes après permutation
        int N = source0.get_size(); // nombre de colonnes après permutation

        // Initialisation des nouvelles structures CSR
        row_csr.resize(M + 1, 0);
        col_csr.resize(val_csr0.size()); // Estimation de la taille des colonnes après permutation
        data.resize(val_csr0.size());    // Estimation de la taille des valeurs après permutation

        // Stocker les permutations localement pour éviter les appels répétés
        auto target_perm = target0.get_permutation();
        auto source_perm = source0.get_permutation();
        std::vector<int> inverse_perm(N); // où N est la taille de source_perm
        for (int j = 0; j < N; ++j) {
            inverse_perm[source_perm[j]] = j; // stocke l'index d'origine
        }
        std::vector<std::pair<int, double>> col_data(val_csr0.size());
        std::transform(col_csr0.begin(), col_csr0.end(), val_csr0.begin(), col_data.begin(), [&inverse_perm](double x, double y) {
            return std::make_pair(inverse_perm[int(x)], y);
        }); // Boucle sur les lignes permutées
        int repp = 0;
        // #pragma omp parallel for
        std::cout << "[____________________]" << std::flush;

        for (int i = 0; i < M; i++) {
            if (i > repp * M / 20) {
                repp += 1;
                std::cout << "\r[";
                for (int kk = 0; kk < repp; ++kk) {
                    std::cout << "#";
                }
                for (int kk = repp; kk < 20; ++kk) {
                    std::cout << "_";
                }
                std::cout << "]" << std::flush;
            }
            int row_perm = target_perm[i];
            int r1       = row_csr0[row_perm];
            int r2       = row_csr0[row_perm + 1];
            std::sort(col_data.begin() + r1, col_data.begin() + r2, [](auto &left, auto &right) { return left.first < right.first; });
            std::transform(col_data.begin() + r1, col_data.begin() + r2, col_csr.begin() + row_csr[i], [](const std::pair<int, double> &p) { return double(p.first); });
            std::transform(col_data.begin() + r1, col_data.begin() + r2, data.begin() + row_csr[i], [](const std::pair<int, double> &p) { return p.second; });

            // for (int k = r1; k < r1 + 10; ++k) {
            //     std::cout << '(' << col_data[k].first << ',' << col_data[k].second << ')' << ',';
            // }
            // std::cout << '\n';
            // std::sort(col_data.begin() + r1, col_data.begin() + r2, [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
            //     return a.first < b.first; // Tri ascendant selon .first
            // });
            // std::cout << "!!!!!!!!!!!!!!!!" << std::endl;
            // for (int k = r1; k < r1 + 10; ++k) {
            //     std::cout << '(' << col_data[k].first << ',' << col_data[k].second << ')' << ',';
            // }
            // std::cout << '\n';
            row_csr[i + 1] = row_csr[i] + (r2 - r1);
        }
        // std::transform(col_data.begin(), col_data.end(), col_csr.begin(), [](const std::pair<int, double> &p) { return double(p.first); });
        // std::transform(col_data.begin(), col_data.end(), data.begin(), [](const std::pair<int, double> &p) { return p.second; });
        std::cout << '\n';
        // std::cout << norm2(row_csr) << ',' << norm2(col_csr) << ',' << norm2(data) << "!!!" << norm2(row_csr0) << ',' << norm2(col_csr0) << ',' << norm2(val_csr0) << std::endl;
    }

    // CSR_generator(const Cluster<CoordinatePrecision> &target0,
    //               const Cluster<CoordinatePrecision> &source0,
    //               const std::vector<double> &row_csr0,
    //               const std::vector<double> &col_csr0,
    //               const std::vector<double> &val_csr0) : target(target0), source(source0) {
    //     int M = target0.get_size(); // nombre de lignes après permutation
    //     int N = source0.get_size(); // nombre de colonnes après permutation

    //     // Initialisation des nouvelles structures CSR
    //     row_csr.resize(M + 1, 0);
    //     col_csr.resize(val_csr0.size()); // Estimation de la taille des colonnes après permutation
    //     data.resize(val_csr0.size());    // Estimation de la taille des valeurs après permutation
    //     // Stocker les permutations localement pour éviter les appels répétés
    //     auto target_perm = target0.get_permutation();
    //     auto source_perm = source0.get_permutation();
    //     std::vector<int> inverse_perm(N); // où N est la taille de source_perm
    //     for (int j = 0; j < N; ++j) {
    //         inverse_perm[source_perm[j]] = j; // stocke l'index d'origine
    //     }
    //     std::vector<std::pair<int, double>> col_data(val_csr0.size());
    //     std::transform(col_csr0.begin(), col_csr0.end(), val_csr0.begin(), col_data.begin(), [&inverse_perm](double x, double y) {
    //         return std::make_pair(inverse_perm[int(x)], y);
    //     }); // Boucle sur les lignes permutées
    //     int repp = 0;
    //     // #pragma omp parallel for
    //     std::cout << "[____________________]" << std::flush;

    //     for (int i = 0; i < M; i++) {
    //         if (i > repp * M / 20) {
    //             repp += 1;
    //             std::cout << "\r[";
    //             for (int kk = 0; kk < repp; ++kk) {
    //                 std::cout << "#";
    //             }
    //             for (int kk = repp; kk < 20; ++kk) {
    //                 std::cout << "_";
    //             }
    //             std::cout << "]" << std::flush;
    //         }
    //         int row_perm = target_perm[i];
    //         int r1       = row_csr0[row_perm];
    //         int r2       = row_csr0[row_perm + 1];
    //         std::transform(col_data.begin() + r1, col_data.begin() + r2, col_csr.begin() + row_csr[i], [](const std::pair<int, double> &p) { return double(p.first); });
    //         std::transform(col_data.begin() + r1, col_data.begin() + r2, data.begin() + row_csr[i], [](const std::pair<int, double> &p) { return p.second; });

    //         // for (int k = r1; k < r1 + 10; ++k) {
    //         //     std::cout << '(' << col_data[k].first << ',' << col_data[k].second << ')' << ',';
    //         // }
    //         // std::cout << '\n';
    //         // std::sort(col_data.begin() + r1, col_data.begin() + r2, [](const std::pair<int, double> &a, const std::pair<int, double> &b) {
    //         //     return a.first < b.first; // Tri ascendant selon .first
    //         // });
    //         // std::cout << "!!!!!!!!!!!!!!!!" << std::endl;
    //         // for (int k = r1; k < r1 + 10; ++k) {
    //         //     std::cout << '(' << col_data[k].first << ',' << col_data[k].second << ')' << ',';
    //         // }
    //         // std::cout << '\n';
    //         row_csr[i + 1] = row_csr[i] + (r2 - r1);
    //     }
    //     // std::transform(col_data.begin(), col_data.end(), col_csr.begin(), [](const std::pair<int, double> &p) { return double(p.first); });
    //     // std::transform(col_data.begin(), col_data.end(), data.begin(), [](const std::pair<int, double> &p) { return p.second; });
    //     std::cout << '\n';
    //     // std::cout << norm2(row_csr) << ',' << norm2(col_csr) << ',' << norm2(data) << "!!!" << norm2(row_csr0) << ',' << norm2(col_csr0) << ',' << norm2(val_csr0) << std::endl;
    // }

    void mat_vect(const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out) const {
        int n = row_csr.size() - 1;
        for (int i = 0; i < n; ++i) {
            for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                out[i] += data[j] * in[col_csr[j]];
            }
        }
    }

    void mat_vect(const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out, char trans) const {
        int n = row_csr.size() - 1;

        // Vérifier si la transposition est requise
        if (trans == 'N') {
            // Multiplication standard
            for (int i = 0; i < n; ++i) {
                out[i] = 0; // Réinitialiser la sortie
                for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                    out[i] += data[j] * in[col_csr[j]];
                }
            }
        } else if (trans == 'T') {
            // Multiplication par la transposée
            // Réinitialiser le vecteur de sortie à zéro
            // std::fill(out.begin(), out.end(), 0);

            for (int i = 0; i < n; ++i) {
                for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                    out[col_csr[j]] += data[j] * in[i]; // in[i] est multiplié par les valeurs de la ligne
                }
            }
        } else {
            throw std::invalid_argument("Invalid transpose character. Use 'N' for normal or 'T' for transpose.");
        }
    }

    void mat_vect(const double &alpha, const CoefficientPrecision *in, const double &beta, CoefficientPrecision *&out, char trans) const {
        int n = row_csr.size() - 1;

        // Vérifier si la transposition est requise
        if (trans == 'N') {
            // Multiplication standard
            for (int i = 0; i < n; ++i) {
                // out[i] = 0; // Réinitialiser la sortie
                for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                    out[i] = beta * out[i] + alpha * data[j] * in[col_csr[j]];
                }
            }
        } else if (trans == 'T') {
            // Multiplication par la transposée
            // Réinitialiser le vecteur de sortie à zéro
            // std::fill(out.begin(), out.end(), 0);

            for (int i = 0; i < n; ++i) {
                for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                    out[col_csr[j]] = beta * out[col_csr[j]] + alpha * data[j] * in[i]; // in[i] est multiplié par les valeurs de la ligne
                }
            }
        } else {
            throw std::invalid_argument("Invalid transpose character. Use 'N' for normal or 'T' for transpose.");
        }
    }
    void get_row(const int i, std::vector<CoefficientPrecision> &out) const {
        int n = row_csr.size() - 1;

        for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
            out[col_csr[j]] += data[j]; // in[i] est multiplié par les valeurs de la ligne
        }
    }
    // trouve le pivot dans la matrice entière
    // void get_row(const int &ii, std::vector<CoefficientPrecision> &out) const {
    //     // std::fill(out.begin(), out.end(), 0); // Réinitialise tout à zéro
    //     // for (int k = row_csr[i]; k < row_csr[i + 1]; ++k) {
    //     //     out[col_csr[k]] = data[k];
    //     // }
    //     int n = row_csr.size() - 1;
    //     std::vector<CoefficientPrecision> in(n);
    //     std::cout << n << ',' << ii << std::endl;
    //     in[ii] = 1.0;

    //     for (int i = 0; i < n; ++i) {
    //         for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
    //             std::cout << out.size() << ',' << col_csr[j] << std::endl;
    //             out[col_csr[j]] += data[j] * in[i]; // in[i] est multiplié par les valeurs de la ligne
    //         }
    //     }
    // }
    void findPivot(int &pivot_row, int &pivot_col) const {
        auto max_it        = std::max_element(data.begin(), data.end());
        int pivot_index    = std::distance(data.begin(), max_it);
        double pivot_value = *max_it;

        // Récupérer le pivot_col à partir de `col_indices`
        pivot_col = col_csr[pivot_index];

        // Récupérer le pivot_row à partir de `row_ptr`
        for (int i = 0; i < row_csr.size(); ++i) {
            if (row_csr[i] <= pivot_index && pivot_index < row_csr[i + 1]) {
                pivot_row = i;
                break;
            }
        }
    }

    // trouve le pivot dans un sous blocs , renvoie les coeff ij de la matrice entière, pas du sous blocs ;
    double findPivot_subblock(int &pivot_row, int &pivot_col, const int &sizet, const int &size_s, const int &oft, const int &ofs) const {
        double max_value = -std::numeric_limits<double>::infinity(); // Valeur max initiale
        int pivot_index  = -1;

        // Parcourir le sous-bloc (débuter à `oft` et `ofs`)
        for (int i = oft; i < oft + sizet; ++i) {
            for (int idx = row_csr[i]; idx < row_csr[i + 1]; ++idx) {
                int j = col_csr[idx];

                // Ne considérer que les éléments dans le sous-bloc
                if (j >= ofs && j < ofs + size_s) {
                    if (std::abs(data[idx]) > max_value) {
                        max_value   = std::abs(data[idx]);
                        pivot_index = idx;
                    }
                }
            }
        }

        if (pivot_index == -1) {
            // std::cerr << "no pivot found " << std::endl;
            return max_value;
        }

        // Récupérer le pivot_col à partir de `col_indices`
        pivot_col = col_csr[pivot_index];

        // Récupérer le pivot_row à partir de `row_ptr`
        for (int i = oft; i < oft + sizet; ++i) {
            if (row_csr[i] <= pivot_index && pivot_index < row_csr[i + 1]) {
                pivot_row = i;
                break;
            }
        }
        return max_value;
    }
    void row_axpy(const int &i0, const double &alpha, const std::vector<CoefficientPrecision> &in) {
        std::transform(data.begin() + row_csr[i0], data.begin() + row_csr[i0 + 1], // Plage des éléments de x
                       in.begin() + row_csr[i0],                                   // Plage des éléments de y
                       data.begin() + row_csr[i0],                                 // Stocker le résultat dans x
                       [alpha](double xi, double yi) {                             // Lambda fonction pour faire x[k] = x[k] - a * y[k]
                           return xi - alpha * yi;
                       });
    }

    void multiply_column(int j, double alpha) {
        // Parcourir toutes les lignes de la matrice CSR
        for (int row = 0; row < row_csr.size() - 1; ++row) {
            // Obtenir les indices de début et de fin des éléments de cette ligne
            int row_start = row_csr[row];
            int row_end   = row_csr[row + 1];

            // Parcourir les colonnes non-nulles de la ligne `row`
            for (int idx = row_start; idx < row_end; ++idx) {
                if (col_csr[idx] == j) {
                    // Si la colonne correspond à `j`, multiplier la valeur par `alpha`
                    data[idx] *= alpha;
                }
            }
        }
    }
    void multiply_row(const int &i0, const double &alpha) {
        for (int i = row_csr[i0]; i < row_csr[i0 + 1]; ++i) {
            data[i] = alpha * data[i];
        }
    }
    double findPivot_iterate(const int &M, const int &N, int &pivot_row, int &pivot_col, std::vector<int> &I, std::vector<int> &J) const {
        double max_value = -std::numeric_limits<double>::infinity(); // Valeur max initiale
        int pivot_index  = -1;

        // Parcourir le sous-bloc (débuter à `oft` et `ofs`)
        for (int i = 0; i < M; ++i) {
            if (std::find(I.begin(), I.end(), i) == I.end()) {

                for (int idx = row_csr[i]; idx < row_csr[i + 1]; ++idx) {
                    int j = col_csr[idx];

                    // Ne considérer que les éléments dans le sous-bloc et qui ne sont pas dans J
                    if (std::find(J.begin(), J.end(), j) == J.end()) {
                        if (std::abs(data[idx]) > max_value) {
                            // on regarde si la row est pas déja visité :
                            max_value   = std::abs(data[idx]);
                            pivot_index = idx;
                            pivot_row   = i;
                        }
                    }
                }
            }
        }

        if (pivot_index == -1) {
            // std::cerr << "No pivot found " << std::endl;
            return max_value;
        }

        // Récupérer le pivot_col à partir de `col_csr`
        pivot_col = col_csr[pivot_index];

        // Récupérer le pivot_row à partir de `row_csr`
        // for (int i = 0; i < row_csr.size() - 1; ++i) {
        //     if (row_csr[i] <= pivot_index && pivot_index < row_csr[i + 1]) {
        //         pivot_row = i;
        //         break;
        //     }
        // }

        // Ajouter les indices trouvés à leurs listes respectives
        I.push_back(pivot_row);
        J.push_back(pivot_col);

        return max_value;
    }

    double
    findPivot_subblock_iterate(int &pivot_row, int &pivot_col, const int &sizet, const int &size_s, const int &oft, const int &ofs, std::vector<int> &I, std::vector<int> &J) const {
        double max_value = -std::numeric_limits<double>::infinity(); // Valeur max initiale
        int pivot_index  = -1;

        // Parcourir le sous-bloc (débuter à `oft` et `ofs`)
        for (int i = oft; i < oft + sizet; ++i) {
            // Vérifier si l'indice i n'est pas dans la liste I
            if (std::find(I.begin(), I.end(), i) == I.end()) {

                for (int idx = row_csr[i]; idx < row_csr[i + 1]; ++idx) {
                    int j = col_csr[idx];

                    // Ne considérer que les éléments dans le sous-bloc et qui ne sont pas dans J
                    if (std::find(J.begin(), J.end(), j) == J.end()) {
                        if (j >= ofs && j < ofs + size_s) {
                            if (std::abs(data[idx]) > max_value) {
                                // if (std::find(I.begin(), I.end(), i) == I.end()) {

                                max_value   = std::abs(data[idx]);
                                pivot_index = idx;
                                pivot_row   = i;
                            }
                        }
                    }
                }
            }
        }

        if (pivot_index == -1) {
            // std::cerr << "No pivot found " << std::endl;
            return max_value;
        }

        // Récupérer le pivot_col à partir de `col_csr`
        pivot_col = col_csr[pivot_index];

        // Récupérer le pivot_row à partir de `row_csr`
        // for (int i = oft; i < oft + sizet; ++i) {
        //     if (row_csr[i] <= pivot_index && pivot_index < row_csr[i + 1]) {
        //         pivot_row = i;
        //         break;
        //     }
        // }

        // Ajouter les indices trouvés à leurs listes respectives
        I.push_back(pivot_row);
        J.push_back(pivot_col);

        return max_value;
    }

    CSR_generator
    copy_sub_csr(const int &M, const int &N, const int &oft, const int &ofs) const {
        std::vector<double> newrow(M + 1, 0);     // Initialiser le vecteur des indices de lignes
        std::vector<double> newcol;               // Indices de colonnes pour le sous-bloc
        std::vector<CoefficientPrecision> newval; // Valeurs pour le sous-bloc

        int nnz = 0; // Compteur de non-zero

        // Parcourir les lignes du sous-bloc
        for (int i = 0; i < M; ++i) {
            int original_row = oft + i; // Décalage dans la matrice d'origine
            newrow[i]        = nnz;     // Marquer le début de la ligne i dans newcol/newval

            // Parcourir les éléments non-nuls de la ligne originale
            for (int idx = row_csr[original_row]; idx < row_csr[original_row + 1]; ++idx) {
                int original_col = col_csr[idx]; // Obtenir l'indice de colonne
                if (original_col > ofs + N) {
                    break; // ils ont trié ca sert a rien de continuer
                }

                // Vérifier si l'indice de colonne est dans le sous-bloc
                if (original_col >= ofs && original_col < ofs + N) {
                    newcol.push_back(original_col - ofs); // Décaler l'indice de colonne pour le sous-bloc
                    newval.push_back(data[idx]);          // Copier la valeur correspondante
                    ++nnz;                                // Incrémenter le nombre de non-zéros
                }
            }
        }

        // Finaliser newrow pour la dernière ligne
        newrow[M] = nnz;

        // Créer une nouvelle instance de CSR_generator avec les données du sous-bloc
        return CSR_generator(newrow, newcol, newval, target, source);
    }

    CSR_generator
    get_sub_csr(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
        int M = t.get_size();
        int N = s.get_size();
        std::vector<double> newrow(M + 1, 0);     // Initialiser le vecteur des indices de lignes
        std::vector<double> newcol;               // Indices de colonnes pour le sous-bloc
        std::vector<CoefficientPrecision> newval; // Valeurs pour le sous-bloc
        int oft = t.get_offset() - target.get_offset();
        int ofs = s.get_offset() - source.get_offset();
        int nnz = 0; // Compteur de non-zero

        // Parcourir les lignes du sous-bloc
        for (int i = 0; i < M; ++i) {
            int original_row = oft + i; // Décalage dans la matrice d'origine
            newrow[i]        = nnz;     // Marquer le début de la ligne i dans newcol/newval

            // Parcourir les éléments non-nuls de la ligne originale
            for (int idx = row_csr[original_row]; idx < row_csr[original_row + 1]; ++idx) {
                int original_col = col_csr[idx]; // Obtenir l'indice de colonne

                // Vérifier si l'indice de colonne est dans le sous-bloc
                if (original_col >= ofs && original_col < ofs + N) {
                    newcol.push_back(original_col - ofs); // Décaler l'indice de colonne pour le sous-bloc
                    newval.push_back(data[idx]);          // Copier la valeur correspondante
                    ++nnz;                                // Incrémenter le nombre de non-zéros
                }
            }
        }

        // Finaliser newrow pour la dernière ligne
        newrow[M] = nnz;

        // Créer une nouvelle instance de CSR_generator avec les données du sous-bloc
        return CSR_generator(newrow, newcol, newval, t, s);
    }

    // void plus_external_product(const std::vector<CoefficientPrecision> l , cosnt std::vector<CoefficientPrecision> u){
    //     for (int i  0 ; i < l.size() ; ++i){
    //         if ( l[i] != 0){
    //             for (int j = 0 ; j < )
    //         }
    //     }
    // }
    void mat_vect_bloc(const int &sizet, const int &size_s, const int &oft, const int &ofs, const std::vector<CoefficientPrecision> &in, std::vector<CoefficientPrecision> &out, char trans) const {
        if (trans == 'N') {
            // std::cout << "ici ! " << std::endl;
            // Multiplication standard: CSR * vecteur
            for (int i = oft; i < sizet + oft; ++i) {
                for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                    int col = col_csr[j] - ofs;     // Ajuster l'indice de colonne avec ofs
                    if (col >= 0 && col < size_s) { // Vérifier si la colonne est dans le sous-bloc
                        out[i - oft] += data[j] * in[col];
                    }
                }
            }
        } else if (trans == 'T') {
            // Multiplication transposée: CSR^T * vecteur
            // Note: Pour la transposée, les rôles de `i` et `col` sont inversés.
            for (int i = oft; i < sizet + oft; ++i) {
                for (int j = row_csr[i]; j < row_csr[i + 1]; ++j) {
                    int col = col_csr[j] - ofs;            // Ajuster l'indice de colonne avec ofs
                    if (col >= 0 && col < size_s) {        // Vérifier si la colonne est dans le sous-bloc
                        out[col] += data[j] * in[i - oft]; // Transposer le calcul : out[col] = A^T * in
                    }
                }
            }
        }
    }

    std::vector<Matrix<CoefficientPrecision>> rd_ACA(const int &sizet, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout) const {
        int rk = 1;
        // std::cout << "ACA sur le bloc  ( " << sizet << ',' << oft << ")" << " , ( " << size_s << ',' << ofs << ")" << std::endl;
        Matrix<CoefficientPrecision> L(sizet, 1);
        Matrix<CoefficientPrecision> Ut(size_s, 1);
        Matrix<CoefficientPrecision> temp(sizet, sizet);
        std::vector<Matrix<CoefficientPrecision>> res;
        for (int k = 0; k < sizet; ++k) {
            temp(k, k) = 1.0;
        }
        double norme_LU = 0.0;
        // Matrix<double> bloc(sizet, size_s);
        // this->copy_submatrix(sizet, size_s, oft, ofs, bloc.data());
        auto w = gaussian_vector(size_s, 0.0, 1.0);
        // std::vector<CoefficientPrecision> lr(sizet, 0.0);

        while ((rk < sizet / 2) && (rk < size_s / 2)) {
            std::vector<CoefficientPrecision> Aw(sizet, 0.0);
            // bloc.add_vector_product('N', 1.0, w.data(), 1.0, Aw.data());
            this->mat_vect_bloc(sizet, size_s, oft, ofs, w, Aw, 'N');
            // std::cout << "norme Aw : " << norm2(Aw) << std::endl;
            std::vector<CoefficientPrecision> lr(sizet, 0.0);
            // std::cout << Aw.size() << ',' << lr.size() << ',' << sizet << std::endl;
            temp.add_vector_product('N', 1.0, Aw.data(), 0.0, lr.data());
            // lr           = temp * Aw;
            auto norm_lr = norm2(lr);
            if (norm_lr < 1e-20) {
                break;
            } else {
                lr = mult(1.0 / norm_lr, lr);
            }
            std::vector<CoefficientPrecision> ur(size_s, 0.0);
            // bloc->add_vector_product('T', 1.0, lr.data(), 1.0, ur.data());
            this->mat_vect_bloc(sizet, size_s, oft, ofs, lr, ur, 'T');
            auto norm_ur = norm2(ur);
            // std::cout << "norm2 ur : " << norm_ur << std::endl;
            // Matrix<CoefficientPrecision> lr_ur(sizet, size_s);
            // for (int kk = 0; kk < sizet; ++kk) {
            //     for (int ll = 0; ll < size_s; ++ll) {
            //         lr_ur(kk, ll) = lr[kk] * ur[ll];
            //     }
            // }
            double alpha = 1.0;
            double beta  = -1.0;
            int kk       = 1;
            // Blas<CoefficientPrecision>::gemm("N", "T", &sizet, &size_s, &kk, &alpha, lr.data(), &sizet, ur.data(), &size_s, &alpha, lr_ur.data(), &sizet);
            std::vector<CoefficientPrecision> Llr(L.nb_cols(), 0.0);
            L.add_vector_product('T', 1.0, lr.data(), 1.0, Llr.data());
            // Llr = transp
            std::vector<CoefficientPrecision> ULlr(size_s, 0.0);
            Ut.add_vector_product('N', 1.0, Llr.data(), 1.0, ULlr.data());
            double trace = 0.0;
            for (int k = 0; k < std::min(sizet, size_s); ++k) {
                trace += (ULlr[k] * ur[k]);
            }
            // auto nr_lrur = normFrob(lr_ur); // lui il fait vraiment trés mal
            auto nr_lrur = norm2(ur);
            auto nr      = norme_LU + std::pow(nr_lrur, 2.0) + 2 * trace;
            if (rk > 1 && (std::pow(norm_lr * norm_ur * (size_s - rk), 2.0) <= std::pow(tolerance, 2.0) * nr)) {
                break;
            }
            if (rk == 1) {
                std::copy(lr.data(), lr.data() + sizet, L.data());
                std::copy(ur.data(), ur.data() + size_s, Ut.data());
                // std::cout << "!!!!!! aprés affectation " << normFrob(L) << ',' << normFrob(Ut) << "! " << norm_lr << ',' << norm_ur << std::endl;
            } else {
                Matrix<CoefficientPrecision> new_L(sizet, rk);
                std::copy(L.data(), L.data() + L.nb_rows() * L.nb_cols(), new_L.data());
                std::copy(lr.data(), lr.data() + sizet, new_L.data() + (rk - 1) * sizet);
                Matrix<CoefficientPrecision> new_U(size_s, rk);
                std::copy(Ut.data(), Ut.data() + Ut.nb_rows() * Ut.nb_cols(), new_U.data());
                std::copy(ur.data(), ur.data() + size_s, new_U.data() + (rk - 1) * size_s);
                L  = new_L;
                Ut = new_U;
            }
            w = gaussian_vector(size_s, 0.0, 1.0);
            Blas<CoefficientPrecision>::gemm("N", "T", &sizet, &sizet, &kk, &beta, lr.data(), &sizet, lr.data(), &sizet, &alpha, temp.data(), &sizet);
            // Matrix<CoefficientPrecision> lrlr(sizet, sizet);
            // for (int kk = 0; kk < sizet; ++kk) {
            //     for (int ll = 0; ll < sizet; ++ll) {
            //         lrlr(kk, ll) = lr[kk] * lr[ll];
            //     }
            // }
            // temp = temp - lrlr;
            norme_LU += std::pow(nr_lrur, 2.0);
            rk += 1;
        }
        // std::cout << "ACA end avec rk = " << rk << "sur un bloc ( " << sizet << ',' << oft << ") , (" << size_s << ',' << ofs << ")" << std::endl;
        if (rk >= std::min(sizet / 2, size_s / 2)) {
            flagout = false;
        }
        if (flagout) {
            if (rk == 1) {
                Matrix<CoefficientPrecision> U(1, size_s);
                res.push_back(L);
                res.push_back(U);
            } else {
                Matrix<CoefficientPrecision> U(rk - 1, size_s);
                transpose(Ut, U);
                // LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_mat(L, U);
                // bloc->delete_children();
                // bloc->set_low_rank_data(lr_mat);
                res.push_back(L);
                res.push_back(U);
            }
        } else {
            // Matrix<CoefficientPrecision> dense(this->get_target_cluster().get_size(), this->get_source_cluster().get_size());
            // copy_to_dense(*this, dense.data());
            // Matrix<CoefficientPrecision> U(rk - 1, size_s);
            // transpose(Ut, U);

            // // std::cerr << "aucune approximation trouvée , rk=" << rk << "sur un bloc de taille " << this->get_target_cluster().get_size() << ',' << this->get_source_cluster().get_size() << " et de norme : " << normFrob(dense) << std::endl;
            // std::cerr << "pas d'approximation ,  l'erreur a la fin est " << normFrob(dense - L * U) << std::endl;
            std::cout << "pas d'approximation sur le bloc ( " << sizet << ',' << oft << ") , (" << size_s << ',' << ofs << ")" << std::endl;
        }
        return res;
    }

    // std::vector<Matrix<CoefficientPrecision>> full_ACA(const int &sizet, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout) const {
    //     int rk = 0;
    //     // std::vector<CoefficientPrecision> copy_val = this->data;
    //     std::vector<std::vector<CoefficientPrecision>> Lvect, Uvect;
    //     std::vector<Matrix<CoefficientPrecision>> res;
    //     double norme = 0.0;
    //     // Matrix<CoefficientPrecision> cop(sizet, size_s);
    //     // this->copy_submatrix(sizet, size_s, oft, ofs, cop.data());
    //     // auto sub_csr = this->copy_sub_csr(sizet, size_s, oft, ofs);
    //     // Matrix<CoefficientPrecision> copp(sizet, size_s);
    //     // sub_csr.copy_submatrix(sizet, size_s, 0, 0, copp.data());
    //     // std::cout << "sub csr : " << normFrob(copp - cop) << std::endl;
    //     double nr               = 0.0;
    //     double nm               = 0.0;
    //     bool stopping_criterion = true;
    //     std::vector<int> Ipiv, Jpiv;
    //     bool toosoon = false;
    //     double Uu    = 0.0;
    //     double Ll    = 0.0;
    //     while (stopping_criterion) {
    //         nm = nr;
    //         if (rk >= std::min(sizet, size_s) / 2) {
    //             break;
    //         }
    //         int i0, j0;
    //         auto max_value = this->findPivot_subblock_iterate(i0, j0, sizet, size_s, oft, ofs, Ipiv, Jpiv);
    //         // auto max_value = sub_csr.findPivot_iterate(sizet, size_s, i0, j0, Jpiv);
    //         // i0 = i0 + oft;
    //         // j0 = j0 + ofs;
    //         // i0 = i0;
    //         // j0 = j0;
    //         if (max_value <= 0.0) {
    //             toosoon = true;

    //             break;
    //         }
    //         std::vector<CoefficientPrecision> ei(sizet);
    //         ei[i0 - oft] = 1.0;
    //         std::vector<CoefficientPrecision> uk(size_s);
    //         this->mat_vect_bloc(sizet, size_s, oft, ofs, ei, uk, 'T');
    //         for (int r = 0; r < rk; ++r) {
    //             auto alpha = -1.0 * Lvect[r][i0 - oft];
    //             int inc    = 1;
    //             Blas<CoefficientPrecision>::axpy(&size_s, &alpha, Uvect[r].data(), &inc, uk.data(), &inc);
    //         }
    //         // auto uk = cop.get_row(i0 - oft);
    //         // for (int r = 0; r < rk; ++r) {
    //         //     auto ur = Uvect[r];
    //         //     ur      = mult(-1.0 * Lvect[r][i0 - oft], ur);
    //         //     uk      = uk + ur;
    //         // }
    //         // std::cout << uk[j0 - ofs] << std::endl;
    //         // if (uk[j0 - ofs] == 0.0) {

    //         if (uk[j0 - ofs] == 0.0) {

    //             toosoon = true;
    //             break;
    //         }
    //         uk = mult(1.0 / uk[j0 - ofs], uk);
    //         std::vector<CoefficientPrecision> ej(sizet);
    //         ej[j0 - ofs] = 1.0;
    //         std::vector<CoefficientPrecision> lk(sizet);
    //         this->mat_vect_bloc(sizet, size_s, oft, ofs, ej, lk, 'N');
    //         for (int r = 0; r < rk; ++r) {
    //             auto alpha = -1 * Uvect[r][j0 - ofs];
    //             int inc    = 1;
    //             Blas<CoefficientPrecision>::axpy(&size_s, &alpha, Lvect[r].data(), &inc, lk.data(), &inc);
    //         }

    //         // camcule norme pour critère d'arret

    //         // auto lk = cop.get_col(j0 - ofs);
    //         // for (int r = 0; r < rk; ++r) {
    //         //     auto lr = Lvect[r];
    //         //     lr      = mult(-1.0 * Uvect[r][j0 - ofs], lr);
    //         //     lk      = lk + lr;
    //         // }
    //         Uvect.push_back(uk);
    //         Lvect.push_back(lk);

    //         // std::vector<CoefficientPrecision> temp(rk, 0.0);
    //         // for (int k = 0; k < rk; ++k) {
    //         //     // temp[i] = ui^T uk
    //         //     int size     = 1;
    //         //     double alpha = 1.0;
    //         //     // Blas<CoefficientPrecision>::gemv("T", &size, &size_s, &alpha, Uvect[k].data(), &size, uk.data(), &size, &alpha, temp.data() + k, &size);
    //         //     temp[k] = std::inner_product(Uvect[k].begin(), Uvect[k].end(), uk.begin(), 0.0);
    //         // }
    //         // std::vector<CoefficientPrecision> tempp(rk, 0.0);
    //         // for (int k = 0; k < rk; ++k) {
    //         //     int size     = 1;
    //         //     double alpha = 1.0;
    //         //     // Blas<CoefficientPrecision>::gemv("T", &size, &sizet, &alpha, Lvect[k].data(), &size, lk.data(), &size, &alpha, tempp.data() + k, &size);
    //         //     tempp[k] = std::inner_product(Lvect[k].begin(), Lvect[k].end(), lk.begin(), 0.0);
    //         // }
    //         // double trace = 0.0;
    //         // for (int k = 0; k < rk; ++k) {
    //         //     trace += temp[k] * tempp[k];
    //         // }
    //         // norme = norme + std::pow(nm, 2.0) + 2 * trace;

    //         nr = norm2(lk) * norm2(uk);
    //         // stopping_criterion = ((size_s - rk) * std::pow(nr, 2.0) > std::pow(tolerance, 2.0) * norme);
    //         stopping_criterion = ((size_s - rk) * nr > tolerance);
    //         // sub_csr.multiply_row(i0 - oft, 0.000000001);
    //         // sub_csr.multiply_column(j0 - ofs, 0.000000001);
    //         // std::cout << "rk = " << rk << "norme : " << norm2(uk) << ',' << norm2(lk) << std::endl;

    //         rk += 1;
    //         // Matrix<CoefficientPrecision> L(sizet, rk);
    //         // Matrix<CoefficientPrecision> U(rk, size_s);
    //         // for (int k = 0; k < rk; ++k) {
    //         //     auto ll = Lvect[k];
    //         //     auto uu = Uvect[k];
    //         //     for (int l = 0; l < sizet; ++l) {
    //         //         L(l, k) = ll[l];
    //         //     }
    //         //     for (int l = 0; l < size_s; ++l) {
    //         //         U(k, l) = uu[l];
    //         //     }
    //         // }
    //         // norme = norm2(lk) * norm2(uk);
    //         // std::cout << "erreur au rang  " << rk << " est " << normFrob(cop - L * U) << "     " << normFrob(L) << ',' << normFrob(U) << ',' << norm2(uk) << std::endl;
    //         // stopping_criterion = (normFrob(cop - L * U) > tolerance);
    //         // std::cout << "stoping : " << stopping_criterion << std::endl;
    //         // nr += (norme + 2 * nr * norme);
    //         // rk += 1;
    //         // stopping_criterion = (std::pow(nr, 2.0) > std::pow(tolerance * norme, 2.0))
    //     }
    //     if (rk == 0) {
    //         Matrix<CoefficientPrecision> L(sizet, 1);
    //         Matrix<CoefficientPrecision> U(1, size_s);
    //         res.push_back(L);
    //         res.push_back(U);
    //         return res;
    //     } else if (rk >= std::min(sizet, size_s) / 2 || toosoon) {
    //         flagout = false;
    //         return res;
    //     } else {
    //         Matrix<CoefficientPrecision> L(sizet, rk);
    //         Matrix<CoefficientPrecision> Ut(size_s, rk);
    //         for (int k = 0; k < rk; ++k) {
    //             auto lk = Lvect[k];
    //             auto uk = Uvect[k];
    //             // std::cout << "nom dans a boucle  " << norm2(lk) << ',' << norm2(uk) << std::endl;
    //             for (int l = 0; l < sizet; ++l) {
    //                 L(l, k) = lk[l];
    //             }
    //             for (int l = 0; l < size_s; ++l) {
    //                 Ut(l, k) = uk[l];
    //             }
    //         }
    //         Matrix<CoefficientPrecision> U(rk, size_s);
    //         transpose(Ut, U);
    //         Matrix<CoefficientPrecision> cop(sizet, size_s);
    //         // this->copy_submatrix(sizet, size_s, oft, ofs, cop.data());
    //         // // std::cout << sizet << ',' << size_s << ',' << cop.nb_rows() << ',' << cop.nb_cols() << L.nb_rows() << ',' << L.nb_cols() << ',' << U.nb_rows() << ',' << U.nb_cols() << std::endl;
    //         // std::cout << "ACA trouve une approx d'erreur : " << normFrob(cop - L * U) / normFrob(cop) << std::endl;
    //         // std::cout << nr << ',' << norme << ',' << std::pow(normFrob(L * U), 2.0) << ',' << tolerance << std::endl;

    //         res.push_back(L);
    //         res.push_back(U);
    //         return res;
    //     }
    // }

    bool is_zero() {
        bool res;
        if (data.size() == 0) {
            res = true;
        } else {
            res = false;
        }
        return res;
    }

    std::vector<Matrix<CoefficientPrecision>> zero_map(const int &sizet, const int &sizes, const int &oft0, const int &ofs0, bool &flagout) const {
        int oft         = oft0 - target.get_offset();
        int ofs         = ofs0 - source.get_offset();
        int row_oft     = row_csr[oft];
        int lmin        = col_csr[row_oft];
        int row_of_plus = row_csr[sizet + oft] - 1;
        int lmax        = col_csr[row_of_plus];
        std::vector<Matrix<CoefficientPrecision>> res;
        if (oft > ofs) {
            if (sizes + ofs < lmin) {
                // bloc a gauche de la première valeur non nulle -> bloc nulle
                Matrix<CoefficientPrecision> A(sizet, 0);
                Matrix<CoefficientPrecision> B(0, sizes);
                res.push_back(A);
                res.push_back(B);
            } else {
                flagout = false;
            }
        } else if (oft < ofs) {
            if (ofs > lmax) {
                Matrix<CoefficientPrecision> A(sizet, 0);
                Matrix<CoefficientPrecision> B(0, sizes);
                res.push_back(A);
                res.push_back(B);
            } else {
                flagout = false;
            }
        } else {
            flagout = false;
        }

        return res;
    }
    int data_size() {
        return data.size();
    }
    std::vector<Matrix<CoefficientPrecision>> full_ACA(const int &sizet, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout) const {
        int rk = 0;
        std::vector<std::vector<CoefficientPrecision>> Lvect, Uvect;
        std::vector<Matrix<CoefficientPrecision>> res;
        double norme            = 0.0;
        auto sub_csr            = this->copy_sub_csr(sizet, size_s, oft, ofs);
        double nr               = 0.0;
        double nm               = 0.0;
        bool stopping_criterion = true;
        std::vector<int> Ipiv, Jpiv;
        bool toosoon = false;
        double Uu    = 0.0;
        double Ll    = 0.0;
        while (stopping_criterion) {
            nm = nr;
            if (rk >= std::min(sizet, size_s) / 2) {
                break;
            }
            int i0, j0;
            auto max_value = sub_csr.findPivot_iterate(sizet, size_s, i0, j0, Ipiv, Jpiv);
            if (max_value <= 0.0) {
                toosoon = true;

                break;
            }
            std::vector<CoefficientPrecision> uk(size_s);
            std::vector<CoefficientPrecision> ei(sizet);
            ei[i0] = 1.0;
            sub_csr.get_row(i0, uk);
            for (int r = 0; r < rk; ++r) {
                // auto alpha = -1.0 * Lvect[r][i0 - oft];
                auto alpha = -1.0 * Lvect[r][i0];

                int inc = 1;
                Blas<CoefficientPrecision>::axpy(&size_s, &alpha, Uvect[r].data(), &inc, uk.data(), &inc);
            }
            if (uk[j0] == 0.0) {

                toosoon = true;
                break;
            }
            double alpha = 1.0 / uk[j0];
            // std::transform(uk.begin(), uk.end(), uk.begin(), [alpha](double x) { return alpha * x; });
            for (auto &val : uk)
                val *= alpha;
            std::vector<CoefficientPrecision> ej(size_s);
            // ej[j0 - ofs] = 1.0;
            ej[j0] = 1.0;

            std::vector<CoefficientPrecision> lk(sizet);
            sub_csr.mat_vect(ej, lk, 'N');

            for (int r = 0; r < rk; ++r) {
                auto alpha = -1 * Uvect[r][j0];
                int inc    = 1;
                Blas<CoefficientPrecision>::axpy(&size_s, &alpha, Lvect[r].data(), &inc, lk.data(), &inc);
            }
            Uvect.push_back(uk);
            Lvect.push_back(lk);
            nr                 = norm2(lk) * norm2(uk);
            stopping_criterion = ((size_s - rk) * nr > tolerance);
            rk += 1;
        }
        if (rk == 0) {
            Matrix<CoefficientPrecision> L(sizet, 1);
            Matrix<CoefficientPrecision> U(1, size_s);
            res.push_back(L);
            res.push_back(U);
            return res;
        } else if (rk >= std::min(sizet, size_s) / 2 || toosoon) {
            flagout = false;
            return res;
        } else {
            Matrix<CoefficientPrecision> L(sizet, rk);
            Matrix<CoefficientPrecision> Ut(size_s, rk);
            for (int k = 0; k < rk; ++k) {
                // auto lk = Lvect[k];
                // auto uk = Uvect[k];
                // for (int l = 0; l < sizet; ++l) {
                //     L(l, k) = lk[l];
                // }
                // for (int l = 0; l < size_s; ++l) {
                //     Ut(l, k) = uk[l];
                // }
                std::copy(Lvect[k].begin(), Lvect[k].end(), L.data() + k * sizet);   // Évite la boucle
                std::copy(Uvect[k].begin(), Uvect[k].end(), Ut.data() + k * size_s); // Évite la boucle
            }
            Matrix<CoefficientPrecision> U(rk, size_s);
            transpose(Ut, U);
            Matrix<CoefficientPrecision> cop(sizet, size_s);
            res.push_back(L);
            res.push_back(U);
            return res;
        }
    }

    std::vector<Matrix<CoefficientPrecision>> full_ACA(const int &sizet, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout, bool flagzeros) const {
        int rk = 0;
        std::vector<std::vector<CoefficientPrecision>> Lvect, Uvect;
        std::vector<Matrix<CoefficientPrecision>> res;
        double norme            = 0.0;
        auto sub_csr            = this->copy_sub_csr(sizet, size_s, oft, ofs);
        double nr               = 0.0;
        double nm               = 0.0;
        bool stopping_criterion = true;
        std::vector<int> Ipiv, Jpiv;
        bool toosoon = false;
        double Uu    = 0.0;
        double Ll    = 0.0;
        while (stopping_criterion) {
            nm = nr;
            if (rk >= std::min(sizet, size_s) / 2) {
                break;
            }
            int i0, j0;
            auto max_value = sub_csr.findPivot_iterate(sizet, size_s, i0, j0, Ipiv, Jpiv);
            if (max_value <= 0.0) {
                toosoon = true;

                break;
            }
            std::vector<CoefficientPrecision> uk(size_s);
            std::vector<CoefficientPrecision> ei(sizet);
            ei[i0] = 1.0;
            sub_csr.get_row(i0, uk);
            for (int r = 0; r < rk; ++r) {
                // auto alpha = -1.0 * Lvect[r][i0 - oft];
                auto alpha = -1.0 * Lvect[r][i0];

                int inc = 1;
                Blas<CoefficientPrecision>::axpy(&size_s, &alpha, Uvect[r].data(), &inc, uk.data(), &inc);
            }
            if (uk[j0] == 0.0) {

                toosoon = true;
                break;
            }
            double alpha = 1.0 / uk[j0];
            // std::transform(uk.begin(), uk.end(), uk.begin(), [alpha](double x) { return alpha * x; });
            for (auto &val : uk)
                val *= alpha;
            std::vector<CoefficientPrecision> ej(size_s);
            // ej[j0 - ofs] = 1.0;
            ej[j0] = 1.0;

            std::vector<CoefficientPrecision> lk(sizet);
            sub_csr.mat_vect(ej, lk, 'N');

            for (int r = 0; r < rk; ++r) {
                auto alpha = -1 * Uvect[r][j0];
                int inc    = 1;
                Blas<CoefficientPrecision>::axpy(&sizet, &alpha, Lvect[r].data(), &inc, lk.data(), &inc);
            }
            Uvect.push_back(uk);
            Lvect.push_back(lk);
            nr                 = norm2(lk) * norm2(uk);
            stopping_criterion = ((size_s - rk) * nr > tolerance);
            rk += 1;
        }
        if (rk == 0) {
            Matrix<CoefficientPrecision> L(sizet, 0);
            Matrix<CoefficientPrecision> U(0, size_s);
            res.push_back(L);
            res.push_back(U);
            flagzeros = true;
            return res;
        } else if (rk >= std::min(sizet, size_s) / 2 || toosoon) {
            flagout = false;
            return res;
        } else {
            Matrix<CoefficientPrecision> L(sizet, rk);
            // Matrix<CoefficientPrecision> Ut(size_s, rk);
            Matrix<CoefficientPrecision> U(rk, size_s);

            for (int k = 0; k < rk; ++k) {
                std::copy(Lvect[k].begin(), Lvect[k].end(), L.data() + k * sizet); // Évite la boucle
                for (int j = 0; j < size_s; ++j) {
                    U(k, j) = Uvect[k][j]; // Remplissage direct de U
                }
                // std::copy(Uvect[k].begin(), Uvect[k].end(), Ut.data() + k * size_s); // Évite la boucle
            }
            // Matrix<CoefficientPrecision> U(rk, size_s);
            // transpose(Ut, U);
            res.push_back(L);
            res.push_back(U);
            return res;
        }
    }

    std::vector<Matrix<CoefficientPrecision>> full_ACA_sub(const int &sizet, const int &size_s, const int &oft, const int &ofs, const double &tolerance, bool &flagout, bool flagzeros) const {
        int rk = 0;
        std::vector<std::vector<CoefficientPrecision>> Lvect, Uvect;
        std::vector<Matrix<CoefficientPrecision>> res;
        double norme = 0.0;
        // auto sub_csr            = this->copy_sub_csr(sizet, size_s, oft, ofs);
        double nr               = 0.0;
        double nm               = 0.0;
        bool stopping_criterion = true;
        std::vector<int> Ipiv, Jpiv;
        bool toosoon = false;
        double Uu    = 0.0;
        double Ll    = 0.0;
        while (stopping_criterion) {
            nm = nr;
            if (rk >= std::min(sizet, size_s) / 2) {
                break;
            }
            int i0, j0;
            auto max_value = this->findPivot_iterate(sizet, size_s, i0, j0, Ipiv, Jpiv);
            if (max_value <= 0.0) {
                toosoon = true;

                break;
            }
            std::vector<CoefficientPrecision> uk(size_s);
            std::vector<CoefficientPrecision> ei(sizet);
            ei[i0] = 1.0;
            this->get_row(i0, uk);
            for (int r = 0; r < rk; ++r) {
                // auto alpha = -1.0 * Lvect[r][i0 - oft];
                auto alpha = -1.0 * Lvect[r][i0];

                int inc = 1;
                Blas<CoefficientPrecision>::axpy(&size_s, &alpha, Uvect[r].data(), &inc, uk.data(), &inc);
            }
            if (uk[j0] == 0.0) {

                toosoon = true;
                break;
            }
            double alpha = 1.0 / uk[j0];
            // std::transform(uk.begin(), uk.end(), uk.begin(), [alpha](double x) { return alpha * x; });
            for (auto &val : uk)
                val *= alpha;
            std::vector<CoefficientPrecision> ej(size_s);
            // ej[j0 - ofs] = 1.0;
            ej[j0] = 1.0;

            std::vector<CoefficientPrecision> lk(sizet);
            this->mat_vect(ej, lk, 'N');

            for (int r = 0; r < rk; ++r) {
                auto alpha = -1 * Uvect[r][j0];
                int inc    = 1;
                Blas<CoefficientPrecision>::axpy(&sizet, &alpha, Lvect[r].data(), &inc, lk.data(), &inc);
            }
            Uvect.push_back(uk);
            Lvect.push_back(lk);
            nr                 = norm2(lk) * norm2(uk);
            stopping_criterion = ((size_s - rk) * nr > tolerance);
            rk += 1;
        }
        if (rk == 0) {
            Matrix<CoefficientPrecision> L(sizet, 0);
            Matrix<CoefficientPrecision> U(0, size_s);
            res.push_back(L);
            res.push_back(U);
            flagzeros = true;
            return res;
        } else if (rk >= std::min(sizet, size_s) / 2 || toosoon) {
            flagout = false;
            return res;
        } else {
            Matrix<CoefficientPrecision> L(sizet, rk);
            // Matrix<CoefficientPrecision> Ut(size_s, rk);
            Matrix<CoefficientPrecision> U(rk, size_s);

            for (int k = 0; k < rk; ++k) {
                std::copy(Lvect[k].begin(), Lvect[k].end(), L.data() + k * sizet); // Évite la boucle
                for (int j = 0; j < size_s; ++j) {
                    U(k, j) = Uvect[k][j]; // Remplissage direct de U
                }
                // std::copy(Uvect[k].begin(), Uvect[k].end(), Ut.data() + k * size_s); // Évite la boucle
            }
            // Matrix<CoefficientPrecision> U(rk, size_s);
            // transpose(Ut, U);
            res.push_back(L);
            res.push_back(U);
            return res;
        }
    }

    CoefficientPrecision get_coeff(const int &i, const int &j) const {
        // Début et fin des colonnes pour la ligne i
        int row_start = row_csr[i];
        int row_end   = row_csr[i + 1];

        // Parcourir les colonnes associées à la ligne i
        for (int idx = row_start; idx < row_end; idx++) {
            if (col_csr[idx] == j) {
                // Si la colonne j est trouvée, retourner la valeur correspondante
                return data[idx];
            }
        }
        // Si la colonne j n'est pas trouvée, retourner 0.0
        return 0.0;
    }

    // void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
    //     const auto &target_permutation = target.get_permutation();
    //     const auto &source_permutation = source.get_permutation();
    //     if (N == 1) {
    //         std::vector<CoefficientPrecision> ej(source.get_size());
    //         ej[col_offset] = 1.0;
    //         // std::vector<CoefficientPrecision> colj(traget.get_size());
    //         std::vector<CoefficientPrecision> colj(target.get_size());

    //         this->mat_vect(ej, colj);
    //         for (int i = 0; i < M; ++i) {
    //             // double rand;
    //             // generate_random_scalar(rand, -1.0, 1.0);
    //             // ptr[i] = colj[i + row_offset] + rand / (1.0 + std::abs(1.0 * (i + row_offset - col_offset)));
    //             ptr[i] = colj[i + row_offset];
    //         }
    //     } else if (M == 1) {
    //         std::vector<CoefficientPrecision> ei(target.get_size());
    //         ei[row_offset] = 1.0;
    //         std::vector<CoefficientPrecision> rowi(source.get_size());
    //         this->mat_vect_bloc(target.get_size(), source.get_size(), 0, 0, ei, rowi, 'T');
    //         for (int j = 0; j < N; ++j) {
    //             // double rand;
    //             // generate_random_scalar(rand, -1.0, 1.0);
    //             // ptr[j] = rowi[j + col_offset] + rand / (1.0 + std::abs(1.0 * (row_offset - j - col_offset)));
    //             ptr[j] = rowi[j + col_offset];
    //         }
    //     } else {
    //         for (int i = 0; i < M; ++i) {
    //             for (int j = 0; j < N; ++j) {
    //                 // double rand;
    //                 // generate_random_scalar(rand, -1.0, 1.0);
    //                 // ptr[i + M * j] = this->get_coeff(i + row_offset, j + col_offset) + rand / (1.0 + std::abs(1.0 * (i + row_offset - j - col_offset)));
    //                 ptr[i + M * j] = this->get_coeff(i + row_offset, j + col_offset);
    //             }
    //         }
    //         // for (int j = 0; j < N; ++i) {
    //         //     std::vector<CoefficientPrecision> ej(target.get_size());
    //         //     ej[j + col_offset] = 1.0;
    //         //     std::vector<CoefficientPrecision>
    //         //     this->mat_vect_bloc(M, source.get_size(), row_offset, col_offset)
    //         // }
    //     }
    // }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        if (N == 1) {
            std::vector<CoefficientPrecision> ej(source.get_size());
            ej[col_offset] = 1.0;
            std::vector<CoefficientPrecision> colj(target.get_size());

            this->mat_vect(ej, colj);
            for (int i = 0; i < M; ++i) {
                // double rand;
                // generate_random_scalar(rand, -1.0, 1.0);
                // ptr[i] = colj[i + row_offset] + rand / (1.0 + std::abs(1.0 * (i + row_offset - col_offset)));
                ptr[i] = colj[i + row_offset - target.get_offset()];
            }
        } else if (M == 1) {
            std::vector<CoefficientPrecision> ei(target.get_size());
            ei[row_offset] = 1.0;
            std::vector<CoefficientPrecision> rowi(source.get_size());
            this->mat_vect_bloc(target.get_size(), source.get_size(), 0, 0, ei, rowi, 'T');
            for (int j = 0; j < N; ++j) {
                ptr[j] = rowi[j + col_offset - source.get_offset()];
            }
        } else {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    ptr[i + M * j] = this->get_coeff(i + row_offset - target.get_offset(), j + col_offset - source.get_offset());
                }
            }
        }
    }
    // int nb_rows() {
    //     return target.get_size();
    // }
    // int nb_cols() {
    //     return source.get_size();
    // }
    // ///////////////////////////////// LINALG pour affecter des CSR au feuilles denses
    // void add_matrix_vector_product(char trans, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out);
    // void add_matrix_matrix_product(char transa, char transb, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const Matrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C);
};
// template <class CoefficientPrecision, class CoordinatePrecision>
// void CSR_generator<CoefficientPrecision, CoordinatePrecision>::add_matrix_vector_product(char trans, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {
//     mat_vect(in, out, trans);
// }

// template <class CoefficientPrecision, class CoordinatePrecision>
// void CSR_generator<CoefficientPrecision, CoordinatePrecision>::add_matrix_matrix_product(char transa, char transb, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const Matrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {
//     if (transa == 'N') {
//         if (transb == 'N') {
//             for (int k = 0; k < B.nb_cols(); ++k) {
//                 if (B.data().size() < B.nb_rows() * B.nb_cols()) {
//                     // B est csr

//                 } else {
//                     CoefficientPrecision *colB_k = std::copy(B.data() + k * B.nb_rows(), B.data() + (k + 1) * B.nb_rows());
//                     std::vector<CoefficientPrecision> col_Ck(C.data() + k * C.nb_rows(), C.data() + (k + 1) * C.nb_rows());
//                     add_matrix_vector_product('N', alpha, A, colB_k, beta, col_Ck);
//                 }
//             }
//         }
//     }
// }

template <class CoefficientPrecision, class CoordinatePrecision>
void build_hmatrix(const CSR_generator<double, double> &generator, HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix) {
    auto &t  = hmatrix->get_target_cluster();
    auto &s  = hmatrix->get_source_cluster();
    bool adm = hmatrix->compute_admissibility(t, s);
    if (adm) {
        bool flag      = true;
        bool flagzeros = false;
        // auto res  = generator.rd_ACA(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), hmatrix->get_epsilon(), flag);
        // auto res = generator.full_ACA(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), hmatrix->get_epsilon(), flag);
        // auto res = generator.full_ACA(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), hmatrix->get_epsilon(), flag, flagzeros);
        auto res = zero_map(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), flag);

        if (flag) {
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(res[0], res[1]);
            hmatrix->set_low_rank_data(lr);
        } else {
            if (t.get_children().size() > 0) {
                for (auto &t_child : t.get_children()) {
                    for (auto &s_child : s.get_children()) {
                        auto child = hmatrix->add_child(t_child.get(), s_child.get());
                        build_hmatrix(generator, child);
                    }
                }
            } else {
                Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
                generator.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), dense.data());
                hmatrix->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
            }
        }
    } else {
        if (t.get_children().size() > 0) {
            for (auto &t_child : t.get_children()) {
                for (auto &s_child : s.get_children()) {
                    auto child = hmatrix->add_child(t_child.get(), s_child.get());
                    build_hmatrix(generator, child);
                }
            }
        } else {
            Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
            generator.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), dense.data());
            hmatrix->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
        }
    }
}

template <class CoefficientPrecision, class CoordinatePrecision>
void build_hmatrix(const CSR_generator<double, double> &generator, HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix, const int &ref, std::unique_ptr<double> &cpt, std::unique_ptr<int> &trash, const int maxblocksize) {
    auto &t  = hmatrix->get_target_cluster();
    auto &s  = hmatrix->get_source_cluster();
    bool adm = false;
    if (t.get_size() < maxblocksize) {
        adm = hmatrix->compute_admissibility(t, s);
    }
    // double tempp = *cpt;
    // int repp     = ((tempp * 20) / ref) + 1;
    // if (repp > *trash) {
    //     std::cout << "\r[";
    //     for (int kk = 0; kk < repp; ++kk) {
    //         std::cout << "#";
    //     }
    //     for (int kk = repp; kk < 20; ++kk) {
    //         std::cout << "_";
    //     }
    //     std::cout << "]" << std::flush;
    //     *trash += 1;
    // }

    if (adm) {
        bool flag      = true;
        bool flagzeros = false;
        // auto res  = generator.rd_ACA(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), hmatrix->get_epsilon(), flag);
        // auto res = generator.full_ACA(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), hmatrix->get_epsilon(), flag);
        // auto res = generator.full_ACA(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), hmatrix->get_epsilon(), flag, flagzeros);
        auto res = generator.zero_map(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), flag);

        if (flag) {
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(res[0], res[1]);
            hmatrix->set_low_rank_data(lr);
            // double temp                     = tempp + t.get_size() * s.get_size();
            // std::unique_ptr<double> new_cpt = std::move(temp);
            // *cpt += t.get_size() * s.get_size();
        } else {
            if (t.get_children().size() > 0) {
                for (auto &t_child : t.get_children()) {
                    for (auto &s_child : s.get_children()) {
                        auto child = hmatrix->add_child(t_child.get(), s_child.get());
                        build_hmatrix(generator, child, ref, cpt, trash, maxblocksize);
                    }
                }
            } else {
                Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
                generator.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), dense.data());
                hmatrix->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
                // double temp                     = tempp + t.get_size() * s.get_size();
                // std::unique_ptr<double> new_cpt = std::move(temp);
                // *cpt += t.get_size() * s.get_size();
            }
        }
    } else {
        if (t.get_children().size() > 0) {
            for (auto &t_child : t.get_children()) {
                for (auto &s_child : s.get_children()) {
                    auto child = hmatrix->add_child(t_child.get(), s_child.get());
                    build_hmatrix(generator, child, ref, cpt, trash, maxblocksize);
                }
            }
        } else {
            Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
            generator.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), dense.data());
            hmatrix->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
            // double temp                     = tempp + t.get_size() * s.get_size();
            // std::unique_ptr<double> new_cpt = std::move(temp); //
            // *cpt += t.get_size() * s.get_size();
        }
    }
}

template <class CoefficientPrecision, class CoordinatePrecision>
void build_hmatrix_smart(const CSR_generator<double, double> &generator, HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix, const double &ref, double &cpt, int &trash) {
    auto &t  = hmatrix->get_target_cluster();
    auto &s  = hmatrix->get_source_cluster();
    bool adm = false;
    if (t.get_size() < 60000) {

        adm = hmatrix->compute_admissibility(t, s);
    }
    auto son_generator = generator.get_sub_csr(t, s);
    // int repp           = ((cpt * 20) / ref) + 1;
    double repp = ((cpt * 20) / ref) + 1;

    if (repp > trash) {
        std::cout << "\r[";
        for (int kk = 0; kk < repp; ++kk) {
            std::cout << "#";
        }
        for (int kk = int(repp); kk < 20; ++kk) {
            std::cout << "_";
        }
        std::cout << "]" << std::flush;
        trash += 1;
    }
    if (son_generator.data_size() == 0) {
        Matrix<CoefficientPrecision> A(t.get_size(), 0);
        Matrix<CoefficientPrecision> B(0, s.get_size());
        LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(A, B);
        hmatrix->set_low_rank_data(lr);
        cpt += t.get_size() * s.get_size();
    } else {
        // std::cout << cpt << ',' << ref << ',' << repp << std::endl;
        if (adm) {
            // bool flag = son_generator.is_zero();
            bool flag      = true;
            bool flagzeros = false;
            auto res       = son_generator.full_ACA_sub(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), hmatrix->get_epsilon(), flag, flagzeros);

            if (flag) {
                Matrix<CoefficientPrecision> A(t.get_size(), 0);
                Matrix<CoefficientPrecision> B(0, s.get_size());
                LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(A, B);
                hmatrix->set_low_rank_data(lr);
                cpt += t.get_size() * s.get_size();
            } else {
                if (t.get_children().size() > 0) {
                    for (auto &t_child : t.get_children()) {
                        for (auto &s_child : s.get_children()) {
                            auto child = hmatrix->add_child(t_child.get(), s_child.get());
                            build_hmatrix_smart(son_generator, child, ref, cpt, trash);
                        }
                    }
                } else {
                    Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
                    son_generator.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), dense.data());
                    hmatrix->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
                    // std::cout << "feuille de nse de taille : " << t.get_size() * s.get_size() << " contient nnz :  " << son_generator.data_size() << std::endl;
                    cpt += t.get_size() * s.get_size();
                }
            }
        } else {
            if (t.get_children().size() > 0) {
                for (auto &t_child : t.get_children()) {
                    for (auto &s_child : s.get_children()) {
                        auto child = hmatrix->add_child(t_child.get(), s_child.get());
                        build_hmatrix_smart(son_generator, child, ref, cpt, trash);
                    }
                }
            } else {
                Matrix<CoefficientPrecision> dense(t.get_size(), s.get_size());
                son_generator.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), dense.data());
                hmatrix->set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(dense));
                // std::cout << "feuille dense de taille : " << t.get_size() * s.get_size() << " contient nnz :  " << son_generator.data_size() << std::endl;

                cpt += t.get_size() * s.get_size();
            }
        }
    }
}
double random_value() {
    // static std::random_device rd;                     // Graine aléatoire
    // static std::mt19937 gen(rd());                    // Mersenne Twister comme générateur
    // static std::normal_distribution<> dist(0.0, 1.0); // Distribution normale

    // Retourner un nombre aléatoire suivant N(0,1)
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    double number = distribution(generator);
    return number;
}
template <class CoefficientPrecision, class CoordinatePrecision>
class CSR_generator_random : public VirtualGenerator<CoefficientPrecision> {
  private:
    std::vector<double> data;
    std::vector<double> row_csr;
    std::vector<double> col_csr;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;
    const std::vector<CoordinatePrecision> &points;
    const int dim;

  public:
    // CSR_generator(std::vector<double> &data0, std::vector<double> &row_csr0, std::vector<double> &col_csr0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : data(data0), row_csr(row_csr0), col_csr(col_csr0), target(target0), source(source0) {}

    CSR_generator_random(const Cluster<CoordinatePrecision> &target0,
                         const Cluster<CoordinatePrecision> &source0,
                         const std::vector<double> &row_csr0,
                         const std::vector<double> &col_csr0,
                         const std::vector<double> &val_csr0,
                         const std::vector<CoordinatePrecision> &points0,
                         const int &dim0) : target(target0), source(source0), points(points0), dim(dim0) {

        int M = target0.get_size(); // nombre de lignes après permutation
        int N = source0.get_size(); // nombre de colonnes après permutation

        // Initialisation des nouvelles structures CSR
        row_csr.resize(M + 1, 0); // row_csr_out contient les décalages pour chaque ligne
        int rep = 0;
        // Boucle sur les lignes permutées
        for (int i = 0; i < M; i++) {
            int row_perm = target0.get_permutation()[i];
            int r1       = row_csr0[row_perm];     // début de la ligne dans row_csr
            int r2       = row_csr0[row_perm + 1]; // fin de la ligne dans row_csr

            // Boucle sur les colonnes permutées pour la ligne courante
            int flag = 0;
            for (int j = 0; j < N; j++) {
                int col_perm = source0.get_permutation()[j];

                // Extraire les colonnes et les valeurs de la ligne permutée
                std::vector<int> row_iperm(r2 - r1);
                std::copy(col_csr0.begin() + r1, col_csr0.begin() + r2, row_iperm.begin());

                // Rechercher l'indice de la colonne permutée
                auto it = std::find(row_iperm.begin(), row_iperm.end(), col_perm);
                if (it != row_iperm.end()) {
                    int idx = std::distance(row_iperm.begin(), it);
                    col_csr.push_back(j);               // colonne après permutation
                    data.push_back(val_csr0[r1 + idx]); // valeur associée
                    flag += 1;
                }
            }
            row_csr[i + 1] = row_csr[i] + flag;
        }
    }
    CoefficientPrecision get_coeff(const int &i, const int &j) const {
        // Début et fin des colonnes pour la ligne i
        int row_start = row_csr[i];
        int row_end   = row_csr[i + 1];

        // Parcourir les colonnes associées à la ligne i
        for (int idx = row_start; idx < row_end; idx++) {
            if (col_csr[idx] == j) {
                // Si la colonne j est trouvée, retourner la valeur correspondante
                return data[idx];
            }
        }
        // std::cout << "radom =";
        auto rd = random_value();
        // std::vector<CoefficientPrecision> p1(dim);
        // std::vector<CoefficientPrecision> p2(dim);
        // int iperm = target.get_permutation()[i];
        // int jperm = source.get_permutation()[j];
        // for (int k = 0; k < dim; ++k) {
        //     p1[k] = points[dim * iperm + k];generator
        auto nr = 1.0 * rd / std::pow(std::abs(i - j), 2.0);
        return nr;
    }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        // std::cout << "begin copy_sub" << std::endl;
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                ptr[i + M * j] = this->get_coeff(i + row_offset, j + col_offset);
            }
        }
        // std::cout << "end copy sub" << std::endl;
    }
};
template <class CoefficientPrecision, class CoordinatePrecision>
class my_generator : public VirtualGenerator<CoefficientPrecision> {
  private:
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;
    const std::vector<CoordinatePrecision> points;

  public:
    my_generator(const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0, const std::vector<CoordinatePrecision> &points0) : target(target0), source(source0), points(points0) {}
    CoefficientPrecision get_coeff(const int &i, const int &j) const {
        int iperm = target.get_permutation()[i];
        int jperm = source.get_permutation()[j];
        auto nr   = (points[3 * iperm] - points[3 * jperm]) * (points[3 * iperm] - points[3 * jperm]) + (points[3 * iperm + 1] - points[3 * jperm + 1]) * (points[3 * iperm + 1] - points[3 * jperm + 1]) + (points[3 * iperm + 2] - points[3 * jperm + 2]) * (points[3 * iperm + 2] - points[3 * jperm + 2]);
        return 1.0 / (0.1 + nr);
    }
    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                ptr[i + M * j] = this->get_coeff(i + row_offset, j + col_offset);
            }
        }
    }
};

template <class CoefficientPrecision, class CoordinatePrecision>
class Matricegenerator : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    Matricegenerator(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0) {}

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = mat(target_permutation[i + row_offset], source_permutation[j + col_offset]);
            }
        }
    }

    Matrix<CoefficientPrecision> get_mat() { return mat; }

    Matrix<CoefficientPrecision> get_perm_mat() {
        CoefficientPrecision *ptr = new CoefficientPrecision[target.get_size() * source.get_size()];
        this->copy_submatrix(target.get_size(), target.get_size(), 0, 0, ptr);
        Matrix<CoefficientPrecision> M;
        M.assign(target.get_size(), source.get_size(), ptr, true);
        return M;
    }
    Matrix<CoefficientPrecision> get_unperm_mat() {
        CoefficientPrecision *ptr      = new CoefficientPrecision[target.get_size() * source.get_size()];
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < target.get_size(); i++) {
            for (int j = 0; j < source.get_size(); j++) {
                ptr[target_permutation[i] + source_permutation[j] * target.get_size()] = mat(i, j);
            }
        }
        Matrix<CoefficientPrecision> M;
        M.assign(target.get_size(), source.get_size(), ptr, true);
        return M;
    }
};
template <class CoefficientPrecision, class CoordinatePrecision>
Matrix<CoefficientPrecision> get_unperm_mat(const Matrix<CoefficientPrecision> &mat, Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source) {
    CoefficientPrecision *ptr      = new CoefficientPrecision[target.get_size() * source.get_size()];
    const auto &target_permutation = target.get_permutation();
    const auto &source_permutation = source.get_permutation();
    for (int i = 0; i < target.get_size(); i++) {
        for (int j = 0; j < source.get_size(); j++) {
            ptr[target_permutation[i] + source_permutation[j] * target.get_size()] = mat(i, j);
        }
    }
    Matrix<CoefficientPrecision> M;
    M.assign(target.get_size(), source.get_size(), ptr, false);
    return M;
}
template <class CoefficientPrecision, class CoordinatePrecision>
class Matricegenerator_noperm : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    Matricegenerator_noperm(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0) {}

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = mat(i, j);
            }
        }
    }

    // Matrix<CoefficientPrecision> get_mat() { return mat; }

    // Matrix<CoefficientPrecision> get_perm_mat() {
    //     CoefficientPrecision *ptr = new CoefficientPrecision[target.get_size() * source.get_size()];
    //     this->copy_submatrix(target.get_size(), target.get_size(), 0, 0, ptr);
    //     Matrix<CoefficientPrecision> M;
    //     M.assign(target.get_size(), source.get_size(), ptr, true);
    //     return M;
    // }
    // Matrix<CoefficientPrecision> get_unperm_mat() {
    //     CoefficientPrecision *ptr      = new CoefficientPrecision[target.get_size() * source.get_size()];
    //     const auto &target_permutation = target.get_permutation();
    //     const auto &source_permutation = source.get_permutation();
    //     for (int i = 0; i < target.get_size(); i++) {
    //         for (int j = 0; j < source.get_size(); j++) {
    //             ptr[target_permutation[i] + source_permutation[j] * target.get_size()] = mat(i, j);
    //         }
    //     }
    //     Matrix<CoefficientPrecision> M;
    //     M.assign(target.get_size(), source.get_size(), ptr, true);
    //     return M;
    // }
};

Matrix<double> extract(const int &nr, const int &nc, const int &ofr, const int &ofc, Matrix<double> M) {
    Matrix<double> res(nr, nc);
    for (int k = 0; k < nr; ++k) {
        for (int l = 0; l < nc; ++l) {
            res(k, l) = M(k + ofr, l + ofc);
        }
    }
    return res;
}
// Champ de vecteur pour nos test
template <class CoordinatePrecision>
std::vector<CoordinatePrecision> my_vector(const std::vector<CoordinatePrecision> X) {
    std::vector<CoordinatePrecision> res(X.size());
    // double nr  = std::pow(1.0 + (600.0 - X[0]) * (600.0 - X[0]) / 90000.0, 0.5);

    // res[0] = 1.0;
    // res[1] = (600.0 - X[0]) / (300.0);
    double alpha = 1.0;
    res[0]       = 1.0;
    res[1]       = sin(3.14 * alpha * X[0] / 1200.0);
    return res;
}

template <class CoordinatePrecision>
std::vector<CoordinatePrecision> transform_point(std::vector<CoordinatePrecision> (*func)(std::vector<CoordinatePrecision>), const std::vector<CoordinatePrecision> &x0, const CoordinatePrecision &xmin, double &delta) {
    auto current_point = x0;
    auto dist          = 0.0;
    // std::cout << "________________________________" << std::endl;
    while (xmin <= current_point[0]) {
        // auto next_point = current_point - delta * func(current_point);
        std::vector<CoordinatePrecision> vect = func(current_point);
        // std::cout << current_point[0] << ',' << current_point[1] << '|' << vect[0] << ',' << vect[1] << std::endl;
        // double nr                             = 0;
        // for (auto &v : vect) {
        //     nr += (v * v);
        // }
        // auto n = std::pow(1.0 * nr, 0.5);
        // auto vect_norm = vect / norm2(vect);
        // std::cout << "vect :  " << vect[0] << ',' << vect[1] << " norme :" << norm2(vect) << " norme vect : " << vect_norm[0] << ',' << vect_norm[1] << std::endl;
        // std::cout << "norme de v " << n << std::endl;
        // for (int k = 0; k < vect.size(); ++k) {
        //     vect[k] = vect[k] / (1.0 * n);
        // }
        // auto next_point = current_point - multiply_vector_by_scalar(vect_norm, delta * 1.0);
        // auto next_x = current_point[0] - delta * vect[0];
        // auto next_y = current_point[1] - delta * vect[1];
        // // auto next_point = current_point - multiply_vector_by_scalar(vect, delta * 1.0);
        // std::vector<CoordinatePrecision> next_point(2);
        // next_point[0] = next_x;
        // next_point[1] = next_y;
        // auto next_point = current_point - multiply_vector_by_scalar(vect, delta);
        auto next_point = current_point - mult(delta, vect);

        dist += delta;
        // std::cout << "next point : " << next_point[0] << ',' << next_point[1] << std::endl;
        current_point = next_point;
    }
    double scale = 1.0;
    std::vector<CoordinatePrecision> res(current_point);
    res[0] = scale * dist;
    res[1] = current_point[1];
    return res;
}

std::vector<double> generate_random_vector(int size) {
    std::random_device rd;  // Source d'entropie aléatoire
    std::mt19937 gen(rd()); // Générateur de nombres pseudo-aléatoires

    std::uniform_real_distribution<double> dis(1.0, 10.0); // Plage de valeurs pour les nombres aléatoires (ici de 1.0 à 10.0)

    std::vector<double> random_vector;
    random_vector.reserve(size); // Allocation de mémoire pour le vecteur

    for (int i = 0; i < size; ++i) {
        random_vector.push_back(dis(gen)); // Ajout d'un nombre aléatoire dans la plage à chaque itération
    }

    return random_vector;
}

template <class CoordinatePrecision>
std::vector<std::vector<CoordinatePrecision>> trajectoire(std::vector<CoordinatePrecision> (*func)(std::vector<CoordinatePrecision>), const std::vector<CoordinatePrecision> &x0, const CoordinatePrecision &xmin, double &delta) {
    auto current_point = x0;
    auto dist          = 0.0;
    std::vector<std::vector<CoordinatePrecision>> traj;
    traj.push_back(x0);
    // auto x = current_point[0];
    // auto y = current_point[1];
    while (current_point[0] >= xmin) {
        // auto next_point = current_point - delta * func(current_point);
        std::vector<CoordinatePrecision> vect = func(current_point);
        // double nr                             = 0;
        // for (auto &v : vect) {
        //     nr += (v * v);
        // }
        // auto n = std::pow(1.0 * nr, 0.5);
        auto vect_norm = vect / norm2(vect);
        // std::cout << "norme de v " << n << std::endl;
        // for (int k = 0; k < vect.size(); ++k) {
        //     vect[k] = vect[k] / (1.0 * n);
        // }
        // auto next_point = current_point - multiply_vector_by_scalar(vect_norm, delta * 1.0);
        auto next_point = current_point - multiply_vector_by_scalar(vect, delta);
        traj.push_back(next_point);
        dist += delta;
        // std::cout << "next point : " << next_point[0] << ',' << next_point[1] << std::endl;
        current_point = next_point;
    }
    return traj;
}

template <class CoordinatePrecision>
std::vector<CoordinatePrecision> transform_cluster(std::vector<CoordinatePrecision> (*func)(std::vector<CoordinatePrecision>), const std::vector<CoordinatePrecision> &points, const CoordinatePrecision &xmin, double &delta) {
    // std::vector<std::vector<CoordinatePrecision>> res(points.size());
    std::vector<CoordinatePrecision> res(points.size());
    // for (auto &x : points) {
    for (int k = 0; k < points.size() / 2; ++k) {
        std::vector<CoordinatePrecision> x(2);
        x[0]       = points[2 * k];
        x[1]       = points[2 * k + 1];
        auto new_x = transform_point(func, x, xmin, delta);
        // res[rep] = new_x ;
        res[2 * k]     = new_x[0];
        res[2 * k + 1] = new_x[1];
    }
    return res;
}

void save_points_to_csv(const std::vector<std::vector<double>> &points, const std::string &filename) {
    // Ouvrir le fichier en mode écriture
    std::ofstream file(filename);

    // Vérifier que le fichier est ouvert correctement
    if (!file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << std::endl;
        return;
    }

    // Parcourir chaque point (vecteur de doubles)
    for (const auto &point : points) {
        // Parcourir les coordonnées d'un point
        for (int i = 0; i < point.size(); ++i) {
            file << point[i];
            if (i < point.size() - 1) {
                file << ","; // Ajouter une virgule entre les coordonnées
            }
        }
        file << "\n"; // Nouvelle ligne après chaque point
    }

    // Fermer le fichier
    file.close();
    std::cout << "Les points ont été sauvegardés dans le fichier " << filename << std::endl;
}

template <class CoordinatePrecision>
std::vector<std::vector<CoordinatePrecision>> generate_grid(const int N, const int space_dim, const std::vector<std::pair<CoordinatePrecision, CoordinatePrecision>> &limits) {
    std::vector<std::vector<CoordinatePrecision>> res(std::pow(N * 1.0, space_dim * 1.0));
    if (space_dim == 2) {
        auto xm = limits[0].first;
        auto xM = limits[0].second;
        auto ym = limits[0].first;
        auto yM = limits[0].second;
        for (int k = 0; k < N; ++k) {
            for (int l = 0; l < N; ++l) {
                std::vector<CoordinatePrecision> point(2);
                auto xx        = xm + k * (xM - xm) * 1.0 / (N * 1.0);
                auto yy        = ym + l * (yM - ym) * 1.0 / (N * 1.0);
                point[0]       = xx;
                point[1]       = yy;
                res[k + N * l] = point;
            }
        }
    } else {
        std::cerr << "mauvaise space dim" << std::endl;
    }
    return res;
}

void hello() {
    //     std::cout << "         |===" << std::endl;
    //     std::cout << "        /|\\" << std::endl;
    //     std::cout << "       / | \\" << std::endl;
    //     std::cout << "      /  |  \\" << std::endl;
    //     std::cout << "     /   |   \\" << std::endl;
    //     std::cout << "    /    |    \\" << std::endl;
    //     std::cout << "___/_____|_____\\_" << std::endl;
    //     std::cout << " \\       |        /" << std::endl;
    //     std::cout << "~~\\~~~~~~~~~~~~~~/~~~~~~~~" << std::endl;
    //     std::cout << "    TEST HLU MATRICES LEO  " << std::endl;
    //     std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    std::cout << "   _ " << std::endl;
    std::cout << "  / | " << std::endl;
    std::cout << "/    |_" << std::endl;
    std::cout << "/_  -  |  |                       ,:',:`,:' " << std::endl;
    std::cout << "/ / /     |                     __||_||_||_||___" << std::endl;
    std::cout << "|    |     / |             ____['''''''''''''''']___" << std::endl;
    std::cout << "/   /     /   |           / ' '''''''''''''''''''' / " << std::endl;
    std::cout << "~~^~^~^~^~^~~^~^~^~~^~^~^~~^~^~^~^~^~^^~^~^~^~^~^~^~^~~^~^~^~^~~^~^" << std::endl;
    std::cout << "    TEST HLU MATRICES LEO  " << std::endl;
    std::cout << "~~^~^~^~^~^~~^~^~^~~^~^~^~~^~^~^~^~^~^^~^~^~^~^~^~^~^~~^~^~^~^~~^~^" << std::endl;
}
int main() {
    hello();
    /// TEST AVEC HTOOL ET LES MATRICES DE LEO
    for (int ss = 3; ss < 20; ++ss) {
        double epsilon = 1e-6;
        double eta     = 4;
        // int sizex      = 50 * ss;
        // int sizey      = 50 * ss;
        int sizex = 40 * ss;
        int sizey = 40 * ss;
        // int minclusize = ((sizex + 50) / 100) * 100;
        int minclusize = sizex / 10;
        // for (int minclusize = 45; minclusize < 65; ++minclusize) {
        // int minclusize = 60;

        std::cout << "______________________________ minclustersize : " << minclusize << std::endl;
        // int minclusize = 50;

        // int sizex = 100;
        // int sizey = 100;
        int size = sizex * sizey;

        std::cout << "epsilon et eta : " << epsilon << ',' << eta << std::endl;
        std::cout << "_______________________________________________________" << std::endl;
        std::cout << "sizex x sizey  = " << sizex << " x " << sizey << std::endl;

        ///////////////////////
        ////// GENERATION MAILLAGE  :  size points en 2 dimensions

        std::vector<double> p(2 * size);

        for (int j = 0; j < sizex; j++) {
            for (int k = 0; k < sizey; k++) {

                p[2 * (j + k * sizex) + 0] = j * 1200 / sizex;
                p[2 * (j + k * sizex) + 1] = k * 1200 / sizey;
            }
        }

        // // Recuperation du CSR
        for (int Eps = 2; Eps < 3; ++Eps) {
            int eps = 2 * Eps;
            std::cout << "______________________________" << std::endl;
            std::cout << "eps = " << eps << std::endl;
            auto csr_file = get_csr("/work/sauniear/Documents/Code_leo/VersionLight2/size_" + std::to_string(sizex) + "x" + std::to_string(sizey) + "_alpha1_epsilon" + std::to_string(eps) + ".csv");

            auto val = csr_file[0];
            auto row = csr_file[1];
            auto col = csr_file[2];
            std::cout << "csr size : " << row.size() << ',' << col.size() << "     ,     " << val.size() << std::endl;

            ///////////////////////////////////////////////////////////////////////////////////

            //// MAILLAGE DEFORME
            double limit0 = 0.0;
            double delta  = 1.0;

            std::vector<double> p_deformed = transform_cluster(my_vector, p, limit0, delta);
            std::vector<std::vector<double>> deformed;
            std::vector<double> def;
            std::vector<double> points_ref;
            for (int k = 0; k < size; ++k) {
                std::vector<double> poin(2);
                poin[0]   = p_deformed[2 * k];
                poin[1]   = p_deformed[2 * k + 1];
                double zz = 0.0;
                deformed.push_back(poin);
                def.push_back(p_deformed[2 * k]);
                def.push_back(p_deformed[2 * k + 1]);
                def.push_back(zz);
            }
            std::cout << "pdeformed ok " << std::endl;
            // Creation clusters bande
            std::vector<double> bb(3);
            bb[0] = 1;
            bb[1] = 0;
            std::vector<double> bperp(3);
            bperp[0] = 0;
            bperp[1] = 1;
            ClusterTreeBuilder<double> directional_build;
            // directional_build.set_minclustersize(100);
            directional_build.set_minclustersize(minclusize);

            // CHOIX DU SPLITTING
            std::shared_ptr<ConstantDirection<double>> strategy_directional = std::make_shared<ConstantDirection<double>>(bperp);
            // std::shared_ptr<MixedDirection<double>> strategy_directional = std::make_shared<MixedDirection<double>>(bperp, sizex);

            // a commenter si on veut un splitting normal
            directional_build.set_direction_computation_strategy(strategy_directional);
            std::cout << "build cluster tree " << std::endl;
            std::shared_ptr<Cluster<double>> root_directional = std::make_shared<Cluster<double>>(directional_build.create_cluster_tree(def.size() / 3, 3, def.data(), 2, 2));
            std::cout << "cluster tree ok " << std::endl;

            // HMATRIX BUILDER
            HMatrixTreeBuilder<double, double> hmatrix_directional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);
            // fullACA<double> compressor_fullACA;
            // hmatrix_directional_builder.set_low_rank_generator(std::make_shared<fullACA<double>>(compressor_fullACA)) ;

            // hmatrix_directional_builder.set_low_rank_generator()

            // CONDITION D ADMISSIBILITE
            std::shared_ptr<Bandee<double>>
                directional_admissibility = std::make_shared<Bandee<double>>(bb, def);
            // std::shared_ptr<Mixed_Bandee<double>>
            //     directional_admissibility = std::make_shared<Mixed_Bandee<double>>(bb, def, sizex);

            // std::shared_ptr<Bande<double>> directional_admissibility = std::make_shared<Bande<double>>(bb, root_directional->get_permutation());

            // a commenter si on veut la condition standard
            hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);

            /////////////////////////////////////
            // MATRICE QU ON DONNE A HTOOL
            std::cout << "récupération CSR" << std::endl;

            auto start = std::chrono::high_resolution_clock::now();

            CSR_generator<double, double> gen_csr(*root_directional, *root_directional, row, col, val);

            auto stop     = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
            std::cout << "création générateur : " << duration.count() << std::endl;
            // std::cout << "build hmat " << std::endl;
            // start                    = std::chrono::high_resolution_clock::now();
            // auto hmatrix_directional = hmatrix_directional_builder.build(gen_csr);
            // stop                     = std::chrono::high_resolution_clock::now();
            // auto duration_hmat       = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
            // std::cout << " build ok : " << duration_hmat.count() << std::endl;
            // // hmatrix_directional.save_plot("hmatrix_directional");

            // std::cout << "compression directional : " << hmatrix_directional.get_compression() << std::endl;
            /// décommenter si on veut v"érifier l'eereur sur la hmatrice
            // Matrix<double> reference(size, size);
            // gen_csr.copy_submatrix(size, size, 0, 0, reference.data());
            // Matrix<double> directional_dense(size, size);
            // copy_to_dense(hmatrix_directional, directional_dense.data());
            // std::cout << "erreur sur hmat : " << normFrob(directional_dense - reference) << "  pour " << normFrob(reference) << std::endl;

            HMatrix<double, double> no_htool(*root_directional, *root_directional);
            no_htool.set_epsilon(epsilon);
            no_htool.set_eta(eta);
            // no_htool.set_low_rank_generator(hmatrix_directional.get_low_rank_generator());
            no_htool.set_admissibility_condition(directional_admissibility);

            // build_hmatrix(gen_csr, &no_htool);
            double S  = (1.0 * size) * (1.0 * size);
            double zz = 0.0;
            int Z     = 0;
            // std::unique_ptr<double> cpt = std::make_unique<double>(0.0);
            // std::unique_ptr<int> rep    = std::make_unique<int>(0);
            // build_hmatrix(gen_csr, &no_htool, S, cpt, rep, 4500);
            std::cout << "size: " << S << ',' << size << std::endl;
            build_hmatrix_smart(gen_csr, &no_htool, S, zz, Z);
            // Matrix<double> no_dense(size, size);
            // Matrix<double> gen_dense(size, size);
            // copy_to_dense(no_htool, no_dense.data());
            // gen_csr.copy_submatrix(size, size, 0, 0, gen_dense.data());
            // std::cout << " ???? " << normFrob(no_dense - gen_dense) << std::endl;

            // for (auto &l : no_htool.get_leaves()) {
            //     Matrix<double> *temp;
            //     if (l->is_dense()) {
            //         temp =
            //     }
            // }

            // no_htool.save_plot("hmatrix_no_build");
            // Matrix<double> no_htool_dense(size, size);
            // copy_to_dense(no_htool, no_htool_dense.data());
            // Matrix<double> reference(size, size);
            // gen_csr.copy_submatrix(size, size, 0, 0, reference.data());
            // std::cout << "erreur build sans htool  " << normFrob(reference - no_htool_dense) << " et comrpession : " << no_htool.get_compression() << std::endl;
            std::cout
                << "compression build no htool " << no_htool.get_compression() << std::endl;

            // ///////////////////////////////////
            // // FACTORISATION LU
            std::cout << "Factoriation LU" << std::endl;
            auto x_random = generate_random_vector(size);
            Matrix<double> X_random(size, 1);
            X_random.assign(size, 1, x_random.data(), false);
            std::vector<double> ax_directional(size);
            gen_csr.mat_vect(x_random, ax_directional);
            Matrix<double> Ax_directional(size, 1);
            for (int k = 0; k < size; ++k) {
                Ax_directional(k, 0) = ax_directional[k];
            }

            // HMatrix<double> Lh(*root_directional, *root_directional);
            // Lh.set_admissibility_condition(directional_admissibility);
            // Lh.set_epsilon(epsilon);
            // Lh.set_eta(eta);
            // Lh.set_low_rank_generator(hmatrix_directional.get_low_rank_generator());

            // HMatrix<double, double> Uh(*root_directional, *root_directional);
            // Uh.set_admissibility_condition(directional_admissibility);
            // Uh.set_epsilon(epsilon);
            // Uh.set_eta(eta);
            // Uh.set_low_rank_generator(hmatrix_directional.get_low_rank_generator());

            std::cout << "lu start " << std::endl;

            start = std::chrono::high_resolution_clock::now();
            // HLU_fast(hmatrix_directional, *root_directional, &Lh, &Uh);

            // lu_factorization(hmatrix_directional);
            // lu_factorization(no_htool);
            lu_factorization_csr(no_htool);

            stop = std::chrono::high_resolution_clock::now();
            // no_htool.save_plot("HLU_eps" + std::to_string(eps));
            auto duration_lu = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
            std::cout << "time lu : " << duration_lu.count() << std::endl;
            start = std::chrono::high_resolution_clock::now();

            // hmatrix_directional.save_plot("hlu_directional");
            // lu_solve('N', hmatrix_directional, Ax_directional);
            lu_solve('N', no_htool, Ax_directional);

            stop = std::chrono::high_resolution_clock::now();
            std::cout << "lu ok " << std::endl;
            auto duration_solve = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
            std::cout << "compression du LU directional : " << no_htool.get_compression() << std::endl;
            std::cout << "-------------------------->erreur solve lu directional : " << normFrob(Ax_directional - X_random) / normFrob(X_random) << std::endl;
            std::cout << "-------------------------->time lu et solve : " << duration_lu.count() << "," << duration_solve.count() << std::endl;
            ////////////////////////////////////
        }
        std::cout
            << "fini" << std::endl;
        // }
    }
    return 0;
}