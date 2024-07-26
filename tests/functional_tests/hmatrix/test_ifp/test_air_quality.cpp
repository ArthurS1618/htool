
#include <chrono>
#include <ctime>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/interface.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/matrix.hpp>

#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>

#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <htool/wrappers/wrapper_lapack.hpp>
#include <iostream>
#include <string>

#include <random>
using namespace htool;

////////////////////////
/////// ROUTINES
///// fonction pour appliquer la perm inverse de la numerotation de htool a une matrice (j'arrive pas a me servir de l'option "NOPERM")
template <class CoefficientPrecision, class CoordinatePrecision>
Matrix<CoefficientPrecision> get_unperm_mat(const Matrix<CoefficientPrecision> mat, const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source) {
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
class CSR_generator : public VirtualGenerator<CoefficientPrecision> {
  private:
    std::vector<double> data;
    std::vector<double> row_csr;
    std::vector<double> col_csr;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    CSR_generator(std::vector<double> &data0, std::vector<double> &row_csr0, std::vector<double> &col_csr0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : data(data0), row_csr(row_csr0), col_csr(col_csr0), target(target0), source(source0) {}

    void copy_submatrix(int M, int N, int row_offset, int col_offset, double *ptr) const override {
        const auto &target_permutation = target.get_permutation();
        const auto &source_permutation = source.get_permutation();
        for (int i = 0; i < M; ++i) {
            // int r1 = (int)row_csr[target_permutation[i + row_offset]];
            // int r2 = (int)row_csr[target_permutation[i + row_offset] + 1];
            // std::vector<double> row_iperm(r2 - r1);
            // std::copy(col_csr.begin() + r1, col_csr.begin() + r2, row_iperm.begin());
            // for (int l = r1; l < r2; ++l) {
            //     int j = (int)col_csr[l];
            //     if ((j >= col_offset) && (j < col_offset + N)) {
            //         int j_perm = source_permutation[j];
            //         // auto it    = std::find_if(row_iperm.begin(), row_iperm.end(), [j_perm](int s) {
            //         //     return (s == j_perm);
            //         // });
            //         // int lperm  = std::distance(row_iperm.begin(), it);
            //         int lperm = 0;
            //         bool flag = true;
            //         while (lperm < row_iperm.size() && flag) {
            //             if (col_csr[r1 + lperm] == j_perm) {
            //                 flag = false;
            //             } else {
            //                 lperm += 1;
            //             }
            //         }
            //         if (!flag) {
            //             // std::cout << "!" << std::endl;
            //             ptr[i + M * (j - col_offset)] = data[lperm + r1];
            //         }
            //     }
            // }
            for (int j = 0; j < N; ++j) {
                int j_perm = source_permutation[j + col_offset];
                int r1     = (int)row_csr[target_permutation[i + row_offset]];
                int r2     = (int)row_csr[target_permutation[i + row_offset] + 1];
                std::vector<double> row_iperm(r2 - r1);
                std::copy(col_csr.begin() + r1, col_csr.begin() + r2, row_iperm.begin());
                auto it   = std::find_if(row_iperm.begin(), row_iperm.end(), [j_perm](int s) {
                    return (s == j_perm);
                });
                int lperm = std::distance(row_iperm.begin(), it);
                if (lperm < row_iperm.size()) {
                    ptr[i + M * j] = data[r1 + lperm];
                } else {
                    ptr[i + M * j] = 0.0;
                }
            }

            // std::vector<double> index(row_csr[target_permutation[i + row_offset] + 1] - row_csr[target_permutation[i + row_offset]]);
            // for (int k = 0; k < index.size(); ++k) {
            //     index[k] = source_permutation[col_csr[k + row_csr[target_permutation[i + row_offset]]]]
            // }
            // std::vector<double> index_restr(N);
            // for ()
            //     std::copy(col_csr.begin() + row_csr[target_permutation[i + row_offset]], col_csr.begin() + row_csr[target_permutation[i + row_offset] + 1], index.begin());
            // std::vector<double> index_restr;
            // for (int k = 0; k < index.size(); ++k) {
            //     if ((index[k] >= col_offset) && (index[k] < col_offset + N)) {
            //         index_restr.push_back(index[k]);
            //     }
            // }
            // for (int j = 0; j < index_restr.size(); ++j) {
            //     ptr[i + M * j] = data[source_permutation[index_restr[j]]];
            // }
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

void save_vector(const std::vector<double> &v, const std::string &filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Erreur: Impossible d'ouvrir le fichier " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < v.size(); ++i) {
        file << v[i];
        if (i < v.size() - 1) {
            file << ","; // Ajouter une virgule entre les éléments, sauf après le dernier élément
        }
    }

    file.close();
}

Matrix<double> extract(const int &nr, const int &nc, const int &ofr, const int &ofc, Matrix<double> M) {
    Matrix<double> res(nr, nc);
    for (int k = 0; k < nr; ++k) {
        for (int l = 0; l < nc; ++l) {
            res(k, l) = M(k + ofr, l + ofc);
        }
    }
    return res;
}

/// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% /////
using namespace htool;

int main() {
    double epsilon = 1e-6;
    double eta     = 5;
    int size       = 10000;
    int sizex      = 100;
    int sizey      = 100;
    std::cout << "epsilon et eta : " << epsilon << ',' << eta << std::endl;
    std::vector<double> compr_norm;
    std::vector<double> err_norm;
    std::vector<double> compr_dir;
    std::vector<double> err_dir;
    // for (int eps = 0; eps < 6; ++eps) {
    std::cout << "_______________________________________________________" << std::endl;
    // std::cout << "epse = " << eps << std::endl;
    // epsilon = epsilon * std::pow(10.0, (eps + 1) * .0);

    // std::cout << "HLU factortisation , size =" << size << std::endl;
    ///////////////////////
    ////// GENERATION MAILLAGE  :  size points en 3 dimensions

    std::vector<double> p(3 * size);
    for (int j = 0; j < sizex; j++) {
        for (int k = 0; k < sizey; k++) {
            p[3 * (j + k * sizex) + 0] = j * 1200 / sizex;
            p[3 * (j + k * sizex) + 1] = k * 1200 / sizey;
            p[3 * (j + k * sizex) + 2] = 0.0;
        }
    }

    // Creation clusters
    // normal
    ClusterTreeBuilder<double> recursive_build_strategy;
    recursive_build_strategy.set_minclustersize(20);
    std::shared_ptr<Cluster<double>> root_cluster = std::make_shared<Cluster<double>>(recursive_build_strategy.create_cluster_tree(p.size() / 3, 3, p.data(), 2, 2));
    // directional
    std::vector<double> bb(3);
    bb[0] = 1;
    bb[1] = 1;
    std::vector<double> bperp(3);
    bperp[0] = 1;
    bperp[1] = -1;
    ClusterTreeBuilder<double> directional_build;
    std::shared_ptr<ConstantDirection<double>> strategy_directional = std::make_shared<ConstantDirection<double>>(bperp);
    directional_build.set_direction_computation_strategy(strategy_directional);
    directional_build.set_minclustersize(20);
    std::shared_ptr<Cluster<double>> root_directional = std::make_shared<Cluster<double>>(directional_build.create_cluster_tree(p.size() / 3, 3, p.data(), 2, 2));

    //////////////////////

    //////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    // Recuperation du CSR
    int eps = 5;
    std::cout << "eps = " << eps << std::endl;
    auto csr_file = get_csr("/work/sauniear/Documents/Code_leo/VersionLight2/CSR_file_epsilon_-" + std::to_string(eps) + "_adim_nx100_b11.csv");
    auto val      = csr_file[0];
    auto row      = csr_file[1];
    auto col      = csr_file[2];
    Matrix<double> vrai_mat(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = row[i]; j < row[i + 1]; ++j) {
            vrai_mat(i, col[j]) = val[j];
        }
    }
    ///////////////////
    ///// Inversion
    auto temp = vrai_mat;
    // std::cout << "vrai mat : " << normFrob(vrai_mat) << "---------->" << vrai_mat.nb_rows() << ',' << vrai_mat.nb_cols() << std::endl;
    std::vector<int> ipiv(size, 0.0);
    int info = -1;
    Lapack<double>::getrf(&size, &size, temp.data(), &size, ipiv.data(), &info);
    Matrix<double> L(size, size);
    Matrix<double> U(size, size);
    for (int k = 0; k < size; ++k) {
        U(k, k) = temp(k, k);
        L(k, k) = 1.0;
        for (int l = 0; l < k; ++l) {
            L(k, l) = temp(k, l);
            U(l, k) = temp(l, k);
        }
    }
    auto LU = temp;
    std::cout << "norm lu " << normFrob(LU) << std::endl;
    std::vector<int> ipivo(size);
    int lwork = size * size;
    std::vector<double> work(lwork);
    // Effectue l'inversion
    Lapack<double>::getri(&size, temp.data(), &size, ipiv.data(), work.data(), &lwork, &info);
    std::cout << "inversion ok  " << normFrob(temp) << "---------->" << temp.nb_rows() << ',' << temp.nb_cols() << std::endl;
    // Matrix<double> id(size, size);
    // for (int k = 0; k < size; ++k) {
    //     id(k, k) = 1.0;
    // }
    // Matrix<double> erreur_inv = vrai_mat * temp - id;
    // std::cout << normFrob(erreur_inv) / std::pow(size * 1.0, 0.5) << std::endl;
    // auto ref = temp;
    // Matricegenerator<double, double> generator_directional(vrai_mat, *root_directional, *root_directional);
    // Matricegenerator<double, double> generator_normal(vrai_mat, *root_cluster, *root_cluster);

    Matricegenerator<double, double> generator_inv_directional(temp, *root_directional, *root_directional);
    Matricegenerator<double, double> generator_inv_normal(temp, *root_cluster, *root_cluster);
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);

    HMatrixTreeBuilder<double, double> hmatrix_directional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);
    std::shared_ptr<Bande<double>> directional_admissibility = std::make_shared<Bande<double>>(bb, root_directional->get_permutation());
    hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);

    // Assemblage hmatrices
    // Matrix<double> gen_directional_dense(size, size);
    // Matrix<double> gen_normal_dense(size, size);
    // generator_directional.copy_submatrix(size, size, 0, 0, gen_directional_dense.data());
    // generator_normal.copy_submatrix(size, size, 0, 0, gen_normal_dense.data());
    // auto root_hmatrix = hmatrix_tree_builder.build(generator_normal);
    // std::cout << "normal build ok " << std::endl;
    // auto directional_hmatrix = hmatrix_directional_builder.build(generator_directional);
    // std::cout << "directional build ok " << std::endl;
    // Matrix<double> normal_dense(size, size);
    // Matrix<double> directional_dense(size, size);
    // copy_to_dense(root_hmatrix, normal_dense.data());
    // copy_to_dense(directional_hmatrix, directional_dense.data());
    // auto nr = normFrob(gen_normal_dense);
    // std::cout << "erreur normal build : " << normFrob(gen_normal_dense - normal_dense) / nr << std::endl;
    // std::cout << "compression normal build :" << root_hmatrix.get_compression() << std::endl;
    // std::cout << "erreur directional build : " << normFrob(gen_directional_dense - directional_dense) / nr << std::endl;
    // std::cout << "compression directional build :" << directional_hmatrix.get_compression() << std::endl;
    // std::cout << "norme matrice " << nr << std::endl;
    /// inverse
    Matrix<double> gen_directional_dense(size, size);
    Matrix<double> gen_normal_dense(size, size);
    generator_inv_directional.copy_submatrix(size, size, 0, 0, gen_directional_dense.data());
    generator_inv_normal.copy_submatrix(size, size, 0, 0, gen_normal_dense.data());
    auto root_hmatrix = hmatrix_tree_builder.build(generator_inv_normal);
    std::cout << "normal build ok " << std::endl;
    auto directional_hmatrix = hmatrix_directional_builder.build(generator_inv_directional);
    std::cout << "directional build ok " << std::endl;
    Matrix<double> normal_dense(size, size);
    Matrix<double> directional_dense(size, size);
    copy_to_dense(root_hmatrix, normal_dense.data());
    copy_to_dense(directional_hmatrix, directional_dense.data());
    auto nr = normFrob(gen_normal_dense);
    std::cout << "erreur normal build : " << normFrob(gen_normal_dense - normal_dense) / nr << std::endl;
    std::cout << "compression normal build :" << root_hmatrix.get_compression() << std::endl;
    std::cout << "erreur directional build : " << normFrob(gen_directional_dense - directional_dense) / nr << std::endl;
    std::cout << "compression directional build :" << directional_hmatrix.get_compression() << std::endl;
    std::cout << "nr :" << nr << std::endl;
    std::cout << "leaves normal : " << root_hmatrix.get_leaves().size() << " et directional : " << directional_hmatrix.get_leaves().size() << std::endl;

    ///// HLU
    info = -1;
    temp = gen_directional_dense;
    Matrix<double> Xdense(size, 1);
    generate_random_matrix(Xdense);
    auto error = normFrob(directional_dense * Xdense - gen_directional_dense * Xdense) / normFrob(gen_directional_dense * Xdense);

    std::cout << "> Errors on hmatrix lu solve: " << error << std::endl;
    // Lapack<double>::getrf(&size, &size, temp.data(), &size, ipiv.data(), &info);
    // Matrix<double> Xdense(size, 1);
    // generate_random_matrix(Xdense);

    // Matrix<double> Ydense(size, 1);
    // generate_random_matrix(Ydense);

    // auto Bdense = gen_directional_dense * Xdense;
    // Matrix<double> Bdense(Xdense);
    // Matrix<double> Bdense2(Xdense);

    // add_matrix_matrix_product('N', 'N', 1.0, gen_directional_dense, Xdense, 0.0, Bdense);
    // add_matrix_matrix_product('N', 'N', 1.0, directional_dense, Xdense, 0.0, Bdense2);
    // std::cout << " y-y0  " << normFrob(Bdense2 - Bdense) / normFrob(Bdense) << std::endl;

    // LU factorization
    // Matrix<double> matrix_test = Bdense;
    // Matrix<double> matrix_test  = Ydense;
    // Matrix<double> matrix_test2 = Ydense;

    // lu_factorization(directional_hmatrix);
    // std::cout << "LU ok " << std::endl;
    // Matrix<double> ludense(size, size);
    // std::cout << "compression " << directional_hmatrix.get_compression() << std::endl;
    // copy_to_dense(directional_hmatrix, ludense.data());
    // std::cout << "erreur LU :  " << normFrob(ludense - temp) / normFrob(temp) << std::endl;

    // lu_solve('N', directional_hmatrix, matrix_test);

    // // auto error = normFrob(Xdense - matrix_test) / normFrob(Xdense);
    // // auto error = normFrob(directional_dense * matrix_test - gen_directional_dense * matrix_test) / normFrob(gen_directional_dense * matrix_test);

    // std::cout << "> Errors on hmatrix lu solve: " << error << std::endl;
    // directional_hmatrix.save_plot("inv_directional_eta5_eps6_diff2");
    // root_hmatrix.save_plot("inv_normal_eta5_eps6_diff2");

    // LU
    // HMatrixTreeBuilder<double, double> lunormal_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);

    // HMatrixTreeBuilder<double, double> ludirectional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);

    // ludirectional_builder.set_admissibility_condition(directional_admissibility);
    // auto LUperm_directional = get_unperm_mat(LU, *root_directional, *root_directional);
    // auto LUperm_normal      = get_unperm_mat(LU, *root_cluster, *root_cluster);
    // Matricegenerator<double, double> gen_LU_directional(LUperm_directional, *root_directional, *root_directional);
    // auto LUh_directional = ludirectional_builder.build(gen_LU_directional);
    // std::cout << "directional ok" << std::endl;
    // Matricegenerator<double, double> gen_LU_normal(LUperm_normal, *root_cluster, *root_cluster);
    // auto LUh_normal = lunormal_builder.build(gen_LU_normal);
    // std::cout << "normal ok" << std::endl;
    // Matrix<double> ludir_dense(size, size);
    // copy_to_dense(LUh_directional, ludir_dense.data());
    // Matrix<double> lunorm_dense(size, size);
    // copy_to_dense(LUh_normal, lunorm_dense.data());
    // auto nlu      = normFrob(LU);
    // double ern    = normFrob(lunorm_dense - LU) / nlu;
    // double comprn = LUh_normal.get_compression();
    // double erd    = normFrob(ludir_dense - LU) / nlu;
    // double comprd = LUh_directional.get_compression();
    // std::cout << "norme L/U " << nlu << std::endl;
    // std::cout << "compression directionl L/ U ------------------------->" << comprd << std::endl;
    // std::cout << "error directional L/ U      ------------------------->" << erd << std::endl;
    // std::cout << "/////////////////////////////////" << std::endl;
    // std::cout << "compression normal L/ U     ------------------------->" << comprn << std::endl;
    // std::cout << "error normal L/ U           ------------------------->" << ern << std::endl;
    // std::cout << "_______________________________________________________" << std::endl;
    // LUh_directional.save_plot("LU_directional_eta5_eps6_diff2");
    // LUh_normal.save_plot("LU_normal_eta5_eps6_diff2");

    // comptage des feuilles LR
    // double err = 0.0;
    // int cpt    = 0;
    // int ss     = 0;
    // int rkmean = 0.0;
    // for (auto &l : LUh_directional.get_leaves()) {
    //     if (l->is_low_rank()) {
    //         ss += l->get_target_cluster().get_size();
    //         auto aprox = l->get_low_rank_data()->get_U() * l->get_low_rank_data()->get_V();
    //         auto ref   = extract(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset(), LU);
    //         // std::cout << " erreur sur la feuille (" << l->get_target_cluster().get_size() << ',' << l->get_target_cluster().get_offset() << "),(" << l->get_source_cluster().get_size() << ',' << l->get_source_cluster().get_offset() << ")       ->" << normFrob(ref - aprox) / nlu << std::endl;
    //         // std::cout << "feuille de rang " << l->get_low_rank_data()->get_U().nb_cols() << std::endl;
    //         err += normFrob(ref - aprox);
    //         // err += normFrob(ref - aprox) / nr;

    //         cpt += 1;
    //         rkmean += l->get_low_rank_data()->get_U().nb_cols();
    //     }
    // }
    // auto moy = err / (1.0 * cpt);
    // rkmean   = rkmean / (1.0 * cpt);
    // std::cout << "directional" << std::endl;
    // std::cout << "taille moyenne " << ss / (1.0 * cpt) << ", rang moyen : " << rkmean << std::endl;
    // std::cout << " size lr : " << cpt << std::endl;
    // std::cout << "erreur moyenne " << moy / nlu << std::endl;
    // std::cout << "erreur feuille lr : " << err / nlu << std::endl;
    // err    = 0.0;
    // cpt    = 0;
    // ss     = 0;
    // rkmean = 0;

    // for (auto &l : LUh_normal.get_leaves()) {
    //     if (l->is_low_rank()) {
    //         ss += l->get_target_cluster().get_size();
    //         auto aprox = l->get_low_rank_data()->get_U() * l->get_low_rank_data()->get_V();
    //         auto ref   = extract(l->get_target_cluster().get_size(), l->get_source_cluster().get_size(), l->get_target_cluster().get_offset(), l->get_source_cluster().get_offset(), LU);
    //         // std::cout << " erreur sur la feuille (" << l->get_target_cluster().get_size() << ',' << l->get_target_cluster().get_offset() << "),(" << l->get_source_cluster().get_size() << ',' << l->get_source_cluster().get_offset() << ")       ->" << normFrob(ref - aprox) / nlu << std::endl;
    //         // std::cout << "feuille de rang " << l->get_low_rank_data()->get_U().nb_cols() << std::endl;
    //         err += normFrob(ref - aprox);
    //         // err += normFrob(ref - aprox) / nr;

    //         cpt += 1;
    //         rkmean += l->get_low_rank_data()->get_U().nb_cols();
    //     }
    // }
    // std::cout << "____________________________" << std::endl;
    // std::cout << "normal " << std::endl;
    // auto moy2 = err / (1.0 * cpt);
    // rkmean    = rkmean / (1.0 * cpt);
    // std::cout << "taille moyenne " << ss / (1.0 * cpt) << ", rang moyen : " << rkmean << std::endl;
    // std::cout << " size lr : " << cpt << std::endl;
    // std::cout << "erreur moyenne " << moy2 / nlu << std::endl;
    // std::cout << "erreur feuille lr : " << err / nlu << std::endl;

    //     / Fin test approximabilité hmat

    // / test SVD blocs admissibles
    // erreur sur la feuille (39,1718),(39,1601)   de normal     ->0.962543

    /// tests SVD
    // Matricegenerator<double, double> generator_lu_directional(LU, *root_directional, *root_directional);
    // Matricegenerator<double, double> generator_lu_normal(LU, *root_cluster, *root_cluster);
    // Matrix<double> LUnormal(size, size);
    // Matrix<double> LUdirectional(size, size);
    // generator_lu_directional.copy_submatrix(size, size, 0, 0, LUdirectional.data());
    // generator_lu_normal.copy_submatrix(size, size, 0, 0, LUnormal.data());
    // int nsize = size / 16;
    // int off   = size / 2;
    // Matrix<double> bloc_normal(nsize, nsize);

    // for (int kk = 0; kk < nsize; ++kk) {
    //     for (int ll = 0; ll < nsize; ++ll) {
    //         // bloc_normal(kk, ll) = gen_normal_dense(kk, ll + off);
    //         bloc_normal(kk, ll) = LUnormal(kk, ll + off);
    //     }
    // }
    // std::cout << "       normal : " << normFrob(bloc_normal) << std::endl;

    // Matrix<double> bloc_directional(size / 2, size / 2);

    // for (int kk = 0; kk < size / 2; ++kk) {
    //     for (int ll = 0; ll < size / 2; ++ll) {
    //         // bloc_directional(kk, ll) = gen_directional_dense(kk, ll + off);
    //         bloc_directional(kk, ll) = LUdirectional(kk, ll + off);
    //     }
    // }

    // // std::cout << "       directional : " << normFrob(bloc_directional) << std::endl;
    // // //////////////////////////::
    // auto bloc_normal = extract(313, 312, 7187, 7500, LU);
    // Matrix<double> Unormal(bloc_normal.nb_rows(), std::min(bloc_normal.nb_rows(), bloc_normal.nb_cols()));
    // Matrix<double> Vnormal(std::min(bloc_normal.nb_rows(), bloc_normal.nb_cols()), bloc_normal.nb_cols());
    // std::cout << "svd bloc normal " << std::endl;
    // int Nnormal = 3 * std::max(bloc_normal.nb_rows(), bloc_normal.nb_cols());
    // std::vector<double> Snormal(std::min(bloc_normal.nb_rows(), bloc_normal.nb_cols()), 0.0);
    // int infonormal;
    // int lworknormal = 10 * Nnormal;
    // std::vector<double> rworknormal(lwork);
    // std::vector<double> worknormal(lwork);
    // int mm = bloc_normal.nb_rows();
    // int nn = bloc_normal.nb_cols();
    // int rr = std::min(bloc_normal.nb_rows(), bloc_normal.nb_cols());
    // Lapack<double>::gesvd("N", "N", &mm, &nn, bloc_normal.data(), &mm, Snormal.data(), Unormal.data(), &mm, Vnormal.data(), &rr, worknormal.data(), &lworknormal, rworknormal.data(), &infonormal);
    // std::cout << "SVD bloc normal ok " << std::endl;
    // save_vector(Snormal, "SVDU_pb_ACA6_eta5_eps5.csv");
    // std::cout << "save SVD ok " << std::endl;

    // Matrix<double> Udirectional(bloc_directional.nb_rows(), std::min(bloc_directional.nb_rows(), bloc_directional.nb_cols()));
    // Matrix<double> Vdirectional(std::min(bloc_directional.nb_rows(), bloc_directional.nb_cols()), bloc_directional.nb_cols());
    // std::cout << "SVD bloc directionel " << std::endl;
    // int Ndirectional = 3 * std::max(bloc_directional.nb_rows(), bloc_directional.nb_cols());
    // std::vector<double> Sdirectional(std::min(bloc_directional.nb_rows(), bloc_directional.nb_cols()), 0.0);
    // int infodirectional;
    // int lworkdirectional = 10 * Ndirectional;
    // std::vector<double> rworkdirectional(lworkdirectional);
    // std::vector<double> workdirectional(lworkdirectional);
    // mm = bloc_directional.nb_rows();
    // nn = bloc_directional.nb_cols();
    // rr = std::min(bloc_directional.nb_rows(), bloc_directional.nb_cols());
    // Lapack<double>::gesvd("N", "N", &mm, &nn, bloc_directional.data(), &mm, Sdirectional.data(), Udirectional.data(), &mm, Vdirectional.data(), &rr, workdirectional.data(), &lworkdirectional, rworkdirectional.data(), &infodirectional);
    // std::cout << "SVD bloc directional ok" << std::endl;
    // save_vector(Sdirectional, "SVDU_directional_ACA3_eta5_eps5.csv");
    // std::cout << "save directional ok " << std::endl;

    ////// FIN test SVD

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    ///////////////////
    // HMATRIX
    // std::cout << "HMATRIX assemblage" << std::endl;
    // HMatrixTreeBuilder<double, double> hmatrix_tree_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);

    // HMatrixTreeBuilder<double, double> hmatrix_directional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);

    // HMatrixTreeBuilder<double, double> ldirectional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);
    // HMatrixTreeBuilder<double, double> udirectional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);
    // HMatrixTreeBuilder<double, double> lunormal_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);

    // HMatrixTreeBuilder<double, double> ludirectional_builder(*root_directional, *root_directional, epsilon, eta, 'N', 'N', -1, -1, -1);

    // std::shared_ptr<Bande<double>> directional_admissibility = std::make_shared<Bande<double>>(bb, root_directional->get_permutation());
    // ldirectional_builder.set_admissibility_condition(directional_admissibility);
    // udirectional_builder.set_admissibility_condition(directional_admissibility);
    // ludirectional_builder.set_admissibility_condition(directional_admissibility);

    // hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);

    // build
    // auto root_hmatrix = hmatrix_tree_builder.build(generator_inv_normal);
    // std::cout << "normal build ok " << std::endl;
    // auto directional_hmatrix = hmatrix_directional_builder.build(generator_inv_directional);
    // std::cout << "directional build ok " << std::endl;
    // Matrix<double> normal_dense(size, size);
    // Matrix<double> directional_dense(size, size);
    // copy_to_dense(root_hmatrix, normal_dense.data());
    // copy_to_dense(directional_hmatrix, directional_dense.data());
    // auto nr = normFrob(gen_normal_dense);
    // std::cout << "erreur normal build : " << normFrob(gen_normal_dense - normal_dense) / nr << std::endl;
    // std::cout << "compression normal build :" << root_hmatrix.get_compression() << std::endl;
    // std::cout << "erreur directional build : " << normFrob(gen_directional_dense - directional_dense) / nr << std::endl;
    // std::cout << "compression directional build :" << directional_hmatrix.get_compression() << std::endl;
    // std::cout << "nr :" << nr << std::endl;
    // std::cout << "leaves normal : " << root_hmatrix.get_leaves().size() << " et directional : " << directional_hmatrix.get_leaves().size() << std::endl;

    //////////////////////////////////////////////////////////
    // auto Lperm_normal = get_unperm_mat(L, *root_cluster, *root_cluster);

    // auto Uperm_normal = get_unperm_mat(U, *root_cluster, *root_cluster);

    // // std::cout << "matrice ok " << std::endl;
    // Matricegenerator<double, double> gen_L_normal(Lperm_normal, *root_cluster, *root_cluster);
    // Matricegenerator<double, double> gen_U_normal(Uperm_normal, *root_cluster, *root_cluster);

    // HMatrixTreeBuilder<double, double> lnormal_tree_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);
    // HMatrixTreeBuilder<double, double> unormal_tree_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);
    // auto Lh_normal = lnormal_tree_builder.build(gen_L_normal);
    // auto Uh_normal = unormal_tree_builder.build(gen_U_normal);
    // Matrix<double> lnormal_dense(size, size);
    // copy_to_dense(Lh_normal, lnormal_dense.data());
    // Matrix<double> unormal_dense(size, size);
    // copy_to_dense(Uh_normal, unormal_dense.data());
    // auto nl = normFrob(L);
    // auto nu = normFrob(U);
    // std::cout << "norme L and U " << nl << ',' << nu << std::endl;
    // std::cout << "compression normal L and U " << Lh_normal.get_compression() << ',' << Uh_normal.get_compression() << std::endl;
    // std::cout << "error normal L and U    " << normFrob(lnormal_dense - L) / nl << ',' << normFrob(unormal_dense - U) / nu << std::endl;

    // //////////
    // auto Lperm_directional = get_unperm_mat(L, *root_directional, *root_directional);
    // auto Uperm_directional = get_unperm_mat(U, *root_directional, *root_directional);
    // auto LUperm_directional = get_unperm_mat(LU, *root_directional, *root_directional);
    // auto LUperm_normal      = get_unperm_mat(LU, *root_cluster, *root_cluster);

    // // Matricegenerator<double, double> gen_L_directional(Lperm_directional, *root_directional, *root_directional);
    // // Matricegenerator<double, double> gen_U_directional(Uperm_directional, *root_directional, *root_directional);
    // Matricegenerator<double, double> gen_LU_directional(LUperm_directional, *root_directional, *root_directional);
    // auto LUh_directional = ludirectional_builder.build(gen_LU_directional);
    // Matricegenerator<double, double> gen_LU_normal(LUperm_normal, *root_cluster, *root_cluster);
    // auto LUh_normal = lunormal_builder.build(gen_LU_normal);

    // // auto Lh_directional = ldirectional_builder.build(gen_L_directional);
    // // auto Uh_directional = udirectional_builder.build(gen_U_directional);

    // // Matrix<double> ldir_dense(size, size);
    // // copy_to_dense(Lh_directional, ldir_dense.data());
    // // Matrix<double> udir_dense(size, size);
    // // copy_to_dense(Uh_directional, udir_dense.data());
    // Matrix<double> ludir_dense(size, size);
    // copy_to_dense(LUh_directional, ludir_dense.data());
    // Matrix<double> lunorm_dense(size, size);
    // copy_to_dense(LUh_normal, lunorm_dense.data());

    // // Matrix<double> lnormal_dense(size, size);
    // // copy_to_dense(Lh_normal, lnormal_dense.data());
    // // Matrix<double> unormal_dense(size, size);
    // // copy_to_dense(Uh_normal, unormal_dense.data());
    // auto nlu      = normFrob(LU);
    // double ern    = normFrob(lunorm_dense - LU) / nlu;
    // double comprn = LUh_normal.get_compression();
    // double erd    = normFrob(ludir_dense - LU) / nlu;
    // double comprd = LUh_directional.get_compression();
    // std::cout << "norme L/U " << nlu << std::endl;
    // std::cout << "compression directionl L/ U ------------------------->" << comprd << std::endl;
    // std::cout << "error directional L/ U      ------------------------->" << erd << std::endl;
    // std::cout << "/////////////////////////////////" << std::endl;
    // std::cout << "compression normal L/ U     ------------------------->" << comprn << std::endl;
    // std::cout << "error normal L/ U           ------------------------->" << ern << std::endl;
    // std::cout << "_______________________________________________________" << std::endl;
    // compr_norm.push_back(comprn);
    // compr_dir.push_back(comprd);
    // err_norm.push_back(ern);
    // err_dir.push_back(erd);
    // }
    // std::cout << "directional" << std::endl;
    // std::cout << "compression  :";
    // for (int k = 0; k < compr_dir.size(); ++k) {
    //     std::cout << compr_dir[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "error  :";
    // for (int k = 0; k < err_dir.size(); ++k) {
    //     std::cout << err_dir[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "normal" << std::endl;
    // std::cout << "compression  :";
    // for (int k = 0; k < compr_norm.size(); ++k) {
    //     std::cout << compr_norm[k] << ',';
    // }
    // std::cout << std::endl;
    // std::cout << "error  :";

    // for (int k = 0; k < err_norm.size(); ++k) {
    //     std::cout << err_norm[k] << ',';
    // }
    // std::cout << std::endl;

    // std::cout << "epsilon " << LUh_directional.get_epsilon() << std::endl;
    // Matrix<double> xrand(size, 1);
    // generate_random_matrix(xrand);
    // auto Mperm_directional      = get_unperm_mat(vrai_mat, *root_directional, *root_directional);
    // Matrix<double> ydirectional = Mperm_directional * xrand;

    // lu_solve('N', LUh_directional, ydirectional);
    // std::cout << " erreur Lu solve : " << normFrob(xrand - ydirectional) / normFrob(xrand) << std::endl;
    // auto nl = normFrob(L);
    // auto nu = normFrob(U);
    // std::cout << "norme L and U " << nl << ',' << nu << std::endl;
    // std::cout << "compression directional L and U " << Lh_directional.get_compression() << ',' << Uh_directional.get_compression() << std::endl;
    // std::cout << "error directional L and U    " << normFrob(ldir_dense - L) / nl << ',' << normFrob(udir_dense - U) / nu << std::endl;
    // auto xrand             = generate_random_vector(size);
    // auto Mperm_directional = get_unperm_mat(vrai_mat, *root_directional, *root_directional);
    // auto ydirectional      = Mperm_directional * xrand;

    // auto xdirectional = Uh_directional.solve_LU_triangular(Lh_directional, Uh_directional, ydirectional);
    // std::cout << " erreur Lu solve : " << norm2(xrand - xdirectional) / norm2(xrand) << std::endl;

    // Matricegenerator<double, double> gen_L_directional(Lperm_normal, *root_cluster, *root_cluster);
    // Matricegenerator<double, double> gen_U_directional(Lperm_normal, *root_cluster, *root_cluster);
    // std::cout << "generator ok" << std::endl;
    // auto Lh_normal = lnormal_tree_builder.build(gen_L_normal);
    // std::cout << "build L ok" << std::endl;
    // auto Uh_normal = unormal_tree_builder.build(gen_U_normal);
    // std::cout << "buid U ok " << std::endl;
    // Matrix<double> lnormal_dense(size, size);
    // copy_to_dense(Lh_normal, lnormal_dense.data());
    // std::cout << "ldesne ok " << std::endl;
    // Matrix<double> unormal_dense(size, size);
    // copy_to_dense(Uh_normal, unormal_dense.data());

    // auto Lh_directional = ldirectional_builder.build(gen_L_directional);
    // auto Uh_directional = udirectional_builder.build(gen_U_directional);
    // Matrix<double> ldirectional_dense(size, size);
    // copy_to_dense(Lh_directional, ldirectional_dense.data());
    // Matrix<double> udirectional_dense(size, size);
    // copy_to_dense(Uh_directional, udirectional_dense.data());

    // std::cout << "compression direcitonal L and U " << Lh_directional.get_compression() << ',' << Uh_directional.get_compression() << std::endl;
    // std::cout << "erreur directional L and U      " << normFrob(ldirectional_dense - L) / normFrob(L) << ',' << normFrob(udirectional_dense - U) / normFrob(U) << std::endl;

    // auto xrand             = generate_random_vector(size);
    // auto Mperm_normal      = get_unperm_mat(vrai_mat, *root_cluster, *root_cluster);
    // auto Mperm_directional = get_unperm_mat(vrai_mat, *root_directional, *root_directional);
    // auto ynormal           = Mperm_normal * xrand;
    // auto ydirectional      = Mperm_directional * xrand;

    // auto xnormal      = Lh_normal.solve_LU_triangular(Lh_normal, Uh_normal, ynormal);
    // auto xdirectional = Uh_normal.solve_LU_triangular(Lh_directional, Uh_directional, ydirectional);

    // std::cout << "erreur solve LU normal : " << norm2(xrand - xnormal) / norm2(xrand) << std::endl;
    // std::cout << "erreur solve LU directional " << norm2(xrand - xdirectional) / norm2(xrand) << std::endl;
    // std::cout << "compression LU normal " << Lh_normal.get_compression() << ',' << Uh_normal.get_compression() << std::endl;
    // std::cout << "compression LU directional" << Lh_directional.get_compression() << ',' << Uh_normal.get_compression() << std::endl;

    return 0;
}