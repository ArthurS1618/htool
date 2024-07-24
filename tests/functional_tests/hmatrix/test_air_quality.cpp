#include <chrono>
#include <ctime>
#include <htool/matrix/matrix.hpp>

#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>

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
double get_compression(const HMatrix<CoefficientPrecision, CoordinatePrecision> *H) {
    double compr = 0.0;
    for (auto &leaves : H->get_leaves()) {
        if (leaves->is_low_rank()) {
            compr += leaves->get_low_rank_data()->rank_of() * (leaves->get_low_rank_data()->nb_rows() + leaves->get_low_rank_data()->nb_cols());
        } else {
            compr += (leaves->get_dense_data() * leaves->get_dense_data());
        }
    }
    compr = compr / (H->get_target_cluster().get_size() * H->get_source_cluster().get_size());
    compr = 1 - compr;
    return compr;
}
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
/// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% /////
using namespace htool;

int main() {
    double epsilon = 1e-6;
    double eta     = 100.0;
    int size       = 10000;
    int sizex      = 100;
    int sizey      = 100;
    // std::cout << "HLU factortisation , size =" << size << std::endl;
    ///////////////////////
    ////// GENERATION MAILLAGE  :  size points en 3 dimensions
    std::cout << "maillage " << std::endl;

    std::vector<double> p(3 * size);
    for (int j = 0; j < sizex; j++) {
        for (int k = 0; k < sizey; k++) {
            p[3 * (j + k * sizex) + 0] = j * 1200 / sizex;
            p[3 * (j + k * sizex) + 1] = k * 1200 / sizey;
            p[3 * (j + k * sizex) + 2] = 0.0;
        }
    }
    std::cout << "maillage ok" << std::endl;
    ClusterTreeBuilder<double> recursive_build_strategy;
    // Cluster<double> root_cluster                  = recursive_build_strategy.create_cluster_tree();
    std::shared_ptr<Cluster<double>> root_cluster = std::make_shared<Cluster<double>>(recursive_build_strategy.create_cluster_tree(p.size() / 3, 3, p.data(), 2, 2));
    std::cout << "création cluster normal ok" << std::endl;

    std::vector<double> bb(3);
    bb[0] = 1;
    bb[1] = 0;
    std::vector<double> bperp(3);
    bperp[0] = 0;
    bperp[1] = 1;
    ClusterTreeBuilder<double> directional_build;
    // ConstantDirection<double> constdir0(bperp);
    std::shared_ptr<ConstantDirection<double>> constdir = std::make_shared<ConstantDirection<double>>(bperp);
    directional_build.set_direction_computation_strategy(constdir);
    std::shared_ptr<Cluster<double>> root_directional = std::make_shared<Cluster<double>>(directional_build.create_cluster_tree(p.size() / 3, 3, p.data(), 2, 2));
    // ClusterTreeBuilder<double, htool::ComputeLargestExtent<double>, htool::RegularSplitting<double>> directional_build(p.size() / 3, 3, p.data(), 2, 2);
    // std::shared_ptr<Cluster<double>> root_directional = std::make_shared<Cluster<double>>(directional_build.create_directional_tree(bperp));
    save_cluster_tree(*root_directional, "directional_splitting");
    save_cluster_tree(*root_cluster, "normal_splitting");

    std::cout << "création cluster directional ok" << std::endl;
    //////////////////////

    //////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    auto csr_file = get_csr("/work/sauniear/Documents/Code_leo/VersionLight2/CSR_file_epsilon_1_nx100_b10.csv");
    auto val      = csr_file[0];
    auto row      = csr_file[1];
    auto col      = csr_file[2];
    std::cout << col[0] << ',' << col[1] << ',' << col[2] << ',' << col[3] << std::endl;
    Matrix<double> vrai_mat(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = row[i]; j < row[i + 1]; ++j) {
            vrai_mat(i, col[j]) = val[j];
        }
    }
    std::cout << " Csr extracted" << std::endl;
    ///////////////////
    // Generator: donné en argument de la méthode
    // auto permutation = root_cluster_1->get_permutation();
    // GeneratorTestType generator(3, size, size, p1, p1, root_cluster_1, root_cluster_1);
    CSR_generator<double, double> csr_generator(val, row, col, *root_cluster, *root_cluster);
    Matrix<double> reference(size, size);
    csr_generator.copy_submatrix(size, size, 0, 0, reference.data());

    CSR_generator<double, double> csr_generator_directional(val, row, col, *root_directional, *root_directional);
    Matrix<double> reference_directional(size, size);
    csr_generator_directional.copy_submatrix(size, size, 0, 0, reference_directional.data());
    std::cout << "generator ok" << std::endl;

    auto temp = vrai_mat;
    std::cout << "vrai mat : " << normFrob(vrai_mat) << "---------->" << vrai_mat.nb_rows() << ',' << vrai_mat.nb_cols() << std::endl;
    std::vector<int> ipiv(size, 0.0);
    int info = -1;
    Lapack<double>::getrf(&size, &size, temp.data(), &size, ipiv.data(), &info);
    std::cout << "LU ok " << std::endl;
    int infoo = -1;

    // Alloue le tableau de travail avec la taille requise
    std::vector<int> ipivo(size);
    int lwork = size * size;
    std::vector<double> work(lwork);
    // Effectue l'inversion
    Lapack<double>::getri(&size, temp.data(), &size, ipiv.data(), work.data(), &lwork, &info);
    std::cout << "inversion ok  " << normFrob(temp) << "---------->" << temp.nb_rows() << ',' << temp.nb_cols() << std::endl;
    Matrix<double> id(size, size);
    for (int k = 0; k < size; ++k) {
        id(k, k) = 1.0;
    }
    Matrix<double> erreur_inv = vrai_mat * temp - id;
    std::cout << normFrob(erreur_inv) / std::pow(size * 1.0, 0.5) << std::endl;
    auto ref = temp;

    Matricegenerator<double, double> generator_inv_directional(temp, *root_directional, *root_directional);
    Matricegenerator<double, double> generator_inv_normal(temp, *root_cluster, *root_cluster);
    Matrix<double> gen_directional_dense(size, size);
    Matrix<double> gen_normal_dense(size, size);
    generator_inv_directional.copy_submatrix(size, size, 0, 0, gen_directional_dense.data());
    generator_inv_normal.copy_submatrix(size, size, 0, 0, gen_normal_dense.data());

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    ///////////////////
    // HMATRIX
    // std::cout << "HMATRIX assemblage" << std::endl;
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster, root_cluster, epsilon, eta, 'N', 'N');
    HMatrixTreeBuilder<double, double> hmatrix_directional_builder(root_directional, root_directional, epsilon, eta, 'N', 'N');
    std::shared_ptr<Bande<double>> directional_admissibility = std::make_shared<Bande<double>>(bb, root_directional->get_permutation());
    hmatrix_directional_builder.set_admissibility_condition(directional_admissibility);

    // build
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
    std::cout << "compression normal build :" << get_compression(root_hmatrix) << std::endl;
    std::cout << "erreur directional build : " << normFrob(gen_directional_dense - directional_dense) / nr << std::endl;
    std::cout << "compression directional build :" << get_compression(directional_hmatrix) << std::endl;
    std::cout << "nr :" << nr << std::endl;
    std::cout << "leaves normal : " << root_hmatrix.get_leaves().size() << " et directional : " << directional_hmatrix.get_leaves().size() << std::endl;

    //////////////////////////////////////////////////////////
    // Matrix<double> reference_num_htool(size, size);
    // generator.copy_submatrix(size, size, 0, 0, reference_num_htool.data());

    // //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    // ///////////////////
    // // HMATRIX
    // // std::cout << "HMATRIX assemblage" << std::endl;
    // HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

    // // build
    // auto start_time_asmbl = std::chrono::high_resolution_clock::now();
    // auto root_hmatrix     = hmatrix_tree_builder.build(generator);
    // auto end_time_asmbl   = std::chrono::high_resolution_clock::now();
    // auto duration_asmbl   = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_asmbl - start_time_asmbl).count();
    // Matrix<double> ref(size, size);
    // copy_to_dense(root_hmatrix, ref.data());
    // auto comprr = root_hmatrix.get_compression();

    // HMatrix<double, double> L0(root_cluster_1, root_cluster_1);
    // L0.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    // L0.set_epsilon(epsilon);
    // L0.set_low_rank_generator(root_hmatrix.get_low_rank_generator());

    // HMatrix<double, double> U0(root_cluster_1, root_cluster_1);
    // U0.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    // U0.set_epsilon(epsilon);
    // U0.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
    // U0.copy_zero(root_hmatrix);
    // L0.copy_zero(root_hmatrix);
    // auto start_time_lu0 = std::chrono::high_resolution_clock::now();
    // HLU_noperm(root_hmatrix, *root_cluster_1, L0, U0);
    // auto end_time_lu0 = std::chrono::high_resolution_clock::now();
    // auto duration_lu0 = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_lu0 - start_time_lu0).count();

    // // HLU_noperm(root_hmatrix, *root_cluster_1, L0, U0);
    // auto rand   = generate_random_vector(size);
    // auto y0     = reference_num_htool * rand;
    // auto res0   = L0.solve_LU_triangular(L0, U0, y0);
    // auto err    = norm2(rand - res0) / norm2(rand);
    // double cpt1 = 0;
    // double cpt2 = 0;
    // for (auto &l : L0.get_leaves()) {
    //     if (l->get_target_cluster().get_offset() >= l->get_source_cluster().get_offset()) {
    //         if (l->is_dense()) {
    //             cpt1 = 1.0 * cpt1 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
    //         } else {
    //             // cpt1 = cpt1 + l->get_low_rank_data()->Get_U().nb_cols() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
    //             if (l->get_low_rank_data()->rank_of() > 0) {
    //                 cpt1 = cpt1 + l->get_low_rank_data()->rank_of() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
    //             } else {
    //                 cpt1 = cpt1 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
    //             }
    //         }
    //     }
    // }
    // for (auto &l : U0.get_leaves()) {
    //     if (l->get_target_cluster().get_offset() <= l->get_source_cluster().get_offset()) {
    //         if (l->is_dense()) {
    //             cpt1 = 1.0 * cpt1 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
    //         } else {
    //             // cpt1 = cpt1 + l->get_low_rank_data()->Get_U().nb_cols() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
    //             if (l->get_low_rank_data()->rank_of() > 0) {
    //                 cpt1 = cpt1 + l->get_low_rank_data()->rank_of() * 1.0 * (l->get_target_cluster().get_size() + l->get_source_cluster().get_size());
    //             } else {
    //                 cpt1 = cpt1 + 1.0 * (l->get_target_cluster().get_size() * l->get_source_cluster().get_size());
    //             }
    //         }
    //     }
    // }
    // cpt1 = cpt1 / ((L0.get_target_cluster().get_size() * L0.get_source_cluster().get_size()));
    // cpt2 = 1.000 - cpt1;
    // std::cout << "compression de A " << comprr << std::endl;
    // std::cout << "____________________________________________" << std::endl;
    // std::cout << " Size : -------------------------------> " << size << std::endl;
    // std::cout << " Time : -------------------------------> " << duration_lu0 << std::endl;
    // std::cout << " Compression :-------------------------> " << cpt2 << std::endl;
    // std::cout << " Error :-------------------------------> " << err << std::endl;
    // std::cout << "____________________________________________" << std::endl;
    // std::cout << std::endl;

    return 0;
}