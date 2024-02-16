#include <chrono>
#include <ctime>
#include <htool/basic_types/matrix.hpp>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/sum_expressions.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <htool/wrappers/wrapper_lapack.hpp>
#include <iostream>

#include <random>

int testt(int x, int y, int (*f)(int a, int b)) {
    return f(x, y);
}
int add(int x, int y) {
    return x + y;
}
using namespace htool;
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

int get_pivot(const std::vector<double> &x, const std::vector<int> &map) {
    double delta = 0.0;
    int pivot    = 0;
    for (int k = 0; k < x.size(); ++k) {
        if (std::abs(x[k]) > delta) {
            if (map[k] == 0) {
                pivot = k;
                delta = std::abs(x[k]);
            }
        }
    }
    return pivot;
}

int get_pivot(const std::vector<double> &x, const double &previous_pivot) {
    double delta = 0.0;
    int pivot    = 0;
    for (int k = 0; k < x.size(); ++k) {
        if (std::abs(x[k]) > delta) {
            if (k != previous_pivot) {
                pivot = k;
                delta = std::abs(x[k]);
            }
        }
    }
    return pivot;
}

std::vector<Matrix<double>> dumb_aca(const Matrix<double> &A, const int &rkmax, const double &epsilon) {
    std::vector<std::vector<double>> col_U, row_V;
    int M = A.nb_rows();
    int N = A.nb_cols();
    std::vector<int> piv_row(M);
    std::vector<int> piv_col(N);

    // first column pivot
    int jr = 0;
    std::vector<double> col(A.nb_rows());
    for (int k = 0; k < A.nb_rows(); ++k) {
        col[k] = A(k, 0);
    }
    // max element and pivot ;
    double delta = 0.0;
    int ir       = 0;
    for (int k = 0; k < col.size(); ++k) {
        if (std::abs(col[k]) > delta) {
            ir    = k;
            delta = std::abs(col[k]);
        }
    }
    // std::cout << "iiiiiiiiiiiiiiiiiir " << ir << std::endl;
    delta = col[ir];
    // scale column
    for (int k = 0; k < A.nb_rows(); ++k) {
        col[k] = col[k] * (1.0 / delta);
    }
    // std::cout << "alpha deeeense : " << (1.0 / delta) << std::endl;
    col_U.push_back(col);
    // row
    std::vector<double> row(A.nb_cols());
    for (int k = 0; k < A.nb_cols(); ++k) {
        row[k] = A(ir, k);
    }
    row_V.push_back(row);
    double error = 100000;
    int rk       = 1;
    // piv_col.push_back(0);
    // piv_row.push_back(ir);
    piv_col[0]  = 1;
    piv_row[ir] = 1;

    while ((rk < rkmax)) {
        // next col pivot
        std::vector<double> previous_row = row_V[rk - 1];
        // auto new_jr                      = get_pivot(previous_row, jr);
        // jr                               = new_jr;
        auto jr     = get_pivot(previous_row, piv_col);
        piv_col[jr] = 1;

        // Uk = col(jr)- sum ( ui vi[jr]) ;
        std::vector<double> current_col(A.nb_rows(), 0.0);
        // for (int k = 0; k < A.nb_rows(); ++k) {
        //     current_col[k] = A(k, jr);
        // }
        std::vector<double> e_jr(A.nb_cols(), 0.0);
        e_jr[jr] = 1.0;
        A.add_vector_product('N', 1.0, e_jr.data(), 1.0, current_col.data());
        int inc = 1;

        for (int k = 0; k < rk; ++k) {
            // col_k -= col_U[k] * row_V[k] *col_V[k][jr]
            double alpha = -1.0 * row_V[k][jr];
            auto col_k   = col_U[k];
            Blas<double>::axpy(&N, &alpha, col_k.data(), &inc, current_col.data(), &inc);
        }
        // next row pivot
        // ir = get_pivot(current_col, -1);
        ir = get_pivot(current_col, -1);
        // piv_row[ir] = 1;
        // piv_row.push_back(ir);

        delta = current_col[ir];
        std::cout << "deltaaaaaa " << delta << std::endl;
        // scale column
        for (int k = 0; k < A.nb_rows(); ++k) {
            current_col[k] = current_col[k] * (1.0 / delta);
        }
        col_U.push_back(current_col);
        // Vk = row(ir) -sum( uj[ir]vj)
        std::vector<double> current_row(A.nb_cols());
        std::vector<double> e_ir(A.nb_rows(), 0.0);
        e_ir[ir] = 1.0;
        A.add_vector_product('T', 1.0, e_ir.data(), 1.0, current_row.data());
        for (int k = 0; k < rk; ++k) {
            double alpha = -1.0 * col_U[k][ir];
            auto row_k   = row_V[k];
            Blas<double>::axpy(&N, &alpha, row_k.data(), &inc, current_row.data(), &inc);
        }
        // for (int k = 0; k < rk; ++k) {
        //     double alpha = -1.0 * col_U[k][ir];
        //     auto row_k   = row_V[k];
        //     for (int l = 0; l < A.nb_cols(); ++l) {
        //         current_row[l] = current_row[l] + alpha * row_k[l];
        //     }
        //     // Blas<double>::axpy(&N, &alpha, row_k.data(), &inc, current_row.data(), &inc);
        // }
        std::cout << rk << "    " << norm2(current_row) << ',' << norm2(current_col) << std::endl;
        row_V.push_back(current_row);
        error = norm2(current_col) * norm2(current_row);
        rk += 1;
    }
    Matrix<double> uu(A.nb_rows(), rk);
    Matrix<double> vv(rk, A.nb_cols());
    for (int l = 0; l < rk; ++l) {
        auto col = col_U[l];
        for (int k = 0; k < A.nb_rows(); ++k) {
            uu(k, l) = col[k];
        }
        auto row = row_V[l];
        for (int k = 0; k < A.nb_cols(); ++k) {
            vv(l, k) = row[k];
        }
    }
    // std::cout << "rk = " << rk << std::endl;
    std::vector<Matrix<double>> res;
    res.push_back(uu);
    res.push_back(vv);

    return res;
}
std::vector<Matrix<double>> ACA_dense(const Matrix<double> &A, const int rkmax, const double varepsilon) {
    std::cout << "--------------------------------------------" << std::endl;
    std::vector<Matrix<double>> res;
    std::vector<std::vector<double>> row_U, col_V;

    std::vector<double> x(A.nb_cols(), 0.0);
    x[0] = 1.0;
    std::vector<double> tempcol(A.nb_rows(), 0.0);
    A.add_vector_product('N', 1.0, x.data(), 1.0, tempcol.data());
    auto delta = std::max_element(tempcol.begin(), tempcol.end());
    int ir     = std::distance(tempcol.begin(), delta);

    // U1 unscaled
    std::vector<double> x0(A.nb_rows(), 0.0);
    x0[ir] = 1.0;
    std::vector<double> row(A.nb_cols(), 0.0);
    A.add_vector_product('T', 1.0, x0.data(), 1.0, row.data());
    double rep = 0.0;
    for (int l = 0; l < A.nb_cols(); ++l) {
        rep += std::abs(row[l] - A(10, l));
    }
    std::cout << "erreur row : " << rep << std::endl;

    // pivot on first row //
    auto delta0 = std::max_element(row.begin(), row.end());
    int jr      = std::distance(row.begin(), delta0);
    // U1
    std::vector<double> scaled(row.size());
    double alpha = 1.0 / *delta0;
    int inc      = 1;
    int N        = A.nb_cols();
    int M        = A.nb_rows();
    Blas<double>::axpy(&N, &alpha, row.data(), &inc, scaled.data(), &inc);

    row_U.push_back(scaled);
    // V1
    std::vector<double> x1(A.nb_cols(), 0.0);
    x1[jr] = 1.0;
    std::vector<double> col(A.nb_rows(), 0.0);
    A.add_vector_product('N', 1.0, x1.data(), 1.0, col.data());
    col_V.push_back(col);
    rep = 0.0;
    for (int k = 0; k < A.nb_rows(); ++k) {
        rep += std::abs(col[k] - A(k, jr));
    }
    std::cout << "erreur col : " << rep << std::endl;
    // second row pivot for the loop
    delta = std::max_element(col.begin(), col.end());
    ir    = std::distance(col.begin(), delta);
    // je suppose qu'on a besoin de plus que le rang 1
    double error = 100000;
    int rk       = 1;
    // while ((error > (rk - 1) * epsilon) && (rk < (target_size + source_size) / 2)) {
    Matrix<double> aa(A.nb_rows(), A.nb_cols());
    for (int k = 0; k < A.nb_rows(); ++k) {
        for (int l = 0; l < A.nb_cols(); ++l) {
            aa(k, l) = col[k] * row[l];
        }
    }
    std::cout << "erreur norme 1 : " << normFrob(A - aa) / normFrob(A) << std::endl;
    // while (rk < rkmax) {
    //     std::cout << "-------------------->rk = " << rk << std::endl;
    //     /////////   row Uk
    //     // get row_A(pivot)
    //     std::vector<double> xk(A.nb_rows(), 0.0);
    //     xk[ir] = 1.0;
    //     std::vector<double> rowk(A.nb_cols(), 0.0);
    //     A.add_vector_product('T', 1.0, x0.data(), 1.0, rowk.data());
    //     // get next pivot jr
    //     // row_Uk  = (row_A_pivot - sum (l(j, pivot_k) row_U(j)))/delta
    //     for (int k = 0; k < rk; ++k) {
    //         alpha         = -1.0 * col_V[k][ir];
    //         auto temp_row = row_U[k];
    //         Blas<double>::axpy(&N, &alpha, temp_row.data(), &inc, rowk.data(), &inc);
    //     }
    //     alpha = 1.0 / *delta;
    //     Blas<double>::axpy(&N, &alpha, rowk.data(), &inc, scaled.data(), &inc);
    //     row_U.push_back(scaled);
    //     delta = std::max_element(rowk.begin(), rowk.end());
    //     jr    = std::distance(rowk.begin(), delta);
    //     std::cout << "row push " << norm2(scaled) << std::endl;

    //     //////// col Vk
    //     // get next pivot
    //     delta = std::max_element(rowk.begin(), rowk.end());
    //     ir    = std::distance(rowk.begin(), delta);
    //     std::cout << "pivot pour v " << *delta << ',' << ir << std::endl;
    //     // col_A_pivot
    //     std::vector<double> yk(A.nb_cols(), 0.0);
    //     yk[ir] = 1.0;
    //     std::vector<double> colk(A.nb_rows(), 0.0);
    //     A.add_vector_product('N', 1.0, yk.data(), 1.0, colk.data());
    //     std::cout << "get col" << norm2(colk) << std::endl;

    //     // col_Vk = (col_A_pivot - sum u(i, pivot)col_V(i))
    //     for (int k = 0; k < rk; ++k) {
    //         alpha = -1.0 * row_U[k][jr];
    //         std::cout << "alpha  " << row_U[k][jr] << std::endl;
    //         auto temp_col = col_V[k];
    //         Blas<double>::axpy(&M, &alpha, temp_col.data(), &inc, colk.data(), &inc);
    //     }
    //     col_V.push_back(colk);
    //     std::cout << "col push" << norm2(colk) << std::endl;

    //     // error pour arret et rank update
    //     error = norm2(rowk) * norm2(colk);
    //     std::cout << "error for k=" << rk << "  = " << error << std::endl;
    //     rk += 1;
    //     // std::cout << "stooping criterion " << (error > (rk - 1) * epsilon) << ',' << (rk < (target_size + source_size) / 2) << std::endl;
    // }
    if (rk == (M + N) / 2) {
        // pas de bonne approx de rang faible
        Matrix<double> unull(1, 1);
        Matrix<double> vnull(1, 1);
        res.push_back(unull);
        res.push_back(vnull);
    } else {
        // on concatène les lignes et les colllonnes ;
        Matrix<double> V(M, rk);
        Matrix<double> U(rk, N);
        for (int l = 0; l < rk; ++l) {
            auto vl = col_V[l];
            for (int k = 0; k < M; ++k) {
                V(k, l) = vl[k];
            }
        }
        for (int k = 0; k < rk; ++k) {
            auto uk = row_U[k];
            for (int l = 0; l < N; ++l) {
                U(k, l) = uk[l];
            }
        }
        res.push_back(V);
        res.push_back(U);
    }
    return res;
}
//////////////////////////
//// GENERATOR -> Prend Une matrice ou un fichier
//////////////////////////
template <class CoefficientPrecision, class CoordinatePrecision>
class Matricegenerator : public VirtualGenerator<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> mat;
    const Cluster<CoordinatePrecision> &target;
    const Cluster<CoordinatePrecision> &source;

  public:
    Matricegenerator(Matrix<CoefficientPrecision> &mat0, const Cluster<CoordinatePrecision> &target0, const Cluster<CoordinatePrecision> &source0) : mat(mat0), target(target0), source(source0){};

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
// std::vector<double> generate_random_vector(int size) {
//     std::random_device rd;  // Source d'entropie aléatoire
//     std::mt19937 gen(rd()); // Générateur de nombres pseudo-aléatoires

//     std::uniform_real_distribution<double> dis(1.0, 10.0); // Plage de valeurs pour les nombres aléatoires (ici de 1.0 à 10.0)

//     std::vector<double> random_vector;
//     random_vector.reserve(size); // Allocation de mémoire pour le vecteur

//     for (int i = 0; i < size; ++i) {
//         random_vector.push_back(dis(gen)); // Ajout d'un nombre aléatoire dans la plage à chaque itération
//     }

//     return random_vector;
// }

void saveVectorToFile(const std::vector<double> &data, const std::string &filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        // Écrit tous les éléments sur une seule ligne avec des virgules comme séparateurs
        for (size_t i = 0; i < data.size() - 1; ++i) {
            file << data[i];
            file << ",";
        }
        file << data[data.size() - 1];

        file.close();
        std::cout << "Vecteur enregistré dans le fichier : " << filename << std::endl;
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour l'écriture." << std::endl;
    }
}

///////////////////////////////////////////////////////////////////
int main() {
    std::vector<double> time_assmbl;
    std::vector<double> time_vect;
    std::vector<double> time_mat;
    std::vector<double> time_lu;

    std::vector<double> err_asmbl;
    std::vector<double> err_vect;
    std::vector<double> err_mat;
    std::vector<double> err_lu;

    std::vector<double> compr_ratio;

    std::cout << "____________________________" << std::endl;
    for (int k = 12; k < 13; ++k) {

        int size = 600 + std::pow(2.0, k / 30.0 * 15.0);
        // 600 + 133 * k;

        std::cout << "_________________________________________" << std::endl;
        std::cout << "HMATRIX" << std::endl;
        double epsilon = 1e-7;
        double eta     = 10;
        std::cout << "size =" << size << std::endl;
        ////// GENERATION MAILLAGE
        std::vector<double> p1(3 * size);
        create_disk(3, 0.0, size, p1.data());

        std::cout << "création cluster ok" << std::endl;
        //////////////////////
        // Clustering
        //////////////////////
        ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy_1(size, 3, p1.data(), 2, 2);

        std::shared_ptr<Cluster<double>> root_cluster_1 = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree());
        std::cout << "Cluster tree ok" << std::endl;
        // Generator
        auto permutation = root_cluster_1->get_permutation();
        GeneratorDouble generator(3, size, size, p1, p1, root_cluster_1, root_cluster_1);

        Matrix<double> reference_num_htool(size, size);
        generator.copy_submatrix(size, size, 0, 0, reference_num_htool.data());
        // HMATRIX
        HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');

        // build
        auto start_time_asmbl = std::chrono::high_resolution_clock::now();
        auto root_hmatrix     = hmatrix_tree_builder.build(generator);
        auto end_time_asmbl   = std::chrono::high_resolution_clock::now();
        auto duration_asmbl   = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_asmbl - start_time_asmbl).count();
        time_assmbl.push_back(duration_asmbl);
        Matrix<double> ref(size, size);
        copy_to_dense(root_hmatrix, ref.data());

        auto er_b = normFrob(reference_num_htool - ref) / normFrob(reference_num_htool);
        err_asmbl.push_back(er_b);
        std::cout << "asmbl ok " << root_hmatrix.get_compression() << std::endl;

        // Matrice vecteur
        // auto x               = generate_random_vector(size);
        // auto start_time_vect = std::chrono::high_resolution_clock::now();
        // std::vector<double> y(size, 0.0);
        // root_hmatrix.add_vector_product('N', 1.0, x.data(), 1.0, y.data());
        // auto end_time_vect = std::chrono::high_resolution_clock::now();
        // auto duration_vect = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_vect - start_time_vect).count();
        // time_vect.push_back(duration_vect);
        // auto temp = reference_num_htool * x;
        // auto er_v = norm2(y - temp) / norm2(temp);
        // err_vect.push_back(er_v);
        // std::cout << "vect ok " << std::endl;

        // Smexpression

        SumExpression<double, double> test_sumexpr(&root_hmatrix, &root_hmatrix);
        auto x_sumexpr      = generate_random_vector(size);
        auto y_prod_sumexpr = test_sumexpr.new_prod('N', x_sumexpr);
        auto prod           = reference_num_htool * reference_num_htool;
        std::vector<double> y_prod_ref_N(size);
        prod.add_vector_product('N', 1.0, x_sumexpr.data(), 1.0, y_prod_ref_N.data());
        std::cout << " error prod " << norm2(y_prod_sumexpr - y_prod_ref_N) / norm2(y_prod_ref_N) << std::endl;
        auto test_restrict = test_sumexpr.Restrict_clean(root_hmatrix.get_children()[0]->get_target_cluster().get_size(), root_hmatrix.get_target_cluster().get_offset(), root_hmatrix.get_children()[0]->get_source_cluster().get_size(), root_hmatrix.get_source_cluster().get_offset());
        auto ref_restr     = prod.get_block(root_cluster_1->get_children()[0]->get_size(), root_cluster_1->get_children()[0]->get_size(), 0, 0);
        auto sh            = test_restrict.get_sh();
        std::cout << sh.size() << std::endl;
        auto x_restr        = generate_random_vector(root_cluster_1->get_children()[0]->get_size());
        auto y_prod_restr_N = test_restrict.new_prod('N', x_restr);
        std::vector<double> y_ref_restr_N(root_cluster_1->get_children()[0]->get_size());
        ref_restr.add_vector_product('N', 1.0, x_restr.data(), 1.0, y_ref_restr_N.data());
        std::cout << "prod restr " << norm2(y_ref_restr_N - y_prod_restr_N) / norm2(y_ref_restr_N) << std::endl;

        std::vector<double> y_ref_restr_T(root_cluster_1->get_children()[0]->get_size());
        ref_restr.add_vector_product('T', 1.0, x_restr.data(), 1.0, y_ref_restr_T.data());
        auto y_prod_restr_T = test_restrict.new_prod('T', x_restr);
        std::cout << "prod restr transp " << norm2(y_prod_restr_T - y_ref_restr_T) / norm2(y_ref_restr_T) << std::endl;

        auto &t         = ((root_cluster_1->get_children()[1])->get_children())[1];
        auto &s         = ((root_cluster_1->get_children()[0])->get_children())[0];
        auto temp_restr = test_sumexpr.Restrict_clean(root_cluster_1->get_children()[1]->get_size(), root_cluster_1->get_children()[1]->get_offset(), root_cluster_1->get_children()[0]->get_size(), root_cluster_1->get_children()[0]->get_offset());
        auto restr_adm  = temp_restr.Restrict_clean(t->get_size(), t->get_offset(), s->get_size(), s->get_offset());
        std::cout << "?" << restr_adm.get_sh().size() << std::endl;
        auto ref_adm = prod.get_block(t->get_size(), s->get_size(), t->get_offset(), s->get_offset());
        // int rank   = 6;
        // auto udata = generate_random_vector(rank * size);
        // auto vdata = generate_random_vector(rank * size);
        // Matrix<double> uu(size, rank);
        // Matrix<double> vv(rank, size);
        // uu.assign(size, rank, udata.data(), false);
        // vv.assign(rank, size, vdata.data(), false);

        // std::cout << normFrob(uu) << ',' << normFrob(vv) << std::endl;
        // auto ref_lr = uu * vv;
        // std::cout << normFrob(ref_lr) << std::endl;

        // std::cout << "ACA sumexpr" << std::endl;
        // auto UV = restr_adm.ACA(20, epsilon);
        // std::cout << " ACA over " << std::endl;
        // auto U = UV[0];
        // auto V = UV[1];
        // std::cout << U.nb_rows() << ',' << U.nb_cols() << ',' << V.nb_rows() << ',' << V.nb_cols() << std::endl;
        // std::cout << ref_adm.nb_rows() << ',' << ref_adm.nb_cols() << ',' << restr_adm.get_target_size() << ',' << restr_adm.get_source_size() << std::endl;
        // std::cout << "erreur ACA " << normFrob(U * V - ref_adm) / normFrob(ref_adm) << std::endl;
        // auto UVtest = dumb_aca(ref_adm, 4, epsilon);
        // auto Utest  = UVtest[0];
        // auto Vtest  = UVtest[1];
        // std::cout << "erreur dense " << normFrob(Utest * Vtest - ref_adm) / normFrob(ref_adm) << std::endl;

        // auto xtest = generate_random_vector(s->get_size());
        // std::cout << norm2((ref_adm * xtest) - (restr_adm.new_prod('N', xtest))) / norm2((ref_adm * xtest)) << std::endl;

        // auto UV = ACA_dense(ref_lr, 1, 0.0000001);
        // auto U  = UV[0];
        // auto V  = UV[1];
        // std::cout << "output aca " << std::endl;
        // std::cout << U.nb_rows() << ',' << U.nb_cols() << ',' << normFrob(U) << std::endl;
        // std::cout << V.nb_rows() << ',' << V.nb_cols() << ',' << normFrob(V) << std::endl;
        // std::cout << normFrob(U * V - ref_lr) / normFrob(ref_lr) << std::endl;
        // auto lr = dumb_aca(ref_lr, rank, epsilon);
        // auto U  = lr[0];
        // auto V  = lr[1];
        // std::cout << normFrob(U * V - ref_lr) / normFrob(ref_lr) << std::endl;

        // Matrice Matrice
        auto start_time_mat = std::chrono::high_resolution_clock::now();
        std::cout << "mat " << std::endl;
        auto prodd = root_hmatrix.hmatrix_product(root_hmatrix);
        std::cout << "mat ok" << std::endl;
        auto end_time_mat = std::chrono::high_resolution_clock::now();
        auto duration_mat = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_mat - start_time_mat).count();
        time_mat.push_back(duration_mat);
        auto tempm = reference_num_htool * reference_num_htool;
        Matrix<double> dense_prod(size, size);
        copy_to_dense(prodd, dense_prod.data());
        auto er_m = normFrob(dense_prod - tempm) / normFrob(tempm);
        err_mat.push_back(er_m);
        auto compr = prodd.get_compression();
        compr_ratio.push_back(compr);
        std::cout << "erreur :" << er_m << "    , compr" << compr << std::endl;

        // root_hmatrix.save_plot("tete_produit");
        // /// HLU
        // Matrix<double> L(size, size);
        // Matrix<double> U(size, size);
        // std::vector<int> ipiv(size, 0.0);
        // int info = -1;
        // auto A   = reference_num_htool;
        // Lapack<double>::getrf(&size, &size, A.data(), &size, ipiv.data(), &info);

        // for (int i = 0; i < size; ++i) {
        //     L(i, i) = 1;
        //     U(i, i) = A(i, i);

        //     for (int j = 0; j < i; ++j) {
        //         L(i, j) = A(i, j);
        //         U(j, i) = A(j, i);
        //     }
        // }
        // std::cout << "________________________" << std::endl;
        // std::cout << "Assemblage de matrice hiérarchique triangulaire pour nos tests : " << std::endl;
        // // Matrice Lh et Uh
        // auto ll = get_unperm_mat(L, *root_cluster_1, *root_cluster_1);
        // Matricegenerator<double, double> l_generator(ll, *root_cluster_1, *root_cluster_1);
        // HMatrixTreeBuilder<double, double> lh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
        // auto Lh = lh_builder.build(l_generator);
        // Matrix<double> ldense(size, size);
        // copy_to_dense(Lh, ldense.data());
        // std::cout << "erreur Lh :" << normFrob(L - ldense) / normFrob(L) << std::endl;

        // auto uu = get_unperm_mat(U, *root_cluster_1, *root_cluster_1);
        // Matricegenerator<double, double> u_generator(uu, *root_cluster_1, *root_cluster_1);
        // HMatrixTreeBuilder<double, double> uh_builder(root_cluster_1, root_cluster_1, epsilon, eta, 'N', 'N');
        // auto Uh = uh_builder.build(u_generator);
        // Matrix<double> udense(size, size);
        // copy_to_dense(Uh, udense.data());
        // std::cout << "erreur uh : " << normFrob(U - udense) / normFrob(U) << std::endl;
        // auto prod_LU = Lh.hmatrix_product(Uh);

        // HMatrix<double, double> Llh(root_cluster_1, root_cluster_1);
        // Llh.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // Llh.set_epsilon(epsilon);
        // Llh.set_low_rank_generator(prod_LU.get_low_rank_generator());

        // HMatrix<double, double> Uuh(root_cluster_1, root_cluster_1);
        // Uuh.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // Uuh.set_epsilon(epsilon);
        // Uuh.set_low_rank_generator(prod_LU.get_low_rank_generator());
        // Uuh.copy_zero(prod_LU);
        // Llh.copy_zero(prod_LU);
        // Matrix<double> prod_dense(size, size);
        // copy_to_dense(prod_LU, prod_dense.data());

        // HMatrix<double, double> Permutation(root_cluster_1, root_cluster_1);
        // Permutation.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // Permutation.set_epsilon(epsilon);
        // Permutation.set_low_rank_generator(prod_LU.get_low_rank_generator());
        // Permutation.copy_zero(prod_LU);
        // // Matrix<double> ref_lu(size, size);
        // // copy_to_dense(produit_LU, ref_lu.data());
        // Matrix<double> dense_perm(size, size);
        // // std::cout << "norme de Lh*Uh la matrice que l'on factorise  : " << normFrob(ref_lu) << "!!" << normFrob(ldense * udense) << std::endl;
        // HMatrix<double, double> unperm(root_cluster_1, root_cluster_1);

        // Permutation.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // unperm.set_admissibility_condition(prod_LU.get_admissibility_condition());
        // Permutation.set_low_rank_generator(prod_LU.get_low_rank_generator());
        // unperm.set_low_rank_generator(prod_LU.get_low_rank_generator());
        // unperm.copy_zero(prod_LU);

        // std::cout << "_________________________________________" << std::endl;
        // auto start_time_lu = std::chrono::high_resolution_clock::now();

        // HMatrix_PLU(prod_LU, *root_cluster_1, Llh, Uuh, Permutation, unperm);
        // auto end_time_lu = std::chrono::high_resolution_clock::now();
        // auto duration_lu = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_lu - start_time_lu).count();
        // time_lu.push_back(duration_lu);
        // copy_to_dense(Permutation, dense_perm.data());
        // Matrix<double> denseunperm(size, size);
        // copy_to_dense(unperm, denseunperm.data());

        // auto test_lu = Llh.hmatrix_product(Uuh);
        // Matrix<double> lu_dense(size, size);
        // copy_to_dense(test_lu, lu_dense.data());
        // double erlu = normFrob(prod_dense - dense_perm * lu_dense) / normFrob(reference_num_htool);
        // err_lu.push_back(erlu);
        // std::cout << erlu << std::endl;
        // Matrix<double> lldense(size, size);
        // copy_to_dense(Llh, lldense.data());
        // Matrix<double> uudense(size, size);
        // copy_to_dense(Uuh, uudense.data());
        // std::cout << "erreur lu " << normFrob(prod_dense - dense_perm * (lldense * uudense)) / normFrob(prod_dense) << " L : " << normFrob(L - lldense) / normFrob(L) << "             U: " << normFrob(U - uudense) / normFrob(udense) << std::endl;

        // HMatrix<double, double> lh(root_cluster_1, root_cluster_1);
        // lh.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
        // lh.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
        // lh.copy_zero(root_hmatrix);

        // HMatrix<double, double> uh(root_cluster_1, root_cluster_1);
        // uh.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
        // uh.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
        // uh.copy_zero(root_hmatrix);

        // HMatrix<double, double> Permutation(root_cluster_1, root_cluster_1);
        // Permutation.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
        // Permutation.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
        // Permutation.copy_zero(root_hmatrix);

        // HMatrix<double, double> unperm(root_cluster_1, root_cluster_1);
        // unperm.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
        // unperm.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
        // unperm.copy_zero(root_hmatrix);

        // HMatrix_PLU(root_hmatrix, *root_cluster_1, lh, uh, Permutation, unperm);
        // Matrix<double> lldense(size, size);
        // copy_to_dense(lh, lldense.data());
        // Matrix<double> uudense(size, size);
        // copy_to_dense(uh, uudense.data());
        // Matrix<double> dense_perm(size, size);
        // copy_to_dense(Permutation, dense_perm.data());

        // std::cout << "erreur lu " << normFrob(dense_perm * ref - (lldense * uudense)) / normFrob(ref) << " L : " << normFrob(L - lldense) / normFrob(L) << "             U: " << normFrob(U - uudense) / normFrob(uudense) << std::endl;
    }
    // saveVectorToFile(time_assmbl, "time_assemble_eps4.txt");
    // saveVectorToFile(time_vect, "time_vect_eps4.txt");
    // saveVectorToFile(time_mat, "time_mat_eps4.txt");
    // // saveVectorToFile(time_lu, "time_lu_eps4.txt");

    // // saveVectorToFile(err_asmbl, "error_assemble_eps4.txt");
    // // saveVectorToFile(err_vect, "error_vect_eps4.txt");
    // saveVectorToFile(err_mat, "error_mat_eps4.txt");
    // saveVectorToFile(err_lu, "error_lu_eps4.txt");

    // saveVectorToFile(compr_ratio, "compression_ratio_eps4.txt");
}