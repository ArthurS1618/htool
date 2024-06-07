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
using namespace htool;

/// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% /////
Matrix<double> generate_random_matrix(const int &nr, const int &nc) {
    std::vector<double> mat(nr * nc);
    Matrix<double> M(nr, nc);
    for (int l = 0; l < nc; ++l) {
        auto rand = generate_random_vector(nr);
        // std::copy(rand.begin(), rand.end(), mat.data() + l * nr);
        for (int k = 0; k < nr; ++k) {
            M(k, l) = rand[k];
        }
    }
    // M.assign(nr, nc, mat.data(), false);
    return M;
}

std::vector<Matrix<double>> truncation(const Matrix<double> &U1, const Matrix<double> &V1, const Matrix<double> &U2, const Matrix<double> &V2, const double &epsilon) {
    int conc_nc  = U1.nb_cols() + U2.nb_cols();
    int interest = (U1.nb_rows() + V1.nb_cols()) / 2.0;
    std::vector<Matrix<double>> res;
    if (conc_nc > interest) {
        return res;
    } else {
        auto Uconc = conc_col(U1, U2);
        auto Vconc = conc_row_T(V1, V2);
        auto QRu   = QR_factorisation(U1.nb_rows(), Uconc.nb_cols(), Uconc);
        auto QRv   = QR_factorisation(Vconc.nb_rows(), Vconc.nb_cols(), Vconc);
        Matrix<double> RuRv(QRu[0].nb_rows(), QRv[0].nb_cols());
        int ldu      = QRu[1].nb_rows();
        int ldv      = QRv[1].nb_cols();
        int rk       = QRu[1].nb_cols();
        double alpha = 1.0;
        double beta  = 1.0;
        int ldc      = std::max(ldu, ldv);
        Blas<double>::gemm("N", "T", &ldu, &ldv, &rk, &alpha, QRu[1].data(), &ldu, QRv[1].data(), &ldv, &beta, RuRv.data(), &ldc);
        Matrix<double> svdU(RuRv.nb_rows(), std::min(RuRv.nb_rows(), RuRv.nb_cols()));
        Matrix<double> svdV(std::min(RuRv.nb_rows(), RuRv.nb_cols()), RuRv.nb_cols());
        auto S        = compute_svd(RuRv, svdU, svdV);
        double margin = S[0];
        auto it       = std::find_if(S.begin(), S.end(), [epsilon, margin](double s) {
            return s < (epsilon * (1.0 + margin));
        });

        int rep = std::distance(S.begin(), it);
        if (rep < interest) {
            Matrix<double> Urestr(svdU.nb_rows(), rep);
            for (int l = 0; l < rep; ++l) {
                Urestr.set_col(l, svdU.get_col(l));
            }
            Matrix<double> Vrestr(rep, svdV.nb_cols());
            for (int k = 0; k < rep; ++k) {
                Vrestr.set_row(k, svdV.get_row(k));
            }
            Matrix<double> srestr(rep, rep);
            for (int k = 0; k < rep; ++k) {
                srestr(k, k) = S[k];
            }
            // int ldq = QRu[0].nb_rows();
            // int dd  = QRu[0].nb_cols();
            // rk      = Urestr.nb_cols();
            // Matrix<double> temp(ldq, rk);
            // Blas<double>::gemm("N", "N", &ldq, &rk, &dd, &alpha, QRu[0].data(), &ldq, Urestr.data(), &ldq, &beta, temp.data(), &ldq);
            // int ldq = QRu[0].nb_rows();
            // int dd  = QRu[0].nb_cols();
            // rk      = Urestr.nb_cols();

            // Matrix<double> temp(ldq, rk); // Matrice résultante 100x30

            // Blas<double>::gemm("N", "N", &ldq, &rk, &dd, &alpha, QRu[0].data(), &ldq, Urestr.data(), &ldq, &beta, temp.data(), &ldq);
            auto res_U = QRu[0] * Urestr * srestr;
            // auto res_U = temp * srestr;

            auto res_V = Vrestr * Vrestr.transp(QRv[0]);
            res.push_back(res_U);
            res.push_back(res_V);
        }
        return res;
    }
    return res;
}
template <typename T>
int test_sum_expression(int size, int rank, htool::underlying_type<T> epsilon) {

    auto A1 = generate_random_matrix(size, rank);
    auto B1 = generate_random_matrix(rank, size);
    auto A2 = generate_random_matrix(size, rank);
    auto B2 = generate_random_matrix(rank, size);
    // for (int k = 0; k < size; ++k) {
    //     for (int l = 0; l < rank - 1; ++l) {
    //         std::cout << A1(k, l) << ',';
    //     }
    //     std::cout << A1(k, rank - 1) << std::endl;
    // }
    Matrix<double> smart(rank, rank);
    for (int k = 0; k < rank; ++k) {
        smart(k, k) = 1.0 / std::pow(2.718, k);
    }
    A1       = A1 * smart;
    A2       = A2 * smart;
    B1       = smart * B1;
    B2       = smart * B2;
    auto ref = A1 * B1 + A2 * B2;
    // LowRankMatrix<T, T> lr(A1, B1);
    // auto res = lr.compute_lr_update(A2, B2, -1, epsilon);
    auto res = truncation(A1, B1, A2, B2, epsilon);
    if (res.size() > 1) {
        std::cout << "ref : " << ref.nb_rows() << ',' << ref.nb_cols() << std::endl;
        std::cout << "U : " << res[0].nb_rows() << ',' << res[0].nb_cols() << std::endl;
        std::cout << "V : " << res[1].nb_rows() << ',' << res[1].nb_cols() << std::endl;
        double error = normFrob(ref - res[0] * res[1]) / normFrob(ref);
        std::cout << "error : " << error << " avec epsilon = " << epsilon << std::endl;
        std::cout << "rank : " << res[0].nb_cols() << " au lieu de " << 2 * rank << std::endl;

        std ::cout << "_________________________________" << std::endl;
        std ::cout << "_________________________________" << std::endl;

        std ::cout << "_________________________________" << std::endl;
        LowRankMatrix<double, double> LR1(A1, B1);
        LowRankMatrix<double, double> LR2(A2, B2);
        bool flag    = false;
        auto test_lr = LR1.formatted_addition(LR2, 1e-6, flag);
        std::cout << normFrob(ref - (test_lr.Get_U() * test_lr.Get_V())) / normFrob(ref) << std::endl;

        std ::cout << "_________________________________" << std::endl;
        std ::cout << "_________________________________" << std::endl;

        std ::cout << "_________________________________" << std::endl;

        std::cout << "________________TEST SUMEXPRESSION_______________" << std::endl;
        std::cout << std::endl;
        std::vector<double> p1(3 * size);
        create_disk(3, 0.0, size, p1.data());
        ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy_1(size, 3, p1.data(), 2, 2);
        std::shared_ptr<Cluster<double>> root_cluster_1 = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree());
        GeneratorTestDoubleSymmetric generator(3, size, size, p1, p1, root_cluster_1, root_cluster_1);
        Matrix<double> reference_num_htool(size, size);
        generator.copy_submatrix(size, size, 0, 0, reference_num_htool.data());
        HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster_1, root_cluster_1, epsilon, 10, 'N', 'N');

        // build
        auto root_hmatrix = hmatrix_tree_builder.build(generator);
        Matrix<double> ref(size, size);
        copy_to_dense(root_hmatrix, ref.data());
        auto comprr = root_hmatrix.get_compression();
        std::cout << "hmatrix compression : " << comprr << std::endl;
        SumExpression_fast<double, double> sum_expr(&root_hmatrix, &root_hmatrix);
    }
    return 0;
}
