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
        // auto Uconc = conc_col(U1, U2);
        Matrix<double> Uconc(U1.nb_rows(), U1.nb_cols() + U2.nb_cols());
        for (int l = 0; l < U1.nb_cols(); ++l) {
            Uconc.set_col(l, U1.get_col(l));
        }
        for (int l = U1.nb_cols(); l < U1.nb_cols() + U2.nb_cols(); ++l) {
            Uconc.set_col(l, U2.get_col(l - U1.nb_cols()));
        }
        auto Vconc = conc_row_T(V1, V2);
        // Matrix<double> Vconc(V1.nb_cols(), V1.nb_rows() + V2.nb_rows());
        // for (int l = 0; l < V1.nb_rows(); ++l) {
        //     Vconc.set_col(l, V1.get_row(l));
        // }
        // for (int l = V1.nb_rows(); l < V1.nb_rows() + V2.nb_rows(); ++l) {
        //     Vconc.set_col(l, V2.get_row(l - V1.nb_cols()));
        // }
        // QR sur U1, U2 et sur (V1, V2)^T ( nr doit être > nc donc on doit faire QR de la transposé pour Vconc )
        auto QRu = QR_factorisation(U1.nb_rows(), Uconc.nb_cols(), Uconc);
        auto QRv = QR_factorisation(Vconc.nb_rows(), Vconc.nb_cols(), Vconc);
        // Matrix<double> RuRv(QRu[1].nb_rows(), QRv[1].nb_rows());
        // Ru*Rv^T
        std::cout << "erreur QR U" << normFrob(QRu[0] * QRu[1] - Uconc) / normFrob(Uconc) << std::endl;
        std::cout << "erreur QR V" << normFrob(QRv[0] * QRv[1] - Vconc) / normFrob(Vconc) << std::endl;

        auto RuRv = QRu[1] * V1.transp(QRv[1]);
        auto tt   = RuRv;
        Matrix<double> svdU(RuRv.nb_rows(), std::min(RuRv.nb_rows(), RuRv.nb_cols()));
        Matrix<double> svdV(std::min(RuRv.nb_rows(), RuRv.nb_cols()), RuRv.nb_cols());
        // svd sur RuRv^T
        auto S = compute_svd(RuRv, svdU, svdV);
        auto s = S[0];
        Matrix<double> ss(RuRv.nb_rows(), RuRv.nb_cols());
        for (int k = 0; k < RuRv.nb_rows(); ++k) {
            ss(k, k) = S[k];
        }
        std::cout << "erreur SVD " << normFrob(svdU * ss * svdV - tt) / normFrob(tt) << std::endl;
        int rep = 0;
        while (s > epsilon && rep < S.size()) {
            rep += 1;
            s = S[rep];
        }
        rep = rep - 1;
        // std::cout << " rep et interest :" << rep << ',' << interest << " s= " << s << ',' << S[rep - 1] << std::endl;
        if (rep < interest) {
            int nr      = U1.nb_rows();
            int nc      = V1.nb_cols();
            auto Urestr = svdU.trunc_col(rep);
            auto Vrestr = svdV.trunc_row(rep);
            Matrix<double> srestr(rep, rep);
            for (int k = 0; k < rep; ++k) {
                srestr(k, k) = S[k];
            }
            auto res_U = QRu[0] * Urestr * srestr;
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
        smart(k, k) = 1.0 / (rank * rank * 1.0);
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
    }
    return 0;
}
