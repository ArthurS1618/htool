#include <chrono>
#include <ctime>
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/sum_expressions.hpp>
#include <htool/hmatrix/sumexpr.hpp>

#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/matrix.hpp>
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

template <typename T>
void get_lu_factorisation(const Matrix<T> &M, Matrix<T> &L, Matrix<T> &U, std::vector<int> &P) {
    auto A   = M;
    int size = A.nb_rows();
    std::vector<int> ipiv(size, 0.0);
    int info = -1;
    Lapack<T>::getrf(&size, &size, A.data(), &size, ipiv.data(), &info);
    for (int i = 0; i < size; ++i) {
        L(i, i) = 1;
        U(i, i) = A(i, i);

        for (int j = 0; j < i; ++j) {
            L(i, j) = A(i, j);
            U(j, i) = A(j, i);
        }
    }
    for (int k = 1; k < size + 1; ++k) {
        P[k - 1] = k;
    }
    for (int k = 0; k < size; ++k) {
        if (ipiv[k] - 1 != k) {
            int temp       = P[k];
            P[k]           = P[ipiv[k] - 1];
            P[ipiv[k] - 1] = temp;
        }
    }
}

// ARTHUR : A = QR avec lapack ------------> row  A > colA
template <typename T>
std::vector<Matrix<T>> QR_factorisation(const int &target_size, const int &source_size, Matrix<T> A) {
    int lda_u   = target_size;
    int lwork_u = 10 * target_size;
    int info_u;

    int N = A.nb_cols();
    std::vector<T> work_u(lwork_u);
    std::vector<T> tau_u(N);
    Lapack<T>::geqrf(&target_size, &N, A.data(), &lda_u, tau_u.data(), work_u.data(), &lwork_u, &info_u);
    Matrix<T> R_u(N, N);
    for (int k = 0; k < N; ++k) {
        for (int l = k; l < N; ++l) {
            R_u(k, l) = A(k, l);
        }
    }
    std::vector<T> workU(lwork_u);
    Lapack<T>::orgqr(&target_size, &N, &std::min(target_size, N), A.data(), &lda_u, tau_u.data(), workU.data(), &lwork_u, &info_u);
    std::vector<Matrix<T>> res;
    // Matrix<double> Q, R;
    // Q.assign(A.nb_rows(), A.nb_cols(), A.data());
    res.push_back(A);
    res.push_back(R_u);
    return res;
}
template <typename T>
Matrix<T> transp(const Matrix<T> &M) {
    Matrix<T> res(M.nb_cols(), M.nb_rows());
    for (int k = 0; k < M.nb_cols(); ++k) {
        for (int l = 0; l < M.nb_rows(); ++l) {
            res(k, l) = M(l, k);
        }
    }
    return res;
}
template <typename T>
int test_sum_expression(int size, int rank, htool::underlying_type<T> epsilon) {
    double eta = 10.0;
    // std::cout << "________________TEST SUMEXPRESSION_______________" << std::endl;
    std::vector<double> p1(3 * size);
    std::cout << p1.size() << ',' << size << std::endl;
    create_disk(3, 0.0, size, p1.data());

    std::cout << "création cluster ok " << p1.size() << std::endl;
    //////////////////////

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

    //////////////////////
    // Clustering
    ClusterTreeBuilder<double> recursive_build_strategy_1;
    std::shared_ptr<Cluster<double>> root_cluster = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree(size, 3, p1.data(), 2, 2));
    Matrix<double> reference(size, size);
    GeneratorTestDouble generator(3, size, size, p1, p1, *root_cluster, *root_cluster, true, true);
    generator.copy_submatrix(size, size, 0, 0, reference.data());
    std::cout << "norme generator: " << normFrob(reference) << std::endl;

    HMatrixTreeBuilder<double, double> hmatrix_tree_builder(*root_cluster, *root_cluster, epsilon, eta, 'N', 'N', -1, -1, -1);

    auto root_hmatrix = hmatrix_tree_builder.build(generator);
    Matrix<double> root_dense(size, size);
    copy_to_dense(root_hmatrix, root_dense.data());
    std::cout << " erreur assemblage : " << normFrob(root_dense - reference) / normFrob(reference) << std::endl;
    // SumExpression_fast<double, double> S(&root_hmatrix, &root_hmatrix);

    // auto &t1    = root_cluster->get_children()[0];
    // auto &s2    = root_cluster->get_children()[1];
    // auto &t11   = t1->get_children()[0];
    // auto &s22   = s2->get_children()[1];
    auto ref_HK = reference * reference;
    // // test prod ;
    // auto x_rand = generate_random_vector(size);
    // auto yref   = ref_HK * x_rand;
    // std::vector<double> y_sumexpr(size);
    // y_sumexpr = S.prod('N', x_rand);
    // std::cout << "erreur produit (H, K) :" << norm2(yref - y_sumexpr) / norm2(yref) << std::endl;

    // // test restrict

    // auto t_temp       = &*root_cluster;
    // auto s_temp       = &*root_cluster;
    // auto sumexpr_temp = S;
    // for (int k = 0; k < root_cluster->get_minimal_depth(); ++k) {
    //     auto &t_child = t_temp->get_children()[0];
    //     auto &s_child = s_temp->get_children()[1];
    //     auto S_child  = sumexpr_temp.restrict_ACA(*t_child, *s_child);
    //     std::cout << "depth : " << k << std::endl;
    //     std::cout << " sr et sh pour restrict " << S_child.get_sr().size() << ',' << S_child.get_sh().size() << std::endl;

    //     auto x_rand_child    = generate_random_vector(s_child->get_size());
    //     auto HK_child        = copy_sub_matrix(ref_HK, t_child->get_size(), s_child->get_size(), t_child->get_offset(), s_child->get_offset());
    //     auto y_ref_child     = HK_child * x_rand_child;
    //     auto y_sumexpr_child = S_child.prod('N', x_rand_child);
    //     std::cout << "erreur prod sur le restrict : " << norm2(y_ref_child - y_sumexpr_child) / norm2(y_ref_child) << std::endl;
    //     std::cout << "______________________________________________________" << std::endl;
    //     auto x_rand_child_T = generate_random_vector(t_child->get_size());
    //     std::vector<double> y_ref_child_T(s_child->get_size());
    //     double alpha = 1.0;
    //     HK_child.add_vector_product('T', 1.0, x_rand_child_T.data(), 1.0, y_ref_child_T.data());
    //     auto y_sumexpr_child_T = S_child.prod('T', x_rand_child_T);
    //     std::cout << "erreur prod sur le restrict transp : " << norm2(y_ref_child_T - y_sumexpr_child_T) / norm2(y_ref_child_T) << std::endl;

    //     std::cout << "______________________________________________________" << std::endl;
    //     std::cout << "______________________________________________________" << std::endl;

    //     std::cout << "______________________________________________________" << std::endl;
    //     std::cout << "______________________________________________________" << std::endl;

    //     t_temp       = t_child.get();
    //     s_temp       = s_child.get();
    //     sumexpr_temp = S_child;
    // }

    std::cout << "////////////////////////////////////////" << std::endl;
    auto hmat_prod = root_hmatrix.hmatrix_product_fast(root_hmatrix);
    Matrix<double> prod_dense(size, size);
    copy_to_dense(hmat_prod, prod_dense.data());
    std::cout << "erreur produit : " << normFrob(ref_HK - prod_dense) / normFrob(ref_HK) << " avec epsilon = " << epsilon << std::endl;
    std::cout << "compression produit : " << hmat_prod.get_compression() << std::endl;
    std::cout << "compression reference : " << root_hmatrix.get_compression() << std::endl;

    // double tt = 0.0;
    // for (int k = 952; k < 968; ++k) {
    //     for (int l = 984; l < 1000; ++l) {
    //         tt += std::pow(ref_HK(k, l), 2.0);
    //     }
    // }
    // std::cout << "0 ? : " << tt << std::endl;

    // test restrict
    // std::cout << std::endl;
    // std::vector<double> p1(3 * size);
    // create_disk(3, 0.0, size, p1.data());
    // ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy_1(size, 3, p1.data(), 2, 2);
    // std::shared_ptr<Cluster<double>> root_cluster_1 = std::make_shared<Cluster<double>>(recursive_build_strategy_1.create_cluster_tree());
    // GeneratorTestDoubleSymmetric generator(3, size, size, p1, p1, root_cluster_1, root_cluster_1);
    // Matrix<double> reference_num_htool(size, size);
    // generator.copy_submatrix(size, size, 0, 0, reference_num_htool.data());
    // HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster_1, root_cluster_1, epsilon, 10, 'N', 'N');

    // // build
    // auto root_hmatrix = hmatrix_tree_builder.build(generator);
    // Matrix<double> ref(size, size);
    // copy_to_dense(root_hmatrix, ref.data());
    // auto comprr = root_hmatrix.get_compression();
    // std::cout << "hmatrix compression : " << comprr << std::endl;
    // SumExpression_fast<double, double> sum_expr(&root_hmatrix, &root_hmatrix);

    return 0;
}
