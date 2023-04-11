#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/sum_expressions.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_hmatrix_product(int size_1, int size_2, int size_3, htool::underlying_type<T> epsilon) {

    srand(1);
    bool is_error = false;

    // Geometry
    vector<double> p1(3 * size_1), p2(3 * size_2), p3(3 * size_3);
    create_disk(3, 0., size_1, p1.data());
    create_disk(3, 0.5, size_2, p2.data());
    create_disk(3, 1., size_3, p3.data());

    // Clustering
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_1(size_1, 3, p1.data(), 2, 2);
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_2(size_2, 3, p2.data(), 2, 2);
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> recursive_build_strategy_3(size_3, 3, p3.data(), 2, 2);

    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_1 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_1.create_cluster_tree());
    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_2 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_2.create_cluster_tree());
    std::shared_ptr<Cluster<htool::underlying_type<T>>> root_cluster_3 = make_shared<Cluster<htool::underlying_type<T>>>(recursive_build_strategy_3.create_cluster_tree());

    // Generator
    GeneratorTestType generator_21(3, size_2, size_1, p2, p1, root_cluster_2, root_cluster_1);
    GeneratorTestType generator_32(3, size_3, size_2, p3, p2, root_cluster_3, root_cluster_2);

    // References
    Matrix<T> dense_matrix_21(root_cluster_2->get_size(), root_cluster_1->get_size()), dense_matrix_32(root_cluster_3->get_size(), root_cluster_2->get_size());
    generator_21.copy_submatrix(dense_matrix_21.nb_rows(), dense_matrix_21.nb_cols(), 0, 0, dense_matrix_21.data());
    generator_32.copy_submatrix(dense_matrix_32.nb_rows(), dense_matrix_32.nb_cols(), 0, 0, dense_matrix_32.data());

    Matrix<T> reference_dense_matrix = dense_matrix_32 * dense_matrix_21;

    // HMatrix
    double eta = 10;

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_21(root_cluster_2, root_cluster_1, epsilon, eta, 'N', 'N');
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_32(root_cluster_3, root_cluster_2, epsilon, eta, 'N', 'N');

    // build
    auto root_hmatrix_21 = hmatrix_tree_builder_21.build(generator_21);
    auto root_hmatrix_32 = hmatrix_tree_builder_32.build(generator_32);
    // auto leaf            = root_hmatrix_21.get_leaves();
    // std::vector<const HMatrix<T, htool::underlying_type<T>> *> lowrank;
    // std::vector<const HMatrix<T, htool::underlying_type<T>> *> hmat;
    // std::vector<const LowRankMatrix<T, htool::underlying_type<T>> *> vect_lr;
    // std::cout << leaf.size() << std::endl;
    // for (int k = 0; k < leaf.size(); ++k) {
    //     auto &l = leaf[k];
    //     if (l->is_low_rank()) {
    //         lowrank.push_back(l);
    //         auto lr = l->get_low_rank_data();
    //         vect_lr.push_back(lr);
    //     } else {
    //         hmat.push_back(l);
    //     }
    // }

    // auto &H_children = root_hmatrix_21.get_children();
    // for (int k = 0; k < H_children.size(); ++k) {
    //     auto &Hk = H_children[k];
    //     std::cout << Hk->get_target_cluster().get_size() << ',' << Hk->get_source_cluster().get_size() << std::endl;
    // }
    // SumExpression<T, htool::underlying_type<T>> sumexpr(&root_hmatrix_21, &root_hmatrix_21);
    // auto srestr = sumexpr.Restrict(100, 0, 100, 0);
    // std::cout << srestr.get_sh().size() << ',' << srestr.get_sr().size() << std::endl;
    // auto srestr1 = srestr.Restrict(50, 0, 50, 0);
    // std::cout << srestr1.get_sh().size() << ',' << srestr1.get_sr().size() << std::endl;
    // auto ssr = srestr1.get_sr();
    // auto lr  = ssr[0];
    // std::cout << lr->nb_rows() << ',' << lr->nb_cols() << ',' << lr->rank_of() << std::endl;
    // auto srestr2 = srestr1.Restrict(25, 0, 25, 0);
    // std::cout << srestr2.get_sh().size() << ',' << srestr2.get_sr().size() << std::endl;
    // auto sssr = srestr2.get_sr();
    // for (int k = 0; k < sssr.size(); ++k) {
    //     auto lllr = sssr[k];
    //     std::cout << lllr->nb_rows() << ',' << lllr->nb_cols() << ',' << lllr->rank_of() << std::endl;
    // }

    SumExpression<T, htool::underlying_type<T>> stest(&root_hmatrix_32, &root_hmatrix_21);
    std::cout << stest.get_coeff(3, 8) << ',' << reference_dense_matrix(3, 8) << std::endl;
    std::cout << stest.get_coeff(3, 8) << std::endl;
    // //////////////////////////////////////
    auto root_hmatrix_31 = root_hmatrix_32.hmatrix_product(root_hmatrix_21);

    // Compute error
    Matrix<T> hmatrix_to_dense_31(root_hmatrix_31.get_target_cluster().get_size(), root_hmatrix_31.get_source_cluster().get_size());
    copy_to_dense(root_hmatrix_31, hmatrix_to_dense_31.data());
    std::cout << "?????" << std::endl;
    std::cout << normFrob(reference_dense_matrix - hmatrix_to_dense_31) / normFrob(reference_dense_matrix) << "\n";

    ////////////////////////////////////////////////////////////////
    // Matrix<T> hmatrix_to_dense_21(root_hmatrix_21.get_target_cluster().get_size(), root_hmatrix_21.get_source_cluster().get_size()), hmatrix_to_dense_32(root_hmatrix_32.get_target_cluster().get_size(), root_hmatrix_32.get_source_cluster().get_size());
    // copy_to_dense(root_hmatrix_21, hmatrix_to_dense_21.data());
    // copy_to_dense(root_hmatrix_32, hmatrix_to_dense_32.data());
    // std::cout << normFrob(dense_matrix_21 - hmatrix_to_dense_21) / normFrob(dense_matrix_21) << "\n";
    // std::cout << normFrob(dense_matrix_32 - hmatrix_to_dense_32) / normFrob(dense_matrix_32) << "\n";

    // // HMatrix product
    // // HMatrix result_hmatrix = root_hmatrix_32.prod(root_hmatrix_21);
    // // HMatrix result_hmatrix = prod(root_hmatrix_32,root_hmatrix_21);

    // cout << "TEST ARTHUR POUR BIEN MANIPULER" << endl;
    // // Hmatrix
    // cout << root_hmatrix_21.get_target_cluster().get_size() << ',' << root_hmatrix_21.get_source_cluster().get_size() << endl;
    // //--------> ca a pas l'aire d'(avoir changer
    // // Accés au fils
    // cout << "nb sons" << endl;
    // cout << root_hmatrix_21.get_children().size() << endl;
    // auto &v  = root_hmatrix_21.get_children();
    // auto &h1 = v[0];
    // auto &h2 = v[1];
    // cout << h1->get_target_cluster().get_size() << ',' << h2->get_source_cluster().get_size() << endl;
    // cout << h1->get_target_cluster().get_offset() << ',' << h2->get_target_cluster().get_offset() << endl;

    // // Test sur les sum expression
    // cout << "TEST SUM EXPRESSION" << endl;
    // cout << "____________________" << endl;
    // // SumExpression<T, htool::underlying_type<T>> sumexpr(h1.get(), h2.get());
    // SumExpression<T, htool::underlying_type<T>> sumexpr(&root_hmatrix_32, &root_hmatrix_21);

    // auto tt = sumexpr.get_sh();

    // cout << tt.size() << endl;
    // auto t0 = tt[0];
    // auto t1 = tt[1];
    // cout << " h32 target size , offset " << endl;
    // cout << t0->get_target_cluster().get_size() << ',' << t0->get_target_cluster().get_offset() << endl;
    // cout << "h32 source size , offset " << endl;
    // cout << t0->get_source_cluster().get_size() << ',' << t0->get_source_cluster().get_offset() << endl;
    // cout << "h21 target size , offset " << endl;
    // cout << t0->get_target_cluster().get_size() << ',' << t0->get_target_cluster().get_offset() << endl;
    // cout << "h21 source size , offset " << endl;
    // cout << t0->get_source_cluster().get_size() << ',' << t0->get_source_cluster().get_offset() << endl;

    // cout << "________________________________" << endl;
    // cout << "test multiplication sumexpr vecteur " << endl;

    // std::vector<T> x(size_3, 1.0);

    // std::vector<T> y  = reference_dense_matrix * x;
    // std::vector<T> yy = sumexpr.prod(x);
    // double norm       = 0;
    // for (int k = 0; k < y.size(); ++k) {
    //     norm += (y[k] - yy[k]) * (y[k] - yy[k]);
    // }
    // norm = sqrt(norm);
    // cout << norm << endl;
    // cout << " ca a l'aire ok " << endl;

    // cout << "test sur la restriction" << endl;
    // auto &target_sons = t0->get_target_cluster().get_children();
    // auto &source_sons = t1->get_source_cluster().get_children();
    // for (int i = 0; i < target_sons.size(); ++i) {
    //     auto &ti = target_sons[i];
    //     for (int j = 0; j < source_sons.size(); ++j) {
    //         auto &sj = source_sons[j];
    //         cout << "target_sons size et offset " << ti->get_size() << ',' << ti->get_offset() << endl;
    //         cout << "source sons size et offset " << sj->get_size() << ',' << sj->get_offset() << endl;
    //         SumExpression<T, htool::underlying_type<T>> restriction = sumexpr.Restrict(ti->get_size(), ti->get_offset(), sj->get_size(), sj->get_offset());
    //         cout << "restrict : sh size: " << restriction.get_sh().size() << ", sr size " << restriction.get_sr().size() << endl;
    //         auto SH = restriction.get_sh();
    //         for (int k = 0; k < SH.size() / 2; ++k) {
    //             auto H  = SH[2 * k];
    //             auto &K = SH[2 * k + 1];
    //             cout << " H target-> " << H->get_target_cluster().get_size() << ',' << H->get_target_cluster().get_offset() << endl;
    //             cout << " H source->" << H->get_source_cluster().get_size() << ',' << H->get_source_cluster().get_offset() << endl;
    //             cout << "K target -> " << K->get_target_cluster().get_size() << ',' << K->get_target_cluster().get_offset() << endl;
    //             cout << "K source ->" << K->get_source_cluster().get_size() << ',' << K->get_source_cluster().get_offset() << endl;
    //         }
    //     }
    // }
    // // 4enfants
    // // SumExpression<T, htool::underlying_type<T>> restr = sumexpr.Restrict(100, 100, 0, 0);
    // // auto test                                         = restr.get_sh();
    // // cout << test.size() << endl;
    // // for (int k = 0; k < test.size() / 2; ++k) {
    // //     cout << "fils " << k << endl;
    // //     auto testk  = test[2 * k];
    // //     auto testkk = test[2 * k + 1];
    // //     cout << " offset target et source de A " << endl;
    // //     cout << testk->get_target_cluster().get_offset() << ',' << testk->get_source_cluster().get_offset() << endl;
    // //     cout << " offset target et source de B " << endl;
    // //     cout << testkk->get_target_cluster().get_offset() << ',' << testkk->get_source_cluster().get_offset() << endl;
    // //     // cout << " size target et source de A "<< endl;
    // //     // cout << testk->get_target_cluster().get_size() << ',' << testk->get_source_cluster().get_size() << endl;
    // //     // cout << " size target et source de B "<< endl;
    // //     // cout << testkk->get_target_cluster().get_size() << ',' << testkk->get_source_cluster().get_size() << endl;
    // // }
    // // cout << "-b-b-b--bbb-b-b-b-b-bb-b-b-b-bb-b-b-b-b-b-b-b-b" << endl;
    // // SumExpression<T, htool::underlying_type<T>> restr2 = restr.Restrict(50, 50, 0, 0);
    // // auto test2                                         = restr2.get_sh();
    // // cout << test2.size() << endl;
    // // for (int k = 0; k < test2.size() / 2; ++k) {
    // //     cout << "fils " << k << endl;
    // //     auto testk  = test2[2 * k];
    // //     auto testkk = test2[2 * k + 1];
    // //     cout << " offset target et source de A " << endl;
    // //     cout << testk->get_target_cluster().get_offset() << ',' << testk->get_source_cluster().get_offset() << endl;
    // //     cout << " offset target et source de B " << endl;
    // //     cout << testkk->get_target_cluster().get_offset() << ',' << testkk->get_source_cluster().get_offset() << endl;
    // //     // cout << " size target et source de A "<< endl;
    // //     // cout << testk->get_target_cluster().get_size() << ',' << testk->get_source_cluster().get_size() << endl;
    // //     // cout << " size target et source de B "<< endl;
    // //     // cout << testkk->get_target_cluster().get_size() << ',' << testkk->get_source_cluster().get_size() << endl;
    // // }
    // // ------------------------> Ca à l'aire de marcher
    // // auto &child = root_hmatrix_21.get_children();
    // // for (int k = 0; k < child.size(); ++k) {
    // //     auto &child_k                                         = child[k];
    // //     auto &t                                               = child_k->get_target_cluster();
    // //     auto &s                                               = child_k->get_source_cluster();
    // //     SumExpression<T, htool::underlying_type<T>> sum_restr = sumexpr.Restrict(t.get_size(), s.get_size(), t.get_offset(), s.get_offset());
    // //     cout << "____________________________" << endl;
    // //     cout << sumexpr.get_sr().size() << ',' << sumexpr.get_sh().size() << endl;
    // //     cout << sum_restr.get_sr().size() << ',' << sum_restr.get_sh().size() << endl;
    // // }
    // cout << "test evaluate" << endl;
    // // // normalement si je lui donne sumexpr = 32*21 je devrais tomber sur le produit

    // Matrix<T> eval = sumexpr.Evaluate();
    // cout << normFrob(eval - reference_dense_matrix) << endl;

    // // // // C'est vraimùent pas des tests incroyables mais pour l'instant tout marche
    // // // // Par contre on a pas pus tester la composante sr , mais bon techniquement c'est des matrices donc il devrait pas il y avoir de pb

    // cout << "hmult" << endl;
    // // shared_ptr<const Cluster<double>> clust1;
    // // clust1 = make_shared<const Cluster<double>>(&root_hmatrix_32.get_target_cluster());
    // // shared_ptr<const Cluster<double>> clust2;
    // // clust2 = make_shared<const Cluster<double>>(&root_hmatrix_21.get_target_cluster());
    // // HMatrix<T, htool::underlying_type<T>> L(root_hmatrix_32.get_root_target(), root_hmatrix_32.get_root_source());
    // // sumexpr.Hmult(&L);
    // HMatrix<T, htool::underlying_type<T>> L = root_hmatrix_32.hmatrix_product(root_hmatrix_21);
    // cout << "hmult ok" << endl;
    // // vector<T> xxx ( L.get_source_cluster().get_size() , 1);
    // // vector<T> yyy ( L.get_target_cluster().get_size(),0);
    // // L.add_vector_product('N',1.0, xxx.data(), 0.0, yyy.data());
    // // cout << norm2 ( yyy-reference_dense_matrix*xxx ) << endl;
    // Matrix<T> L_hmult(L.get_target_cluster().get_size(), L.get_source_cluster().get_size());
    // // cout << "hmult ok" << endl;
    // // auto testmult = L.get_leaves();
    // // auto testleav = root_hmatrix_21.get_leaves();
    // // std::cout << testleav.size() << std::endl;
    // // cout << testmult.size() << endl;
    // // // cout << "leaves ok" << endl;
    // // // vector<T> xxx(L.get_source_cluster().get_size(), 1);
    // // // vector<T> yyy(L.get_target_cluster().get_size(), 0);
    // // // L.add_vector_product('N', 1.0, xxx.data(), 0.0, yyy.data());
    // // // cout << "matrice vecteur " << endl;
    // // // cout << norm2(yyy - reference_dense_matrix * xxx) << endl;
    // copy_to_dense(L, L_hmult.data());
    // cout << "erreur Hmult" << endl;
    // cout << normFrob(L_hmult - reference_dense_matrix) << endl;
    // cout << normFrob(reference_dense_matrix) << endl;
    return is_error;
}
