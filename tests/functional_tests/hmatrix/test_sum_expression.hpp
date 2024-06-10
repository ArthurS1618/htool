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

void assign(const Matrix<double> &M, const int &offset_target, const int &offset_source, HMatrix<double, double> *H_format) {
    if (H_format->get_target_cluster().get_children().size() == 0) {
        Matrix<double> M_restr(H_format->get_target_cluster().get_size(), H_format->get_source_cluster().get_size());
        for (int k = 0; k < H_format->get_target_cluster().get_size(); ++k) {
            for (int l = 0; l < H_format->get_source_cluster().get_size(); ++l) {
                M_restr(k, l) = M(k + H_format->get_target_cluster().get_offset() - offset_target, l + H_format->get_source_cluster().get_offset() - offset_source);
            }
        }
        //  = M.get_block(H_format->get_target_cluster().get_size(), H_format->get_source_cluster().get_size(), H_format->get_target_cluster().get_offset() - offset_target, H_format->get_source_cluster().get_offset() - offset_source);
        H_format->set_dense_data(M_restr);
    } else {
        for (auto &t_child : H_format->get_target_cluster().get_children()) {
            for (auto &s_child : H_format->get_source_cluster().get_children()) {
                auto H_child = H_format->add_child(t_child.get(), s_child.get());
                // if (H_child->get_admissibility_condition()->ComputeAdmissibility(*t_child, *s_child, H_child->get_eta())) {
                // }

                assign(M, offset_target, offset_source, H_child);
            }
        }
    }
}

// H = H +alpha K
// void plus_egal(const double &alpha, HMatrix<double, double> *H, HMatrix<double, double> *K, HMatrix<double, double> *res) {
//     if (H->get_children().size() > 0) {
//         for (auto &H_child : H->get_children()) {
//             if (K->get_children().size() > 0) {
//                 auto K_child   = K->get_block(H->get_target_cluster().get_size(), H_child->get_source_cluster().get_size(), H_child->get_target_cluster().get_offset(), H_child->get_source_cluster().get_offset());
//                 auto res_child = res->add_child(H_child->get_target_cluster(), H_child->get_source_cluster());
//                 plus_egal(alpha, H_child.get(), K_child, res_child);
//             } else {
//                 // ca ou les arbres sont pas allignés normalement c'est pas possible
//                 if (K->is_low_rank()) {
//                     Matrix<double> hdense(H_child->get_target_cluster().get_size(), H_child->get_source_cluster().get_size());
//                     copy_to_dense(*H_child, hdense.data());
//                     int nr      = hdense.nb_rows();
//                     int nc      = hdense.nb_cols();
//                     int rk      = K->get_low_rank_data()->Get_U().nb_cols();
//                     double beta = 1.0;
//                     Blas<double>::gemm("N", "N", &nr, &nc, &rk, &alpha, K->get_low_rank_data()->Get_U().data(), &nr, K->get_low_rank_data()->Get_U().data(), &nr, &beta, hdense.data(), &nr);
//                     H_child->delete_children();
//                     assign(hdense, H_child->get_target_cluster().get_offset(), H_child->get_source_cluster().get_offset(), H_child.get());
//                 } else {
//                     std::cout << "error case : dense block must be at most minblocksize" << std::endl;
//                 }
//             }
//         }
//     } else {
//         if (H->is_low_rank()) {
//             if (K->is_low_rank()) {
//                 auto data_u = K->get_low_rank_data()->Get_U().data();
//                 Matrix<double> U_update(K->get_low_rank_data()->Get_U().nb_rows(), K->get_low_rank_data()->Get_U().nb_cols());
//                 int size = K->get_low_rank_data()->Get_U().nb_rows();
//                 std::vector<double> temp(size);
//                 int inc = 1;
//                 Blas<double>::axpy(&size, &alpha, data_u, &inc, U_update.data(), &inc);
//                 SumExpression_fast<double, double> s_intermediaire;
//                 std::vector<Matrix<double>> data_lr(4);
//                 data_lr[0] = H->get_low_rank_data()->Get_U();
//                 data_lr[1] = H->get_low_rank_data()->Get_V();
//                 data_lr[2] = U_update;
//                 data_lr[3] = K->get_low_rank_data()->Get_V();
//                 s_intermediaire.set_sr(data_lr, H->get_target_cluster().get_offset(), K->get_source_cluster().get_offset());
//                 LowRankMatrix<double, double> lr_bourrin(s_intermediaire, *H->get_low_rank_generator(), H->get_target_cluster(), H->get_source_cluster(), -1, H->get_epsilon());
//                 if (lr_bourrin.Get_U().nb_cols() > 0) {
//                     res->set_low_rank_data(lr_bourrin);
//                 } else {
//                     Matrix<double> temp(H->get_target_cluster().gets_size(), )
//                 }
//             }
//         }
//     }
// }
void format(const HMatrix<double, double> *H, HMatrix<double, double> *H_format) {
    if (H->get_children().size() == 0) {
        if (H->is_low_rank()) {
            H_format->set_low_rank_data(*H->get_low_rank_data());
        } else {
            Matrix<double> dense = *H->get_dense_data();
            assign(dense, H->get_target_cluster().get_offset(), H->get_source_cluster().get_offset(), H_format);
        }
    } else {
        for (auto &H_child : H->get_children()) {
            auto H_format_child = H_format->add_child(&H_child->get_target_cluster(), &H_child->get_source_cluster());
            format(H_child.get(), H_format_child);
        }
    }
}

void hmat_hmat(SumExpression_fast<double, double> &sumexpr, HMatrix<double, double> *res) {
    bool flag_adm = res->get_admissibility_condition()->ComputeAdmissibility(res->get_target_cluster(), res->get_source_cluster(), res->get_eta());
    if (res->get_target_cluster().get_children().size() > 0) {
        if (flag_adm) {
            LowRankMatrix<double, double> lr(sumexpr, *res->get_low_rank_generator(), res->get_target_cluster(), res->get_source_cluster(), -1, res->get_epsilon());
            if ((lr.rank_of() == -1)) {
                if (res->get_target_cluster().get_children().size() > 0) {
                    for (auto &t_child : res->get_target_cluster().get_children()) {
                        for (auto &s_child : res->get_source_cluster().get_children()) {
                            // std::cout << "restrict 1" << std::endl;
                            bool test    = true;
                            auto s_restr = sumexpr.restrict_ACA(*t_child, *s_child, test);
                            // auto s_restr = sumexpr.Restrict(*t_child, *s_child, test);

                            // std::cout << "restrict ok" << std::endl;
                            auto res_child = res->add_child(t_child.get(), s_child.get());
                            if (test == true) {
                                hmat_hmat(s_restr, res_child);
                            } else {
                                Matrix<double> dense = sumexpr.evaluate();
                                res->delete_children();
                                assign(dense, sumexpr.get_target_offset(), sumexpr.get_source_offset(), res);
                            }
                        }
                    }
                } else {
                    // std::cout << " dense hmat 1 " << std::endl;

                    Matrix<double> dense = sumexpr.evaluate();
                    res->set_dense_data(dense);
                    // std::cout << " dense hmat 1     ok " << std::endl;
                }
            } else {
                // std::cout << "affect lr hmat  " << std::endl;

                res->set_low_rank_data(lr);
                // std::cout << "lr push " << normFrob(lr.Get_U() * lr.Get_V()) << std::endl;
                // std::cout << "--------------->affect kclr hmat  ok" << std::endl;
            }
        } else {
            if (res->get_target_cluster().get_children().size() == 0) {
                // std::cout << "affect dense  hmkkat " << std::endl;

                Matrix<double> dense = sumexpr.evaluate();
                res->set_dense_data(dense);
                // std::cout << "dense push " << normFrob(dense) << ',' << res->get_target_cluster().get_size() << ',' << res->get_source_cluster().get_size() << '|' << dense.nb_rows() << ',' << dense.nb_cols() << std::endl;

                // std::cout << "--------------k-k--->affect dense  hmat ok" << std::endl;
            } else {
                for (auto &t_child : res->get_target_cluster().get_children()) {
                    for (auto &s_child : res->get_source_cluster().get_children()) {
                        auto res_child = res->add_child(t_child.get(), s_child.get());
                        // std::cout << "restrict 2   :" << t_child->get_size() << ',' << s_child->get_size() << std::endl;
                        bool test             = true;
                        auto sumexpr_restrict = sumexpr.restrict_ACA(*t_child, *s_child, test);
                        // auto s_restr = sumexpr.Restrict(*t_child, *s_child, test);

                        // std::cout << "restrict ok" << std::endl;
                        if (test == true) {
                            hmat_hmat(sumexpr_restrict, res_child);
                        } else {
                            Matrix<double> dense = sumexpr.evaluate();
                            res->delete_children();
                            assign(dense, sumexpr.get_target_offset(), sumexpr.get_source_offset(), res);
                        }
                    }
                }
            }
        }
    } else {
        Matrix<double> dense = sumexpr.evaluate();
        res->delete_children();
        assign(dense, sumexpr.get_target_offset(), sumexpr.get_source_offset(), res);
    }
}

// std::vector<double> hmat_forward_vector(HMatrix<double, double> *hmat, std::vector<double> rightside) {
//     std::vector<double> leftside(hmat->get_source_cluster().get_size());
//     f
// }

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

    // auto A1 = generate_random_matrix(size, rank);
    // auto B1 = generate_random_matrix(rank, size);
    // auto A2 = generate_random_matrix(size, rank);
    // auto B2 = generate_random_matrix(rank, size);
    // // for (int k = 0; k < size; ++k) {
    // //     for (int l = 0; l < rank - 1; ++l) {
    // //         std::cout << A1(k, l) << ',';
    // //     }
    // //     std::cout << A1(k, rank - 1) << std::endl;
    // // }
    // Matrix<double> smart(rank, rank);
    // for (int k = 0; k < rank; ++k) {
    //     smart(k, k) = 1.0 / std::pow(2.718, k);
    // }
    // A1       = A1 * smart;
    // A2       = A2 * smart;
    // B1       = smart * B1;
    // B2       = smart * B2;
    // auto ref = A1 * B1 + A2 * B2;
    // // LowRankMatrix<T, T> lr(A1, B1);
    // // auto res = lr.compute_lr_update(A2, B2, -1, epsilon);
    // auto res = truncation(A1, B1, A2, B2, epsilon);
    // if (res.size() > 1) {
    //     std::cout << "ref : " << ref.nb_rows() << ',' << ref.nb_cols() << std::endl;
    //     std::cout << "U : " << res[0].nb_rows() << ',' << res[0].nb_cols() << std::endl;
    //     std::cout << "V : " << res[1].nb_rows() << ',' << res[1].nb_cols() << std::endl;
    //     double error = normFrob(ref - res[0] * res[1]) / normFrob(ref);
    //     std::cout << "error : " << error << " avec epsilon = " << epsilon << std::endl;
    //     std::cout << "rank : " << res[0].nb_cols() << " au lieu de " << 2 * rank << std::endl;

    //     std ::cout << "_________________________________" << std::endl;
    //     std ::cout << "_________________________________" << std::endl;

    //     std ::cout << "_________________________________" << std::endl;
    //     LowRankMatrix<double, double> LR1(A1, B1);
    //     LowRankMatrix<double, double> LR2(A2, B2);
    //     bool flag    = false;
    //     auto test_lr = LR1.formatted_addition(LR2, 1e-6, flag);
    //     std::cout << normFrob(ref - (test_lr.Get_U() * test_lr.Get_V())) / normFrob(ref) << std::endl;

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

    // GeneratorDouble generator(3, size, size, p1, p1, root_cluster_1, root_cluster_1);
    Matrix<double> reference_num_htool(size, size);
    generator.copy_submatrix(size, size, 0, 0, reference_num_htool.data());
    HMatrixTreeBuilder<double, double> hmatrix_tree_builder(root_cluster_1, root_cluster_1, epsilon, 100, 'N', 'N');

    // build
    auto root_hmatrix = hmatrix_tree_builder.build(generator);
    Matrix<double> ref(size, size);
    copy_to_dense(root_hmatrix, ref.data());
    auto comprr = root_hmatrix.get_compression();
    std::cout << "----uncorsening----- " << std::endl;
    std::cout << "hmatrix compression : " << comprr << std::endl;

    HMatrix<double, double> hmatrix_uncorsed(root_cluster_1, root_cluster_1);
    hmatrix_uncorsed.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    hmatrix_uncorsed.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
    hmatrix_uncorsed.set_eta(root_hmatrix.get_eta());
    hmatrix_uncorsed.set_epsilon(root_hmatrix.get_epsilon());

    format(&root_hmatrix, &hmatrix_uncorsed);

    Matrix<double> hdense_corsed(size, size);
    copy_to_dense(hmatrix_uncorsed, hdense_corsed.data());
    std::cout << normFrob(ref - hdense_corsed) / normFrob(ref) << std::endl;

    std::cout << "nb leaves : " << root_hmatrix.get_leaves().size() << ',' << hmatrix_uncorsed.get_leaves().size() << std::endl;
    std::cout << "compr uncorsede : " << hmatrix_uncorsed.get_compression() << std::endl;

    SumExpression_fast<double, double> sum_expr(&hmatrix_uncorsed, &hmatrix_uncorsed);
    auto xrand0 = generate_random_vector(size);
    auto prod   = ref * ref;
    bool flag0  = true;
    auto y      = sum_expr.prod('N', xrand0);
    std::cout << "errreur vecteur : " << norm2(prod * xrand0 - y) / norm2(prod * xrand0) << std::endl;

    auto &t1   = root_cluster_1->get_children()[0];
    auto &t2   = root_cluster_1->get_children()[1];
    auto restr = sum_expr.restrict_ACA(*t1, *t2, flag0);
    std::cout << "size sh : " << restr.get_sh().size() << std::endl;
    std::cout << "size sr : " << restr.get_sr().size() << std::endl;
    auto &t11   = root_cluster_1->get_children()[0]->get_children()[0];
    auto &t22   = root_cluster_1->get_children()[1]->get_children()[1];
    auto restr1 = restr.restrict_ACA(*t11, *t22, flag0);
    std::cout << "size sh : " << restr1.get_sh().size() << std::endl;
    std::cout << "size sr : " << restr1.get_sr().size() << std::endl;
    auto &t111  = root_cluster_1->get_children()[0]->get_children()[0]->get_children()[0];
    auto &t222  = root_cluster_1->get_children()[1]->get_children()[1]->get_children()[1];
    auto restr2 = restr1.restrict_ACA(*t111, *t222, flag0);
    std::cout << "size sh : " << restr2.get_sh().size() << std::endl;
    std::cout << "size sr : " << restr2.get_sr().size() << std::endl;
    // auto shh    = restr2.get_sh();
    auto &t1111 = t111->get_children()[0];
    auto &t2222 = t222->get_children()[1];
    auto restr3 = restr2.restrict_ACA(*t1111, *t2222, flag0);
    std::cout << "size sh : " << restr3.get_sh().size() << std::endl;
    std::cout << "size sr : " << restr3.get_sr().size() << std::endl;
    // auto shh = restr2.get_sh();
    // std::cout << " sh ok : " << std::endl;
    // std::cout << "t11   " << t111->get_size() << ',' << t11->get_offset() << " fois  " << t22->get_size() << ',' << t22->get_offset() << std::endl;
    // for (int k = 0; k < shh.size() / 2; ++k) {
    //     auto hh = shh[2 * k];
    //     auto kk = shh[2 * k + 1];

    //     std::cout << '(' << hh->get_target_cluster().get_size() << ',' << hh->get_target_cluster().get_offset() << "),(" << hh->get_source_cluster().get_size() << ',' << hh->get_source_cluster().get_offset() << ')' << std::endl;
    //     std::cout << '(' << kk->get_target_cluster().get_size() << ',' << kk->get_target_cluster().get_offset() << "),(" << kk->get_source_cluster().get_size() << ',' << kk->get_source_cluster().get_offset() << ')' << std::endl;
    // }

    // Matrix<double> prod_restr(t111->get_size(), t222->get_size());
    // for (int k = 0; k < t111->get_size(); ++k) {
    //     for (int l = 0; l < t222->get_size(); ++l) {
    //         prod_restr(k, l) = prod(k + t111->get_offset(), l + t222->get_offset());
    //     }
    // }
    Matrix<double> prod_restr(t1111->get_size(), t2222->get_size());
    for (int k = 0; k < t1111->get_size(); ++k) {
        for (int l = 0; l < t2222->get_size(); ++l) {
            prod_restr(k, l) = prod(k + t1111->get_offset(), l + t2222->get_offset());
        }
    }
    auto xrand = generate_random_vector(t2222->get_size());
    auto yy    = prod_restr * xrand;

    auto yrestr = restr3.prod('N', xrand);

    std::cout << "errreur vecteur restr: " << norm2(yy - yrestr) / norm2(yy) << std::endl;
    std::cout << restr2.get_nr() << ',' << restr2.get_nc() << std::endl;
    std::cout << t1111->get_size() << ',' << t2222->get_size() << std::endl;
    std::cout << "evaluate " << std::endl;
    auto restr3_dense = restr3.evaluate();
    std::cout << normFrob(restr3_dense - prod_restr) / normFrob(prod_restr) << std::endl;
    // Matrix<double> dense12(t1.get_size(), t2.get_size());

    ////////////////////////////////
    SumExpression_fast<double, double> sum_expr_prod(&hmatrix_uncorsed, &hmatrix_uncorsed);

    HMatrix<double, double> test_prod(root_cluster_1, root_cluster_1);
    test_prod.set_admissibility_condition(root_hmatrix.get_admissibility_condition());
    test_prod.set_low_rank_generator(root_hmatrix.get_low_rank_generator());
    test_prod.set_eta(root_hmatrix.get_eta());
    test_prod.set_epsilon(root_hmatrix.get_epsilon());
    hmat_hmat(sum_expr_prod, &test_prod);
    std::cout << "hmat hmat done " << std::endl;
    std::cout << test_prod.get_target_cluster().get_size() << ',' << test_prod.get_source_cluster().get_size() << ',' << test_prod.get_leaves().size() << std::endl;
    Matrix<double> prod_dense(size, size);
    copy_to_dense(test_prod, prod_dense.data());
    std::cout << "copy done" << std::endl;
    std::cout << " erreur prod " << normFrob(prod_dense - prod) / normFrob(prod) << std::endl;
    std::cout << "get_compression : " << test_prod.get_compression() << std::endl;
    // }
    return 0;
}
