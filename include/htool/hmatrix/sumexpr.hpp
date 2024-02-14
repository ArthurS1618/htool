#ifndef SUMEXPR_HPP
#define SUMEXPR_HPP
#include "../basic_types/vector.hpp"
#include <random>

namespace htool {

template <class CoordinatePrecision>
class Cluster;
template <typename CoefficientPrecision, typename CoordinatePrecision>
class HMatrix;
template <typename CoefficientPrecision, typename CoordinatePrecision>
class LowRankMatrix;
template <typename CoefficientPrecision, typename CoordinatePrecision>
class Sumexpr : public VirtualGenerator<CoefficientPrecision> {
  private:
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    using LRtype      = LowRankMatrix<CoefficientPrecision, CoordinatePrecision>;

    // std::vector<const LRtype *> Sr;
    std::vector<Matrix<CoefficientPrecision>> UV;
    std::vector<const HMatrixType *> Sh;
    int target_size;
    int target_offset;
    int source_size;
    int source_offset;
    std::vector<int> target_offset_lr;
    std::vector<int> source_offset_lr;
    bool restrectible = true;

  public:
    Sumexpr(const HMatrixType *A, const HMatrixType *B) {
        Sh.push_back(A);
        Sh.push_back(B);
        target_size   = A->get_target_cluster().get_size();
        target_offset = A->get_target_cluster().get_offset();
        source_size   = B->get_source_cluster().get_size();
        source_offset = B->get_source_cluster().get_offset();
    }

    Sumexpr(const std::vector<const HMatrixType *> &sh, const std::vector<Matrix<CoefficientPrecision>> &uv, const std::vector<int> &oft, const std::vector<int> &ofs, const int &target_size_0, const int &source_size_0, const int &target_offset_0, const int &source_offset_0, const bool &flag) {
        Sh               = sh;
        UV               = uv;
        target_size      = target_size_0;
        target_offset    = target_offset_0;
        source_size      = source_size_0;
        source_offset    = source_offset_0;
        target_offset_lr = oft;
        source_offset_lr = ofs;
        restrectible     = flag;
    }
    bool is_restrectible() const {
        return restrectible;
    }

    const std::vector<Matrix<CoefficientPrecision>> get_sr() const { return UV; }
    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        // std::cout << this->target_size << ',' << this->source_size << ',' << this->target_offset << ',' << this->source_offset << std::endl;
        // std::cout << UV.size() << std::endl;
        Matrix<CoefficientPrecision> res(M, N);
        res.assign(M, N, ptr, false);
        for (int k = 0; k < UV.size() / 2; ++k) {
            int oft = target_offset_lr[k];
            int ofs = source_offset_lr[k];
            // auto lrk = Sr[k];
            auto u = UV[2 * k];
            auto v = UV[2 * k + 1];
            // std::cout << u.nb_rows() << ',' << u.nb_cols() << ',' << M << ',' << row_offset << ',' << oft << std::endl;
            auto uu = u.get_block(M, u.nb_cols(), row_offset - oft, 0);
            auto vv = v.get_block(v.nb_rows(), N, 0, col_offset - ofs);
            uu.add_matrix_product('N', 1.0, vv.data(), 1.0, res.data(), N);
        }

        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto &A = Sh[2 * k];
            auto &B = Sh[2 * k + 1];
            Matrix<CoefficientPrecision> Adense(A->get_target_cluster().get_size(), A->get_source_cluster().get_size());
            Matrix<CoefficientPrecision> Bdense(B->get_target_cluster().get_size(), B->get_source_cluster().get_size());
            copy_to_dense(*A, Adense.data());
            copy_to_dense(*B, Bdense.data());
            if (Adense.nb_rows() == M && Bdense.nb_cols() == N) {
                Adense.add_matrix_product('N', 1.0, Bdense.data(), 1.0, res.data(), N);
                // std::cout << "->" << normFrob(res) << std::endl;
            } else {
                auto Arestr = Adense.get_block(M, A->get_source_cluster().get_size(), row_offset - A->get_target_cluster().get_offset(), 0);
                auto Brestr = Bdense.get_block(B->get_target_cluster().get_size(), N, 0, col_offset - B->get_source_cluster().get_offset());
                Arestr.add_matrix_product('N', 1.0, Brestr.data(), 1.0, res.data(), N);
                // std::cout << "-->" << normFrob(res) << std::endl;
            }
        }
    }

    Sumexpr restrict(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
        int size_t   = t.get_size();
        int size_s   = s.get_size();
        int offset_t = t.get_offset();
        int offset_s = s.get_offset();
        auto uvrestr = UV;
        bool flag    = true;
        auto oft     = target_offset_lr;
        auto ofs     = source_offset_lr;
        std::vector<const HMatrixType *> sh;
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto &A = Sh[2 * k];
            auto &B = Sh[2 * k + 1];
            if (A->get_children().size() > 0) {
                int rep = 0;
                for (auto &a_child : A->get_children()) {
                    if (a_child->get_target_cluster().get_size() == size_t) {
                        rep += 1;
                        // std::cout << "!!!!" << std::endl;
                        auto b_child = B->get_son(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
                        if (b_child == nullptr) {
                            Matrix<CoefficientPrecision> bdense(B->get_target_cluster().get_size(), B->get_source_cluster().get_size());
                            copy_to_dense(*B, bdense.data());
                            Matrix<CoefficientPrecision> brestr = bdense.get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset() - B->get_target_cluster().get_offset(), offset_s - B->get_source_cluster().get_offset());
                            /// je sais pas comment faire des pettites hmatrices donc si l'autre est hmat on a pas le choix wue de la densifier aussi .. ( on doit stocker H et K pas juste HK)
                            if (a_child->is_low_rank()) {
                                auto u = a_child->get_low_rank_data()->Get_U();
                                auto v = a_child->get_low_rank_data()->Get_V() * brestr;
                                uvrestr.push_back(u);
                                uvrestr.push_back(v);
                                oft.push_back(offset_t);
                                ofs.push_back(offset_s);
                            } else {
                                Matrix<CoefficientPrecision> adense(size_t, a_child->get_source_cluster().get_size());
                                copy_to_dense(*a_child, adense.data());
                                // vraiment nulle ;
                                uvrestr.push_back(adense);
                                uvrestr.push_back(brestr);
                                oft.push_back(offset_t);
                                ofs.push_back(offset_s);
                            }
                        } else {
                            // si il y en a une des deux qui est low rank
                            if (a_child->is_low_rank() or b_child->is_low_rank()) {
                                Matrix<CoefficientPrecision> u;
                                Matrix<CoefficientPrecision> v;
                                if (a_child->is_low_rank() && !b_child->is_low_rank()) {
                                    u = a_child->get_low_rank_data()->Get_U();
                                    v = b_child->matrix_hmatrix(a_child->get_low_rank_data()->Get_V());
                                } else if (!a_child->is_low_rank() && b_child->is_low_rank()) {
                                    u = a_child->hmatrix_matrix(b_child->get_low_rank_data()->Get_U());
                                    v = b_child->get_low_rank_data()->Get_V();
                                } else if (a_child->is_low_rank() && b_child->is_low_rank()) {
                                    u = a_child->get_low_rank_data()->Get_U() * a_child->get_low_rank_data()->Get_V() * b_child->get_low_rank_data()->Get_U();
                                    v = b_child->get_low_rank_data()->Get_V();
                                }
                                uvrestr.push_back(u);
                                uvrestr.push_back(v);
                                oft.push_back(offset_t);
                                ofs.push_back(offset_s);
                            } else {
                                // aucune n'est low rank
                                sh.push_back(a_child.get());
                                sh.push_back(b_child);
                            }
                        }
                    }
                }
                if (rep == 0) {
                    std::cout << "problème de htool   : on demande t= " << t.get_size() << ',' << t.get_offset() << " a partir de At =" << A->get_target_cluster().get_size() << ',' << A->get_target_cluster().get_offset() << ',' << A->get_source_cluster().get_size() << ',' << A->get_source_cluster().get_offset() << std::endl;
                    for (auto &a_child : A->get_children()) {
                        std::cout << a_child->get_target_cluster().get_size() << ',' << a_child->get_target_cluster().get_offset() << ',' << a_child->get_source_cluster().get_size() << ',' << a_child->get_source_cluster().get_offset() << std::endl;
                    }
                }
            } else {
                // A et B peuvent pas être low rank sinon elle serait pas dans Sh donc A est dense-> ans tout les cas oun va devoir stocker dans UV parce que je peux pas faire de petites Hmat
                Matrix<CoefficientPrecision> adense(A->get_target_cluster().get_size(), A->get_source_cluster().get_size());
                copy_to_dense(*A, adense.data());
                if (B->get_children().size() > 0) {
                    int rep = 0;
                    for (auto &b_child : B->get_children()) {
                        if (b_child->get_source_cluster().get_size() == size_s) {
                            rep += 1;
                            auto arestr = adense.get_block(size_t, b_child->get_target_cluster().get_size(), offset_t - A->get_target_cluster().get_size(), b_child->get_target_cluster().get_offset() - A->get_source_cluster().get_offset());
                            if (b_child->is_low_rank()) {
                                auto u = arestr * b_child->get_low_rank_data()->Get_U();
                                auto v = b_child->get_low_rank_data()->Get_V();
                                uvrestr.push_back(u);
                                uvrestr.push_back(v);
                                oft.push_back(offset_t);
                                ofs.push_back(offset_s);
                            } else {
                                // pas de bonne solution...
                                Matrix<CoefficientPrecision> bdense(b_child->get_target_cluster().get_size(), b_child->get_source_cluster().get_size());
                                copy_to_dense(*B, bdense.data());
                                uvrestr.push_back(arestr);
                                uvrestr.push_back(bdense);
                                oft.push_back(offset_t);
                                ofs.push_back(offset_s);
                            }
                        }
                    }
                }
            }
        }
        Sumexpr res(sh, uvrestr, oft, ofs, size_t, size_s, offset_t, offset_s, flag);
        return res;
    }

    // Sumexpr restrict(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
    //     int size_t   = t.get_size();
    //     int size_s   = s.get_size();
    //     int offset_t = t.get_offset();
    //     int offset_s = s.get_offset();
    //     auto uvrestr = UV;
    //     bool flag    = true;
    //     auto oft     = target_offset_lr;
    //     auto ofs     = source_offset_lr;
    //     std::vector<const HMatrixType *> sh;
    //     for (int k = 0; k < Sh.size() / 2; ++k) {
    //         auto A = Sh[2 * k];
    //         auto B = Sh[2 * k + 1];
    //         for (auto &a_child : A->get_children()) {
    //             if (a_child->get_target_cluster().get_size() == size_t) {
    //                 // auto &lrgen = *a_child->get_low_rank_generator().get();

    //                 auto b_child = B->get_son(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
    //                 if (b_child != nullptr) {
    //                     // c'est la bonne taille , on regarde si il y en a un low rank
    //                     if (a_child->is_low_rank() or b_child->is_low_rank()) {
    //                         Matrix<CoefficientPrecision> u, v;
    //                         if (a_child->is_low_rank() && b_child->is_low_rank()) {
    //                             u = a_child->get_low_rank_data()->Get_U();
    //                             v = a_child->get_low_rank_data()->Get_V() * b_child->get_low_rank_data()->Get_U() * b_child->get_low_rank_data()->Get_V();
    //                         } else if (a_child->is_low_rank() and !b_child->is_low_rank()) {
    //                             u = a_child->get_low_rank_data()->Get_U();
    //                             v = b_child->mat_hmat(a_child->get_low_rank_data()->Get_V());
    //                         } else if (!a_child->is_low_rank() && b_child->is_low_rank()) {
    //                             u = a_child->hmatrix_matrix(b_child->get_low_rank_data()->Get_U());
    //                             v = b_child->get_low_rank_data()->Get_V();
    //                         }
    //                         std::cout << "uv de norme " << normFrob(u) << ',' << normFrob(v) << std::endl;
    //                         std::cout << "push dans Sr tau, sigma = " << size_t << ',' << offset_t << ',' << size_s << ',' << offset_s << std::endl;
    //                         uvrestr.push_back(u);
    //                         uvrestr.push_back(v);

    //                         oft.push_back(target_offset);
    //                         ofs.push_back(source_offset);

    //                     } // aucune n'est low rank
    //                     else {
    //                         sh.push_back(a_child.get());
    //                         sh.push_back(b_child);
    //                         if ((a_child->get_children().size() == 0) and (b_child->get_children().size() == 0)) {
    //                             flag = false;
    //                         }
    //                     }
    //                 } else {
    //                     /////////-----------NORMALEMENT IL DEVRAIT PAS IL Y AVOIR D'AUTRES CAS , mais il se passe des trucs bizzard du style un bloc avec juste 2 fils
    //                     // B n'a pas le bon bloc par exemple A = : : et b = :
    //                     // je vois pas de bonne solution
    //                     if (B->get_children().size() == 0) {
    //                         // B est un gros bloc dense -> ca devrait pas être possible
    //                         auto btemp = B->get_dense_data()->get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset() - B->get_target_cluster().get_offset(), offset_s - B->get_source_cluster().get_offset());
    //                         if (a_child->is_low_rank()) {
    //                             auto u = a_child->get_low_rank_data()->Get_U();
    //                             auto v = a_child->get_low_rank_data()->Get_V() * btemp;
    //                             std::cout << " on a push U et V de norme " << normFrob(u) << ',' << normFrob(v) << std::endl;
    //                             uvrestr.push_back(u);
    //                             uvrestr.push_back(v);
    //                             oft.push_back(target_offset);
    //                             ofs.push_back(source_offset);

    //                         } else {
    //                             // std::shared_ptr<Cluster<CoordinatePrecision>> shared_target = std::make_shared<Cluster<CoordinatePrecision>>(a_child->get_source_cluster());
    //                             // std::shared_ptr<Cluster<CoordinatePrecision>> shared_source = std::make_shared<Cluster<CoordinatePrecision>>(s);
    //                             // HMatrixType bb(shared_target, shared_source);
    //                             // bb.set_dense_data(btemp);
    //                             // bb.set_low_rank_generator(a_child->get_low_rank_generator());
    //                             // bb.set_admissibility_condition(a_child->get_admissibility_condition());
    //                             // bb.set_epsilon(a_child->get_epsilon());
    //                             // sh.push_back(a_child.get());
    //                             // sh.push_back(&bb);
    //                         }
    //                     } else {
    //                         auto btemp = B->get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
    //                         // normalement lui nous donne le plus gros blocs qui contient le bloc cible
    //                         if (btemp != nullptr) {
    //                             if (btemp->is_low_rank()) {
    //                                 auto u = a_child->hmatrix_matrix(btemp->get_low_rank_data()->Get_U().get_block(a_child->get_source_cluster().get_size(), btemp->get_low_rank_data()->rank_of(), a_child->get_source_cluster().get_offset(), 0));
    //                                 auto v = btemp->get_low_rank_data()->Get_V().get_block(btemp->get_low_rank_data()->rank_of(), size_s, 0, offset_s - btemp->get_source_cluster().get_offset());

    //                                 // LRtype lrtemp(u, v);
    //                                 // sr.push_back(&lrtemp);
    //                                 // std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;
    //                                 uvrestr.push_back(u);
    //                                 uvrestr.push_back(v);
    //                                 std::cout << "on push " << u.nb_rows() << ',' << u.nb_cols() << v.nb_rows() << ',' << v.nb_cols() << std::endl;
    //                                 oft.push_back(target_offset);
    //                                 ofs.push_back(source_offset);

    //                             } else {
    //                                 Matrix<CoefficientPrecision> bdense(btemp->get_target_cluster().get_size(), btemp->get_source_cluster().get_size());
    //                                 copy_to_dense(*btemp, bdense.data());
    //                                 auto brestr = bdense.get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
    //                                 if (a_child->is_low_rank()) {
    //                                     auto u = a_child->get_low_rank_data()->Get_U();
    //                                     auto v = a_child->get_low_rank_data()->Get_V() * brestr;
    //                                     std::cout << "on push " << u.nb_rows() << ',' << u.nb_cols() << v.nb_rows() << ',' << v.nb_cols() << std::endl;

    //                                     uvrestr.push_back(u);
    //                                     uvrestr.push_back(v);
    //                                     oft.push_back(target_offset);
    //                                     ofs.push_back(source_offset);
    //                                 }
    //                             }
    //                         } else {
    //                             // std::shared_ptr<Cluster<CoordinatePrecision>> shared_target = std::make_shared<Cluster<CoordinatePrecision>>(a_child->get_source_cluster());
    //                             // std::shared_ptr<Cluster<CoordinatePrecision>> shared_source = std::make_shared<Cluster<CoordinatePrecision>>(s);
    //                             // HMatrixType bb(shared_target, shared_source);
    //                             // bb.set_dense_data(brestr);
    //                             // bb.set_low_rank_generator(a_child->get_low_rank_generator());
    //                             // bb.set_admissibility_condition(a_child->get_admissibility_condition());
    //                             // bb.set_epsilon(a_child->get_epsilon());
    //                             // sh.push_back(a_child.get());
    //                             // sh.push_back(&bb);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     Sumexpr res(sh, uvrestr, oft, ofs, size_t, size_s, offset_t, offset_s, flag);
    //     return res;
    // }

    // Sumexpr restrict_1(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
    //     int size_t   = t.get_size();
    //     int size_s   = s.get_size();
    //     int offset_t = t.get_offset();
    //     int offset_s = s.get_offset();
    //     auto uvrestr = UV;
    //     bool flag    = true;
    //     auto oft     = target_offset_lr;
    //     auto ofs     = source_offset_lr;
    //     std::vector<const HMatrixType *> sh;
    //     for (int k = 0; k < Sh.size() / 2; ++k) {
    //         auto A = Sh[2 * k];
    //         auto B = Sh[2 * k + 1];
    //         // if (A->get_children().size() == 0) {
    //         //     auto adense = A->get_dense_data();
    //         //     if (B->get_children().size() > 0) {
    //         //         for (auto &b_child : B->get_children()) {
    //         //             if (b_child->get_source_cluster().get_size() == s.get_size()) {
    //         //                 auto arestr = adense->get_block(t.get_size(), b_child->get_target_cluster().get_size(), t.get_offset() - A->get_target_cluster().get_offset(), b_child->get_source_cluster().get_offset() - A->get_source_cluster().get_offset());
    //         //                 if (b_child->is_low_rank()) {
    //         //                     auto u = b_child->get_low_rank_data()->Get_U();
    //         //                     auto v = b_child->get_low_rank_data()->Get_V() * arestr;
    //         //                     uvrestr.push_back(u);
    //         //                     uvrestr.push_back(v);
    //         //                     // sr.push_back(&lrtemp);
    //         //                     // std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;

    //         //                     oft.push_back(target_offset);
    //         //                     ofs.push_back(source_offset);
    //         //                 }
    //         //                 else{
    //         //                     sh.push_back(aresr)
    //         //                 }
    //         //             }
    //         //         }
    //         //     }
    //         // }
    //         for (auto &a_child : A->get_children()) {
    //             if (a_child->get_target_cluster().get_size() == size_t) {
    //                 auto &lrgen = *a_child->get_low_rank_generator().get();

    //                 auto b_child = B->get_son(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
    //                 // on regarde si le bon sous bloque existe dans B
    //                 // if (b_child->get_target_cluster().get_size() == a_child->get_source_cluster().get_size() && b_child->get_source_cluster().get_size() == size_s) {
    //                 if (b_child != nullptr) {
    //                     // c'est la bonne taille , on regarde si il y en a un low rank
    //                     if (a_child->is_low_rank() or b_child->is_low_rank()) {
    //                         // std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //                         Matrix<CoefficientPrecision> u, v;
    //                         if (a_child->is_low_rank() && b_child->is_low_rank()) {
    //                             // u.assign(t.get_size(), a_child->get_low_rank_data()->rank_of(), a_child->get_low_rank_data()->Get_U().data(), false);
    //                             // v.assign(a_child->get_low_rank_data()->rank_of(), s.get_size(), (a_child->get_low_rank_data()->Get_V() * b_child->get_low_rank_data()->Get_U() * b_child->get_low_rank_data()->Get_V()).data(), false);
    //                             // v = a_child->get_low_rank_data()->Get_V() * b_child->get_low_rank_data()->Get_U() * b_child->get_low_rank_data()->Get_V();
    //                             // auto &lrgen = *a_child->get_low_rank_generator().get();
    //                             // lr          = lr.actualise(u, v, lrgen, a_child->get_target_cluster(), b_child->get_source_cluster());
    //                             u = a_child->get_low_rank_data()->Get_U();
    //                             v = a_child->get_low_rank_data()->Get_V() * b_child->get_low_rank_data()->Get_U() * b_child->get_low_rank_data()->Get_V();
    //                         } else if (a_child->is_low_rank() and !b_child->is_low_rank()) {
    //                             // u.assign(t.get_size(), a_child->get_low_rank_data()->rank_of(), a_child->get_low_rank_data()->Get_U().data(), false);
    //                             // v.assign(a_child->get_low_rank_data()->rank_of(), s.get_size(), b_child->matrix_hmatrix(a_child->get_low_rank_data()->Get_V()).data(), false);
    //                             u = a_child->get_low_rank_data()->Get_U();
    //                             v = b_child->mat_hmat(a_child->get_low_rank_data()->Get_V());

    //                             // v = b_child->matrix_hmatrix(a_child->get_low_rank_data()->Get_V());

    //                         } else if (!a_child->is_low_rank() && b_child->is_low_rank()) {
    //                             // u.assign(t.get_size(), b_child-.get_low_rank_data()->rank_of(), a_child->hmatrix_matrix(b_child->get_low_rank_data()->Get_U()).data(), false);
    //                             // v.assign(a_child->get_low_rank_data()->rank_of(), s.get_size(), b_child->get_low_rank_data()->Get_V().data(), false);
    //                             // u = a_child->hmatrix_matrix(b_child->get_low_rank_data()->Get_U());
    //                             // v = b_child->get_low_rank_data()->Get_V();
    //                             u = a_child->hmatrix_matrix(b_child->get_low_rank_data()->Get_U());
    //                             v = b_child->get_low_rank_data()->Get_V();
    //                         }
    //                         // LRtype lrtemp(u, v);
    //                         // std::cout << "on push lr :" << lrtemp.rank_of() << ',' << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << std::endl;
    //                         // std::cout << "on push " << u.nb_rows() << ',' << u.nb_cols() << v.nb_rows() << ',' << v.nb_cols() << std::endl;
    //                         std::cout << "uv de norme " << normFrob(u) << ',' << normFrob(v) << std::endl;
    //                         std::cout << "push dans Sr tau, sigma = " << size_t << ',' << off_t << ',' << source_ << ',' << source_offset << std::endl;
    //                         uvrestr.push_back(u);
    //                         uvrestr.push_back(v);
    //                         // sr.push_back(&lrtemp);
    //                         // std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;

    //                         oft.push_back(target_offset);
    //                         ofs.push_back(source_offset);

    //                     } // aucune n'est low rank
    //                     else {
    //                         sh.push_back(a_child.get());
    //                         sh.push_back(b_child);
    //                         if ((a_child->get_children().size() == 0) and (b_child->get_children().size() == 0)) {
    //                             // std::cout << a_child->get_target_cluster().get_size() << ',' << a_child->get_source_cluster().get_size() << ',' << b_child->get_target_cluster().get_size() << ',' << b_child->get_source_cluster().get_size() << std::endl;
    //                             // std::cout << "---------->" << a_child->get_children().size() << ',' << b_child->get_children().size() << std::endl;
    //                             // std::cout << a_child->get_target_cluster().get_size() << ',' << a_child->get_target_cluster().get_offset() << ',' << a_child->get_source_cluster().get_size() << ',' << a_child->get_source_cluster().get_offset() << std::endl;
    //                             // std::cout << b_child->get_target_cluster().get_size() << ',' << b_child->get_target_cluster().get_offset() << ',' << b_child->get_source_cluster().get_size() << ',' << b_child->get_source_cluster().get_offset() << std::endl;
    //                             // std::cout << a_child->is_dense() << ',' << a_child->is_low_rank() << std ::endl;
    //                             // std::cout << b_child->is_dense() << ',' << b_child->is_low_rank() << std ::endl;

    //                             flag = false;
    //                         }
    //                     }
    //                 } else {
    //                     /////////-----------NORMALEMENT IL DEVRAIT PAS IL Y AVOIR D'AUTRES CAS , mais il se passe des trucs bizzard du style un bloc avec juste 2 fils
    //                     // B n'a pas le bon bloc par exemple A = : : et b = :
    //                     // je vois pas de bonne solution
    //                     // std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //                     if (B->get_children().size() == 0) {
    //                         // B est un gros bloc dense -> ca devrait pas être possible
    //                         auto btemp = B->get_dense_data()->get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset() - B->get_target_cluster().get_offset(), offset_s - B->get_source_cluster().get_offset());
    //                         if (a_child->is_low_rank()) {
    //                             auto u = a_child->get_low_rank_data()->Get_U();
    //                             auto v = a_child->get_low_rank_data()->Get_V() * btemp;
    //                             std::cout << " on a push U et V de norme " << normFrob(u) << ',' << normFrob(v) << std::endl;

    //                             // LRtype lrtemp(u, v);
    //                             // sr.push_back(&lrtemp);
    //                             // std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;
    //                             uvrestr.push_back(u);
    //                             uvrestr.push_back(v);
    //                             // std::cout << "on push " << u.nb_rows() << ',' << u.nb_cols() << v.nb_rows() << ',' << v.nb_cols() << std::endl;
    //                             oft.push_back(target_offset);
    //                             ofs.push_back(source_offset);

    //                         } else {
    //                             // std::shared_ptr<Cluster<CoordinatePrecision>> shared_target = std::make_shared<Cluster<CoordinatePrecision>>(a_child->get_source_cluster());
    //                             // std::shared_ptr<Cluster<CoordinatePrecision>> shared_source = std::make_shared<Cluster<CoordinatePrecision>>(s);
    //                             // HMatrixType bb(shared_target, shared_source);
    //                             // bb.set_dense_data(btemp);
    //                             // bb.set_low_rank_generator(a_child->get_low_rank_generator());
    //                             // bb.set_admissibility_condition(a_child->get_admissibility_condition());
    //                             // bb.set_epsilon(a_child->get_epsilon());
    //                             // sh.push_back(a_child.get());
    //                             // sh.push_back(&bb);
    //                         }
    //                     } else {
    //                         auto btemp = B->get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
    //                         // normalement lui nous donne le plus gros blocs qui contient le bloc cible
    //                         if (btemp != nullptr) {
    //                             if (btemp->is_low_rank()) {
    //                                 auto u = a_child->hmatrix_matrix(btemp->get_low_rank_data()->Get_U().get_block(a_child->get_source_cluster().get_size(), btemp->get_low_rank_data()->rank_of(), a_child->get_source_cluster().get_offset(), 0));
    //                                 auto v = btemp->get_low_rank_data()->Get_V().get_block(btemp->get_low_rank_data()->rank_of(), size_s, 0, offset_s - btemp->get_source_cluster().get_offset());

    //                                 // LRtype lrtemp(u, v);
    //                                 // sr.push_back(&lrtemp);
    //                                 // std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;
    //                                 uvrestr.push_back(u);
    //                                 uvrestr.push_back(v);
    //                                 // std::cout << "on push " << u.nb_rows() << ',' << u.nb_cols() << v.nb_rows() << ',' << v.nb_cols() << std::endl;
    //                                 oft.push_back(target_offset);
    //                                 ofs.push_back(source_offset);

    //                             } else {
    //                                 Matrix<CoefficientPrecision> bdense(btemp->get_target_cluster().get_size(), btemp->get_source_cluster().get_size());
    //                                 copy_to_dense(*btemp, bdense.data());
    //                                 auto brestr = bdense.get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
    //                                 if (a_child->is_low_rank()) {
    //                                     auto u = a_child->get_low_rank_data()->Get_U();
    //                                     auto v = a_child->get_low_rank_data()->Get_V() * brestr;
    //                                     uvrestr.push_back(u);
    //                                     uvrestr.push_back(v);
    //                                     // std::cout << "on push " << u.nb_rows() << ',' << u.nb_cols() << ',' << v.nb_rows() << ',' << v.nb_cols() << std::endl;
    //                                     // LRtype lrtemp(u, v);
    //                                     // sr.push_back(&lrtemp);
    //                                     // std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;
    //                                     oft.push_back(target_offset);
    //                                     ofs.push_back(source_offset);
    //                                 }
    //                             }
    //                         } else {
    //                             // std::shared_ptr<Cluster<CoordinatePrecision>> shared_target = std::make_shared<Cluster<CoordinatePrecision>>(a_child->get_source_cluster());
    //                             // std::shared_ptr<Cluster<CoordinatePrecision>> shared_source = std::make_shared<Cluster<CoordinatePrecision>>(s);
    //                             // HMatrixType bb(shared_target, shared_source);
    //                             // bb.set_dense_data(brestr);
    //                             // bb.set_low_rank_generator(a_child->get_low_rank_generator());
    //                             // bb.set_admissibility_condition(a_child->get_admissibility_condition());
    //                             // bb.set_epsilon(a_child->get_epsilon());
    //                             // sh.push_back(a_child.get());
    //                             // sh.push_back(&bb);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     // std::cout << sh.size() << std::endl;
    //     // for (auto &llr : sr) {
    //     //     std::cout << "lr :   " << llr->rank_of() << std::endl;
    //     // }
    //     Sumexpr res(sh, uvrestr, oft, ofs, size_t, size_s, offset_t, offset_s, flag);
    //     // for (auto &llr : res.get_sr()) {
    //     //     std::cout << " a la fin de restrict   " << llr->rank_of() << ',' << llr->nb_rows() << ',' << llr->nb_cols() << std::endl;
    //     // }
    //     return res;
    // }
};

} // namespace htool

#endif
