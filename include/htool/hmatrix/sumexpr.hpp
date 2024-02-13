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

    std::vector<const LRtype *> Sr;
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

    Sumexpr(std::vector<const HMatrixType *> sh, std::vector<const LRtype *> sr, std::vector<int> oft, std::vector<int> ofs, bool flag) {
        Sh               = sh;
        Sr               = sr;
        auto &A          = sh[0];
        auto &B          = sh[1];
        target_size      = A->get_target_cluster().get_size();
        target_offset    = A->get_target_cluster().get_offset();
        source_size      = B->get_source_cluster().get_size();
        source_offset    = B->get_source_cluster().get_offset();
        target_offset_lr = oft;
        source_offset_lr = ofs;
        restrectible     = flag;
    }
    bool is_restrectible() const {
        return restrectible;
    }
    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        Matrix<CoefficientPrecision> res(M, N);
        res.assign(M, N, ptr, false);
        for (int k = 0; k < Sr.size(); ++k) {
            int oft  = target_offset_lr[k];
            int ofs  = source_offset_lr[k];
            auto lrk = Sr[k];
            lrk->Get_U().get_block(M, lrk->rank_of(), row_offset - oft, 0).add_matrix_product('N', 1.0, lrk->Get_V().get_block(lrk->rank_of(), N, 0, col_offset - ofs).data(), 1.0, res.data(), N);
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
            } else {
                auto Arestr = Adense.get_block(M, A->get_source_cluster().get_size(), row_offset - A->get_target_cluster().get_offset(), 0);
                auto Brestr = Bdense.get_block(B->get_target_cluster().get_size(), N, 0, col_offset - B->get_source_cluster().get_offset());
                Arestr.add_matrix_product('N', 1.0, Brestr.data(), 1.0, res.data(), N);
            }
        }
    }

    Sumexpr restrict(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) const {
        int size_t   = t.get_size();
        int size_s   = s.get_size();
        int offset_t = t.get_offset();
        int offset_s = s.get_offset();
        // on retreint Sr:
        auto sr = Sr;
        if (sr.size() > 0) {
            auto llr = sr[0];
            std::cout << llr->nb_cols() << ',' << llr->nb_rows() << ',' << llr->rank_of() << std::endl;
        }
        bool flag = true;
        auto oft  = target_offset_lr;
        auto ofs  = source_offset_lr;
        std::vector<const HMatrixType *> sh;
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto A = Sh[2 * k];
            auto B = Sh[2 * k + 1];
            for (auto &a_child : A->get_children()) {
                if (a_child->get_target_cluster().get_size() == size_t) {
                    auto &lrgen = *a_child->get_low_rank_generator().get();

                    auto b_child = B->get_son(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
                    // on regarde si le bon sous bloque existe dans B
                    // if (b_child->get_target_cluster().get_size() == a_child->get_source_cluster().get_size() && b_child->get_source_cluster().get_size() == size_s) {
                    if (b_child != nullptr) {
                        // c'est la bonne taille , on regarde si il y en a un low rank
                        if (a_child->is_low_rank() or b_child->is_low_rank()) {
                            Matrix<CoefficientPrecision> u, v;
                            if (a_child->is_low_rank() && b_child->is_low_rank()) {
                                u = a_child->get_low_rank_data()->Get_U();
                                v = a_child->get_low_rank_data()->Get_V() * b_child->get_low_rank_data()->Get_U() * b_child->get_low_rank_data()->Get_V();
                                // auto &lrgen = *a_child->get_low_rank_generator().get();
                                // lr          = lr.actualise(u, v, lrgen, a_child->get_target_cluster(), b_child->get_source_cluster());
                            } else if (a_child->is_low_rank() and !b_child->is_low_rank()) {
                                u = a_child->get_low_rank_data()->Get_U();
                                v = b_child->matrix_hmatrix(a_child->get_low_rank_data()->Get_V());

                            } else if (!a_child->is_low_rank() && b_child->is_low_rank()) {
                                u = a_child->hmatrix_matrix(b_child->get_low_rank_data()->Get_U());
                                v = b_child->get_low_rank_data()->Get_V();
                            }
                            LRtype lrtemp(u, v);
                            sr.push_back(&lrtemp);
                            std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;

                            oft.push_back(target_offset);
                            ofs.push_back(source_offset);

                        } // aucune n'est low rank
                        else {
                            sh.push_back(a_child.get());
                            sh.push_back(b_child);
                            if (a_child->get_children().size() == 0 or b_child->get_children().size()) {
                                flag = false;
                            }
                        }
                    } else {
                        /////////-----------NORMALEMENT IL DEVRAIT PAS IL Y AVOIR D'AUTRES CAS , mais il se passe des trucs bizzard du style un bloc avec juste 2 fils
                        // B n'a pas le bon bloc par exemple A = : : et b = :
                        // je vois pas de bonne solution
                        if (B->get_children().size() == 0) {
                            // B est un gros bloc dense -> ca devrait pas être possible
                            auto btemp = B->get_dense_data()->get_block(a_child->get_target_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset() - B->get_target_cluster().get_offset(), offset_s - B->get_source_cluster().get_offset());
                            if (a_child->is_low_rank()) {
                                auto u = a_child->get_low_rank_data()->Get_U();
                                auto v = a_child->get_low_rank_data()->Get_V() * btemp;

                                LRtype lrtemp(u, v);
                                sr.push_back(&lrtemp);
                                std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;

                                oft.push_back(target_offset);
                                ofs.push_back(source_offset);

                            } else {
                                // std::shared_ptr<Cluster<CoordinatePrecision>> shared_target = std::make_shared<Cluster<CoordinatePrecision>>(a_child->get_source_cluster());
                                // std::shared_ptr<Cluster<CoordinatePrecision>> shared_source = std::make_shared<Cluster<CoordinatePrecision>>(s);
                                // HMatrixType bb(shared_target, shared_source);
                                // bb.set_dense_data(btemp);
                                // bb.set_low_rank_generator(a_child->get_low_rank_generator());
                                // bb.set_admissibility_condition(a_child->get_admissibility_condition());
                                // bb.set_epsilon(a_child->get_epsilon());
                                // sh.push_back(a_child.get());
                                // sh.push_back(&bb);
                            }
                        } else {
                            auto btemp = B->get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
                            // normalement lui nous donne le plus gros blocs qui contient le bloc cible
                            if (btemp->is_low_rank()) {
                                auto u = a_child->hmatrix_matrix(btemp->get_low_rank_data()->Get_U().get_block(a_child->get_source_cluster().get_size(), btemp->get_low_rank_data()->rank_of(), a_child->get_source_cluster().get_offset(), 0));
                                auto v = btemp->get_low_rank_data()->Get_V().get_block(btemp->get_low_rank_data()->rank_of(), size_s, 0, offset_s - btemp->get_source_cluster().get_offset());

                                LRtype lrtemp(u, v);
                                sr.push_back(&lrtemp);
                                std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;

                                oft.push_back(target_offset);
                                ofs.push_back(source_offset);

                            } else {
                                Matrix<CoefficientPrecision> bdense(btemp->get_target_cluster().get_size(), btemp->get_source_cluster().get_size());
                                copy_to_dense(*btemp, bdense.data());
                                auto brestr = bdense.get_block(a_child->get_source_cluster().get_size(), size_s, a_child->get_source_cluster().get_offset(), offset_s);
                                if (a_child->is_low_rank()) {
                                    auto u = a_child->get_low_rank_data()->Get_U();
                                    auto v = a_child->get_low_rank_data()->Get_V() * brestr;

                                    LRtype lrtemp(u, v);
                                    sr.push_back(&lrtemp);
                                    std::cout << lrtemp.nb_rows() << ',' << lrtemp.nb_cols() << ',' << lrtemp.rank_of() << std::endl;
                                    oft.push_back(target_offset);
                                    ofs.push_back(source_offset);

                                } else {
                                    // std::shared_ptr<Cluster<CoordinatePrecision>> shared_target = std::make_shared<Cluster<CoordinatePrecision>>(a_child->get_source_cluster());
                                    // std::shared_ptr<Cluster<CoordinatePrecision>> shared_source = std::make_shared<Cluster<CoordinatePrecision>>(s);
                                    // HMatrixType bb(shared_target, shared_source);
                                    // bb.set_dense_data(brestr);
                                    // bb.set_low_rank_generator(a_child->get_low_rank_generator());
                                    // bb.set_admissibility_condition(a_child->get_admissibility_condition());
                                    // bb.set_epsilon(a_child->get_epsilon());
                                    // sh.push_back(a_child.get());
                                    // sh.push_back(&bb);
                                }
                            }
                        }
                    }
                }
            }
        }
        Sumexpr res(sh, sr, oft, ofs, flag);
        return res;
    }
};

} // namespace htool

#endif
