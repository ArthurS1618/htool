#ifndef HTOOL_BLOCKS_SUM_EXPRESSIONS_HPP
#define HTOOL_BLOCKS_SUM_EXPRESSIONS_HPP
#include "../basic_types/vector.hpp"
namespace htool {
template <class CoordinatePrecision>
class Cluster;
template <typename CoefficientPrecision, typename CoordinatePrecision>
class HMatrix;
template <typename CoefficientPrecision, typename CoordinatePrecision>
class SumExpression : public VirtualGenerator<CoefficientPrecision> {
    // JE VAIS MODFIER CETTE CLASSE
  private:
    // std::vector < std::pair<Block<T> *, Block<T> *> SumExpressionR; --------> Pour moi il y a plus de bloc ddonc on va passer avec les hmat
    // std::vector < std::pair<Block<T> *, Block<T> *> SumExpressionH;
    // std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision>*> Sr; // Vecteur de pair de lw rank
    std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> Sr;
    std::vector<int> offset; // pour keep track des offset des lrmat  -> taille = 2*sr -> offset pour U et V
    // Pour les low ranks c'est plus facile de prendre juste des matrices mais sion j'aio pas les offsets.
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> Sh; // vecteur de pair de hmat
    int nr;
    int nc;

  public:
    // Constructeurr <HMatrix<CoefficientPrecision, CoordinatePrecision>> sr, sh ;HMatrix<CoefficientPrecision,
    SumExpression(HMatrix<CoefficientPrecision, CoordinatePrecision> *A, HMatrix<CoefficientPrecision, CoordinatePrecision> *B) {
        std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> sh;
        sh.push_back(A);
        sh.push_back(B);
        // sh[0] = *A;
        // sh[1] = *B;
        Sh = sh;
        std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> sr;
        Sr = sr;
        nr = A->get_target_cluster().get_size();
        nc = B->get_source_cluster().get_size();
    }
    explicit SumExpression(const std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> &sr, const std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &sh, const std::vector<int> &of) : Sr(sr), Sh(sh), offset(of) {
        if (Sh.size() >= Sr.size()) {
            nr = Sh[0]->get_target_cluster().get_size();
            nc = Sh[1]->get_source_cluster().get_size();
        } else {
            nr = Sr[0].nb_rows();
            nc = Sr[0].nb_cols();
        }
    }

    // ~SumExpression() {
    //       if (Sr.size()> 0 or  Sh.size()> 0)
    //           delete[] &Sr; delete[] &Sh ; delete[] &offset;
    //   }
    // Getters

    std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> get_sr() { return Sr; }
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> get_sh() { return Sh; }
    int get_nr() { return nr; }
    int get_nc() { return nc; }

    // Multiplication par un vecteur
    /*
        std::vector<CoefficientPrecision> prod(const std::vector<CoefficientPrecision> &x, std::vector<CoefficientPrecision> y) {
            // std::vector<CoefficientPrecision> y(this->nr, 0.0);
            for (int k = 0; k < Sr.size(); ++k) {
                y = y + Sr[k] * x;
            }
            for (int k = 0; k < Sh.size(); ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                std::vector<CoefficientPrecision> ytemp(K->get_target_cluster().get_size(), 0.0);
                K->add_vector_product('N', 1.0, x.data(), 0.0, ytemp.data());
                std::vector<CoefficientPrecision> yplus(H->get_target_cluster().get_size(), 0.0);
                H->add_vector_product('N', 1.0, ytemp.data(), 0.0, yplus.data());
                y = y + yplus;
            }
            return y;
        }
        */
    std::vector<CoefficientPrecision> prod(const std::vector<CoefficientPrecision> &x) {
        std::vector<CoefficientPrecision> y(nr, 0.0);
        for (int k = 0; k < Sr.size(); ++k) {
            // auto A = Sr[2*k]; auto B = Sr[2*k+1];
            auto A = Sr[k];
            // std::vector<CoefficientPrecision> ytemp(B.nb_cols(),0);
            // std::vector<CoefficientPrecision> yk(A.nb_cols(),0);

            // B->add_vector_product('N',1.0, x.data(), 0.0, ytemp.data());
            // A->add_vector_product('N', 1.0, ytemp.data(), 0.0, yk.data() );
            // y =  y+yk;
            y = y + A * x;
        }
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto H = Sh[2 * k];
            auto K = Sh[2 * k + 1];
            std::vector<CoefficientPrecision> ytemp(K->get_source_cluster().get_size(), 0.0);
            std::vector<CoefficientPrecision> yk(H->get_source_cluster().get_size(), 0.0);

            K->add_vector_product('N', 1.0, x.data(), 0.0, ytemp.data());
            H->add_vector_product('N', 1.0, ytemp.data(), 0.0, yk.data());
            y = y + yk;
        }
        return (y);
    }

    CoefficientPrecision get_coeff(const int &i, const int &j) {
        std::vector<CoefficientPrecision> xj(nc, 0.0);
        xj[j]                               = 1.0;
        std::vector<CoefficientPrecision> y = this->prod(xj);
        return y[i];
    }

    // Fonctions

    // Restrictions au block tau sigma -> caractérisé par leur taille et offset
    //  SumExpression Restrict( const int& size1, const int& size2 , const int& of1 , const int& of2){
    //    std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> ssr;
    //    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision>*> ssh;
    //    std::vector<int> of;
    //    // Pour les low rank il suffit de restreindre les lignes de U et les collones de V
    //    for (int k =0 ; k < Sr.size() ; ++k){
    //      //auto A = Sr[2*k] ; auto B= Sr[2*k+1];
    //      auto A = Sr[k];
    //      int ofa = offset[2*k] ; int ofb = of[2*k+1];
    //      // normalement si on se débrouille bien A et B sont forcemment low rank (quitte a appeler un tri)
    //      //auto lra = A->get_low_rank_data() ; auto lrb = B->get_low_rank_data();
    //      Matrix<CoefficientPrecision> Ua =  A.Get_U() ; Matrix<CoefficientPrecision> Va = A.Get_V() ;
    //      //Matrix<CoefficientPrecision> Ub = B.Get_U() ; Matrix<CoefficientPrecision> Vb = B.Get_V();
    //      // Maintenant on restreint les lignes de Ua et les colonnes de Vb
    //      Matrix<CoefficientPrecision> U_restr(size1,Ua.nb_cols());
    //      if ( (of1 >= ofa) and (size1 < Ua.nb_rows() + of1-ofa)){
    //        for (int i = 0 ; i < size1 ; ++i ){
    //          for (int j =0 ; j < Ua.nb_cols() ; ++j ){
    //            U_restr(i,j) = Ua ( i+of1-ofa , j) ;
    //          }
    //        }
    //      }
    //      Matrix<CoefficientPrecision> V_restr(Va.nb_rows(), size2);
    //      if ((of2>= ofb) and (size2 < Va.nb_cols() + of2-ofb)){
    //        for (int i =0 ; i < Va.nb_rows() ; ++i ){
    //          for ( int j =0 ; j< size2 ; ++j ){
    //            V_restr( i,j) = Va(i,j+of2-ofb );
    //          }
    //        }
    //    }
    //    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lu(U_restr,V_restr);
    //    //LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lv(Ub,V_restr);
    //    ssr.push_back(lu); of.push_back(of1) ; of.push_back(of2);
    //  }
    //  //pour les hmat
    //  for (int k =0 ; k < Sh.size()/2 ; ++k ){
    //    auto H = Sh[2*k]; auto K = Sh[2*k+1];
    //    if ( H->get_children().size() > K->get_children().size()){
    //      auto& hh = H->get_children() ;
    //      for (int i =0 ; i < hh.size() ; ++i ){
    //        auto& hk = *hh[i];
    //        //on regarde si le fils a le même target que la restr
    //        if ( (hk.get_target_cluster().get_offset() == of1 ) and ( hk.get_target_cluster().get_size() == size1)){
    //          auto& vv= K->get_children();
    //          for (int j = 0 ; j < vv.size() ; ++j ){
    //            auto& vj = *vv[j];
    //            // on regarde si le fils a le meme source que la restr
    //            if( ( vj.get_source_cluster().get_offset() == of2 ) and ( vj.get_source_cluster().get_size() == size2 ) ){
    //              //on rezagrde si on a le dsroit de multiplier les deux -> H.source=K.target
    //              if (( hk.get_source_cluster().get_size() == vj.get_target_cluster().get_size() ) and ( hk.get_source_cluster().get_offset() == vj.get_target_cluster().get_offset()) ){
    //                ssh.push_back(&hk);ssh.push_back(&vj);
    //              }
    //            }
    //        }
    //      }
    //    }
    //    }
    //    else{
    //      auto& vv = K->get_children() ;
    //      for (int i =0 ; i < vv.size() ; ++i ){
    //        auto& vj = *vv[i];
    //        //on regarde si le fils a le même target que la restr
    //        if ( (vj.get_source_cluster().get_offset() == of2 ) and ( vj.get_source_cluster().get_size() == size2)){
    //          auto& vv= H->get_children();
    //          for (int j = 0 ; j < vv.size() ; ++j ){
    //            auto& hk = *vv[j];
    //            // on regarde si le fils a le meme source que la restr
    //            if( ( hk.get_target_cluster().get_offset() == of1 ) and ( vj.get_target_cluster().get_size() == size1 ) ){
    //              //on rezagrde si on a le dsroit de multiplier les deux -> H.source=K.target
    //              if (( hk.get_source_cluster().get_size() == vj.get_target_cluster().get_size() ) and ( hk.get_source_cluster().get_offset() == vj.get_target_cluster().get_offset()) ){
    //                ssh.push_back(&hk);ssh.push_back(&vj);
    //              }
    //            }
    //        }
    //      }
    //    }
    //    }
    //  }
    //  SumExpression res(ssr,ssh, of);
    //  return res;
    //  }

    // SumExpression Restrict(const int &size1, const int &size2, const int &of1, const int &of2) const {
    //     // std::cout << "on rentre dan dans restrict avec target = " << size1 << ',' << of1 << " et source = " << size2 << ',' << of2 << std::endl;
    //     // std::cout << "sr " << Sr.size() << ',' << "Sh" << Sh.size() << std::endl;
    //     std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> ssr;
    //     std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> ssh;
    //     std::vector<int> of;
    //     // Pour les low rank il suffit de restreindre les lignes de U et les collones de V
    //     for (int k = 0; k < Sr.size(); ++k) {
    //         // auto A = Sr[2*k] ; auto B= Sr[2*k+1];
    //         auto A                          = Sr[k];
    //         int ofa                         = offset[2 * k];
    //         int ofb                         = offset[2 * k + 1];
    //         Matrix<CoefficientPrecision> Ua = A.Get_U();
    //         Matrix<CoefficientPrecision> Va = A.Get_V();
    //         // Maintenant on restreint les lignes de Ua et les colonnes de Vb
    //         Matrix<CoefficientPrecision> U_restr(size1, Ua.nb_cols());
    //         if ((of1 >= ofa) and (size1 < Ua.nb_rows() + of1 - ofa)) {
    //             for (int i = 0; i < size1; ++i) {
    //                 for (int j = 0; j < Ua.nb_cols(); ++j) {
    //                     U_restr(i, j) = Ua(i + of1 - ofa, j);
    //                 }
    //             }
    //         }
    //         Matrix<CoefficientPrecision> V_restr(Va.nb_rows(), size2);
    //         if ((of2 >= ofb) and (size2 < Va.nb_cols() + of2 - ofb)) {
    //             for (int i = 0; i < Va.nb_rows(); ++i) {
    //                 for (int j = 0; j < size2; ++j) {
    //                     V_restr(i, j) = Va(i, j + of2 - ofb);
    //                 }
    //             }
    //         }
    //         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lu(U_restr, V_restr);
    //         // LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lv(Ub,V_restr);
    //         ssr.push_back(lu);
    //         of.push_back(of1);
    //         of.push_back(of2);
    //     }
    //     // pour les hmat
    //     for (int rep = 0; rep < Sh.size() / 2; ++rep) {
    //         // std::cout << "tri des sh k = " << rep << std::endl;
    //         auto H = Sh[2 * rep];
    //         auto K = Sh[2 * rep + 1];
    //         // std::cout << "cluster H" << H->get_target_cluster().get_size() << ',' << H->get_target_cluster().get_offset() << ',' << H->get_source_cluster().get_size() << ',' << H->get_source_cluster().get_offset() << std::endl;
    //         // std::cout << "cluster K" << K->get_target_cluster().get_size() << ',' << K->get_target_cluster().get_offset() << ',' << K->get_source_cluster().get_size() << ',' << K->get_source_cluster().get_offset() << std::endl;
    //         //  if ((H->get_children().size() > 0) or (K->get_children().size() > 0)) {
    //         //      if (H->get_children().size() > K->get_children().size()) {
    //         //          auto &hh = H->get_children();
    //         //          for (int i = 0; i < hh.size(); ++i) {
    //         //              auto &hk = *hh[i];
    //         //              // on regarde si le fils a le même target que la restr
    //         //              if ((hk.get_target_cluster().get_offset() == of1) and (hk.get_target_cluster().get_size() == size1)) {
    //         //                  auto &vv = K->get_children();
    //         //                  for (int j = 0; j < vv.size(); ++j) {
    //         //                      auto &vj = *vv[j];
    //         //                      // on regarde si le fils a le meme source que la restr
    //         //                      if ((vj.get_source_cluster().get_offset() == of2) and (vj.get_source_cluster().get_size() == size2)) {
    //         //                          // on rezagrde si on a le dsroit de multiplier les deux -> H.source=K.target
    //         //                          if ((hk.get_source_cluster().get_size() == vj.get_target_cluster().get_size()) and (hk.get_source_cluster().get_offset() == vj.get_target_cluster().get_offset())) {
    //         //                              ssh.push_back(&hk);
    //         //                              ssh.push_back(&vj);
    //         //                          }
    //         //                      }
    //         //                  }
    //         //              }
    //         //          }
    //         //      } else {
    //         //          auto &vv = K->get_children();
    //         //          for (int i = 0; i < vv.size(); ++i) {
    //         //              auto &vj = *vv[i];
    //         //              // on regarde si le fils a le même target que la restr
    //         //              if ((vj.get_source_cluster().get_offset() == of2) and (vj.get_source_cluster().get_size() == size2)) {
    //         //                  auto &hh = H->get_children();
    //         //                  for (int j = 0; j < hh.size(); ++j) {
    //         //                      auto &hk = *hh[j];
    //         //                      // on regarde si le fils a le meme source que la restr
    //         //                      if ((hk.get_target_cluster().get_offset() == of1) and (vj.get_target_cluster().get_size() == size1)) {
    //         //                          // on rezagrde si on a le dsroit de multiplier les deux -> H.source=K.target
    //         //                          if ((hk.get_source_cluster().get_size() == vj.get_target_cluster().get_size()) and (hk.get_source_cluster().get_offset() == vj.get_target_cluster().get_offset())) {
    //         //                              ssh.push_back(&hk);
    //         //                              ssh.push_back(&vj);
    //         //                          }
    //         //                      }
    //         //                  }
    //         //              }
    //         //          }
    //         //      }
    //         //  }
    //         //  // les deux sont des feuilles
    //         //  else {
    //         //      Matrix<CoefficientPrecision> fh(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
    //         //      Matrix<CoefficientPrecision> fk(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
    //         //      copy_to_dense(*H, fh.data());
    //         //      copy_to_dense(*K, fk.data());
    //         //      // extraction si c'est les bon offset
    //         //      if ((of1 > H->get_target_cluster().get_offset()) and (of2 > K->get_source_cluster().get_offset()) and (of1 + size1 < H->get_target_cluster().get_size()) and (of2 + size2 < K->get_source_cluster().get_size())) {
    //         //          Matrix<CoefficientPrecision> Hr(size1, H->get_source_cluster().get_size());
    //         //          Matrix<CoefficientPrecision> Kr(K->get_target_cluster().get_size(), size2);
    //         //          for (int i = 0; i < size1; ++i) {
    //         //              for (int j = 0; j < H->get_source_cluster().get_size(); ++j) {
    //         //                  Hr(i, j) = fh(i + of1 - H->get_target_cluster().get_offset(), j);
    //         //              }
    //         //          }
    //         //          for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
    //         //              for (int j = 0; j < size2; ++j) {
    //         //                  Kr(i, j) = fk(i, j + of2 - K->get_source_cluster().get_offset());
    //         //              }
    //         //          }
    //         //          LowRankMatrix<CoefficientPrecision> lrk(Hr, Kr);
    //         //          ssr.push_back(lrk);
    //         //          of.push_back(of1);
    //         //          of.push_back(of2);
    //         //      }
    //         //  }

    //         // ondoit tester tout les cas :
    //         if ((H->get_children().size() > 0) and (K->get_children().size() > 0)) {
    //             // std::cout << " fils et fils" << std::endl;
    //             auto &H_child = H->get_children();
    //             auto &K_child = K->get_children();
    //             for (int k = 0; k < H->get_children().size(); ++k) {
    //                 auto &H_child_k = H_child[k];
    //                 // on test si child_ k a le bon target
    //                 if ((H_child_k->get_target_cluster().get_size() == size1) and (H_child_k->get_target_cluster().get_offset() == of1)) {
    //                     for (int l = 0; l < K_child.size(); ++l) {
    //                         auto &K_child_l = K_child[l];
    //                         // on test si le source de K_k = target K_l
    //                         if ((K_child_l->get_target_cluster().get_size() == H_child_k->get_source_cluster().get_size()) and (K_child_l->get_target_cluster().get_offset() == H_child_k->get_source_cluster().get_offset())) {
    //                             // on test si k_l a le bon source
    //                             if ((K_child_l->get_source_cluster().get_size() == size2) and (K_child_l->get_source_cluster().get_offset() == of2)) {
    //                                 ssh.push_back(H_child_k.get());
    //                                 ssh.push_back(K_child_l.get());
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         } else if (H->get_children().size() > 0) {
    //             if (K->is_low_rank()) {
    //                 // std::cout << " fils * lr " << std::endl;
    //                 //  normalement il devrait jamais il y avoir H mat* dense
    //                 auto K_lr                      = K->get_low_rank_data();
    //                 Matrix<CoefficientPrecision> U = K_lr->Get_U();
    //                 Matrix<CoefficientPrecision> V = K_lr->Get_V();
    //                 Matrix<CoefficientPrecision> V_restr(V.nb_rows(), size2);
    //                 for (int i = 0; i < V.nb_rows(); ++i) {
    //                     for (int j = 0; j < size2; ++j) {
    //                         V_restr(i, j) = V(i, j + of2 - K->get_source_cluster().get_offset());
    //                     }
    //                 }
    //                 // std::cout << "V_restr ok " << std::endl;
    //                 auto &childs = H->get_children();
    //                 for (int k = 0; k < H->get_children().size(); ++k) {
    //                     auto &child_k = childs[k];
    //                     // std::cout << " child " << k << "ok " << std::endl;
    //                     if ((child_k->get_target_cluster().get_size() == size1)
    //                         and (child_k->get_target_cluster().get_offset() == of1)) {
    //                         // On doit faire H mat colonnes de U pour construire H_k U_restr -> on fait la restriction drect
    //                         Matrix<CoefficientPrecision> HU(size1, U.nb_cols());
    //                         for (int j = 0; j < U.nb_cols(); ++j) {
    //                             std::vector<CoefficientPrecision> col_U(child_k->get_source_cluster().get_size(), 0.0);
    //                             for (int i = 0; i < child_k->get_source_cluster().get_size(); ++i) {
    //                                 col_U[i] = U(i + child_k->get_source_cluster().get_offset() - H->get_target_cluster().get_offset(), j);
    //                             }
    //                             // std::cout << "pour j= " << j << " U_restr ok" << std::endl;
    //                             //  std::vector<CoefficientPrecision> H_col_U(size1, 0.0);
    //                             //   std::cout << child_k->get_target_cluster().get_size() << ',' << child_k->get_source_cluster().get_size() << " x et y " << col_U.size() << ',' << size1 << std::endl;
    //                             //  std::cout << "on veut faire le produit child_k * U_restr on a child_k : " << child_k->get_target_cluster().get_size() << ',' << child_k->get_source_cluster().get_size() << std::endl;
    //                             //  std::cout << " col_U est de taille " << col_U.size() << "et H_col_U " << H_col_U.size() << std::endl;
    //                             //  child_k->add_vector_product('N', 0.0, col_U.data(), 0.0, H_col_U.data());
    //                             //   add vector product ne marche pas j'essaye de passer la hmat en full
    //                             Matrix<CoefficientPrecision> child_k_dense(child_k->get_target_cluster().get_size(), child_k->get_source_cluster().get_size());
    //                             copy_to_dense(*child_k, child_k_dense.data());
    //                             std::vector<CoefficientPrecision> H_col_U = child_k_dense * col_U;
    //                             // std::cout << "produit ok " << std::endl;
    //                             for (int i = 0; i < size1; ++i) {
    //                                 HU(i, j) = H_col_U[i];
    //                             }
    //                         }
    //                         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> hu(HU, V_restr);
    //                         ssr.push_back(hu);
    //                         of.push_back(of1);
    //                         of.push_back(of2);
    //                     }
    //                 }
    //             }
    //             // else if (K->is_dense()) {
    //             //     Matrix<CoefficientPrecision> Kfull(K->get_target_cluster().get_size(), K->get_source_cluster());
    //             //     copy_to_dense(K, Kfull.data());
    //             //     for (int k = 0; k < H->get_children().size(); ++k) {
    //             //         auto child_k = H->get_children()[k] ;
    //             //         if ( (child_k->get_target_cluster().get_size== size1) and ( child_k->get_target_cluster().get_offset() == of1)){
    //             //             Matrix<CoefficientPrecision> K_restr ( child_k->get_target_cluster().get_size() ,  size_2 ) ;
    //             //             for (int i =0 ; i < child_k->get_source_cluster().get_size() ; ++ i ) {
    //             //                 for (int j = 0 ; j < size2 ; ++ j){
    //             //                     K_restr(i,j ) = Kfull(i+ child_k->get_source_cluster().get_offset() - K->get_target_cluster().get_offset() , j + of2 - K->get_source_cluster().get_offset() );
    //             //                 }
    //             //             }

    //             //         }
    //             //     }
    //             // }
    //         } else if (K->get_children().size() > 0) {
    //             if (H->is_low_rank()) {
    //                 // std::cout << "lr * fils" << std::endl;
    //                 //  on fait ligne de V *K
    //                 auto H_lr                      = H->get_low_rank_data();
    //                 Matrix<CoefficientPrecision> U = H_lr->Get_U();
    //                 Matrix<CoefficientPrecision> V = H_lr->Get_V();
    //                 Matrix<CoefficientPrecision> U_restr(size1, U.nb_cols());
    //                 for (int i = 0; i < size1; ++i) {
    //                     for (int j = 0; j < U.nb_cols(); ++j) {
    //                         U_restr(i, j) = V(i + of1 - H->get_target_cluster().get_offset(), j);
    //                     }
    //                 }
    //                 for (int k = 0; k < K->get_children().size(); ++k) {
    //                     auto &child_k = K->get_children()[k];
    //                     if ((child_k->get_source_cluster().get_size() == size2)
    //                         and (child_k->get_source_cluster().get_offset() == of2)) {
    //                         Matrix<CoefficientPrecision> VK(V.nb_rows(), size2);
    //                         for (int i = 0; i < V.nb_rows(); ++i) {
    //                             std::vector<CoefficientPrecision> row_V(V.nb_cols(), 0.0);
    //                             for (int j = 0; j < size2; ++j) {
    //                                 row_V[j] = V(i, j + child_k->get_target_cluster().get_offset() - H->get_source_cluster().get_offset());
    //                             }
    //                             std::vector<CoefficientPrecision> row_V_K(size2, 0.0);
    //                             child_k->add_vector_product('T', 0.0, row_V.data(), 0.0, row_V_K.data());
    //                             for (int j = 0; j < size2; ++j) {
    //                                 VK(i, j) = row_V_K[i];
    //                             }
    //                         }
    //                         LowRankMatrix<CoefficientPrecision, CoordinatePrecision> vk(U_restr, VK);
    //                         ssr.push_back(vk);
    //                         of.push_back(of1);
    //                         of.push_back(of2);
    //                     }
    //                 }
    //             }
    //         } else {
    //             // std::cout << "aucune Hmat " << std::endl;
    //             if ((H->is_low_rank()) and (K->is_low_rank())) {
    //                 // std::cout << "cas 0" << std::endl;
    //                 auto H_lr                      = H->get_low_rank_data();
    //                 auto K_lr                      = K->get_low_rank_data();
    //                 Matrix<CoefficientPrecision> U = H_lr->Get_U();
    //                 Matrix<CoefficientPrecision> V = H_lr->Get_V() * (K_lr->Get_U() * K_lr->Get_V());
    //                 Matrix<CoefficientPrecision> U_restr(size1, U.nb_cols());
    //                 Matrix<CoefficientPrecision> V_restr(V.nb_rows(), size2);
    //                 for (int i = 0; i < size1; ++i) {
    //                     for (int j = 0; j < U.nb_cols(); ++j) {
    //                         U_restr(i, j) = U(i + of1 - H->get_target_cluster().get_offset(), j);
    //                     }
    //                 }
    //                 for (int i = 0; i < V.nb_rows(); ++i) {
    //                     for (int j = 0; j < size2; j++) {
    //                         V_restr(i, j) = V(i, j + of2 - K->get_source_cluster().get_offset());
    //                     }
    //                 }
    //                 LowRankMatrix<CoefficientPrecision, CoordinatePrecision> UV(U_restr, V_restr);
    //                 ssr.push_back(UV);
    //                 of.push_back(of1);
    //                 of.push_back(of2);
    //             } else if (H->is_low_rank()) { // sous entendu K est dense ( sinon je sais pas ce que ca veut dire hiérarchique sans fils et uqi est ni dense ni lr)
    //                 // std::cout << "cas 1" << std::endl;
    //                 auto H_lr                      = H->get_low_rank_data();
    //                 Matrix<CoefficientPrecision> U = H_lr->Get_U();
    //                 Matrix<CoefficientPrecision> V = H_lr->Get_V();
    //                 Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
    //                 copy_to_dense(*K, K_dense.data());
    //                 Matrix<CoefficientPrecision> K_restr(K->get_target_cluster().get_size(), size2);
    //                 for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
    //                     for (int j = 0; j < size2; ++j) {
    //                         K_restr(i, j) = K_dense(i, j + of1 - K->get_source_cluster().get_offset());
    //                     }
    //                 }
    //                 // std::cout << "restriction de K ok " << std::endl;
    //                 Matrix<CoefficientPrecision> U_restr(size1, U.nb_cols());
    //                 for (int i = 0; i < size1; ++i) {
    //                     for (int j = 0; j < U.nb_cols(); ++j) {
    //                         U_restr(i, j) = U(i + of1 - H->get_target_cluster().get_offset(), j);
    //                     }
    //                 }
    //                 // std::cout << " restriction de U ok " << std::endl;
    //                 auto VK = V * K_restr;
    //                 LowRankMatrix<CoefficientPrecision, CoordinatePrecision> vk(U_restr, VK);
    //                 ssr.push_back(vk);
    //                 of.push_back(of1);
    //                 of.push_back(of2);
    //             } else if (K->is_low_rank()) {
    //                 // std::cout << "cas 2 " << std::endl;
    //                 auto K_lr                      = K->get_low_rank_data();
    //                 Matrix<CoefficientPrecision> U = K_lr->Get_U();
    //                 Matrix<CoefficientPrecision> V = K_lr->Get_V();
    //                 // Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
    //                 // copy_to_dense(H->get(), H_dense.data());
    //                 Matrix<CoefficientPrecision> H_dense = *H->get_dense_data();
    //                 Matrix<CoefficientPrecision> V_restr(V.nb_rows(), size2);
    //                 for (int i = 0; i < V.nb_rows(); ++i) {
    //                     for (int j = 0; j < size2; ++j) {
    //                         V_restr(i, j) = V(i, j + of2 - K->get_source_cluster().get_offset());
    //                     }
    //                 }
    //                 Matrix<CoefficientPrecision> H_restr(size1, H->get_source_cluster().get_size());
    //                 for (int i = 0; i < size1; ++i) {
    //                     for (int j = 0; j < H->get_source_cluster().get_size(); ++j) {
    //                         H_restr(i, j) = H_dense(i + of1 - H->get_target_cluster().get_offset(), j);
    //                     }
    //                     auto HU = H_restr * U;
    //                     LowRankMatrix<CoefficientPrecision, CoordinatePrecision> hu(HU, V_restr);
    //                     ssr.push_back(hu);
    //                     of.push_back(of1);
    //                     of.push_back(of2);
    //                 }
    //             } else {
    //                 // std::cout << "cas 3" << std::endl;
    //                 //  normalement le deux sont full ->je sais pas quoi faire -> je les met en loxw rank
    //                 Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
    //                 Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
    //                 Matrix<CoefficientPrecision> H_restr(size1, H->get_source_cluster().get_size());
    //                 Matrix<CoefficientPrecision> K_restr(K->get_target_cluster().get_size(), size2);
    //                 // Matrix<CoefficientPrecision> H_dense = *H->get_dense_data();
    //                 // Matrix<CoefficientPrecision> K_dense = *K->get_dense_data();
    //                 copy_to_dense(*H, H_dense.data());
    //                 // std::cout << "norme du dense " << normFrob(H_dense) << std::endl;
    //                 // std::cout << "cluster H" << H->get_target_cluster().get_size() << ',' << H->get_target_cluster().get_offset() << ',' << H->get_source_cluster().get_size() << ',' << H->get_source_cluster().get_offset() << std::endl;
    //                 // std::cout << "cluster K" << K->get_target_cluster().get_size() << ',' << K->get_target_cluster().get_offset() << ',' << K->get_source_cluster().get_size() << ',' << K->get_source_cluster().get_offset() << std::endl;
    //                 copy_to_dense(*K, K_dense.data());
    //                 for (int i = 0; i < size1; ++i) {
    //                     for (int j = 0; j < H->get_source_cluster().get_size(); ++j) {
    //                         H_restr(i, j) = H_dense(i + of1 - H->get_target_cluster().get_size(), j);
    //                     }
    //                 }
    //                 for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
    //                     for (int j = 0; j < size2; ++j) {
    //                         K_restr(i, j) = K_dense(i, j + of2 - K->get_source_cluster().get_offset());
    //                     }
    //                 }
    //                 LowRankMatrix<CoefficientPrecision> UV(H_restr, K_restr);
    //                 ssr.push_back(UV);
    //                 of.push_back(of1);
    //                 of.push_back(of2);
    //             }
    //         }
    //     }
    //     // std::cout << "on va sortir du restrict " << std::endl;
    //     // std::cout << "ssr.size() " << ssr.size() << ',' << "ssh " << ssh.size() << ',' << "of " << of.size() << std::endl;
    //     SumExpression res(ssr, ssh, of);
    //     // std::cout << " construction de res ok " << std::endl;
    //     return res;
    // }

    // j'arrive pas a le faire en prenant juste les cluster ( les cluster_children sont des uniq_ptr et ca casse tout)
    // SumExpression Restrict(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s) {

    SumExpression Restrict(const int &target_size, const int &source_size, const int &target_offset, const int &source_offset) const {
        std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> sh;
        std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> sr;
        std::vector<int> of;
        for (int k = 0; k < Sr.size(); ++k) {
            std::cout << "sr.size= " << Sr.size() << std::endl;
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr = Sr[k];
            Matrix<CoefficientPrecision> U                              = lr.Get_U();
            Matrix<CoefficientPrecision> V                              = lr.Get_V();
            Matrix<CoefficientPrecision> U_restr(target_size, U.nb_cols());
            Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
            CoefficientPrecision *ptr_U = new CoefficientPrecision[target_size * U.nb_cols()];
            CoefficientPrecision *ptr_V = new CoefficientPrecision[V.nb_rows() * source_size];
            for (int i = 0; i < target_size; ++i) {
                for (int j = 0; j < U.nb_cols(); ++j) {
                    // U_restr(i, j) = U(i + target_offset - offset[2 * k], j);
                    ptr_U[i * U.nb_cols() + j] = U(i + target_offset - offset[2 * k], j);
                }
            }
            for (int i = 0; i < V.nb_rows(); ++i) {
                for (int j = 0; j < source_size; ++j) {
                    // V_restr(i, j) = V(i, j + source_offset - offset[2 * k + 1]);
                    ptr_V[i * source_size + j] = V(i, j + source_offset - offset[2 * k + 1]);
                }
            }
            U_restr.assign(target_size, U.nb_cols(), ptr_U, true);
            V_restr.assign(V.nb_rows(), source_size, ptr_V, true);
            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr_restr(U_restr, V_restr);
            sr.push_back(lr_restr);
            of.push_back(target_offset);
            of.push_back(source_offset);
        }
        for (int rep = 0; rep < Sh.size() / 2; ++rep) {
            HMatrix<CoefficientPrecision, CoordinatePrecision> *H = Sh[2 * rep];
            HMatrix<CoefficientPrecision, CoordinatePrecision> *K = Sh[2 * rep + 1];
            auto &H_childs                                        = H->get_children();
            auto &K_childs                                        = K->get_children();
            if ((H_childs.size() > 0) and (K_childs.size() > 0)) {
                for (int k = 0; k < H_childs.size(); ++k) {
                    auto &H_child = H_childs[k];
                    if ((H_child->get_target_cluster().get_size() == target_size) and (H_child->get_target_cluster().get_offset() == target_offset)) {
                        for (int l = 0; l < K_childs.size(); ++l) {
                            auto &K_child = K_childs[l];
                            if ((H_child->get_source_cluster() == K_child->get_target_cluster()) and ((K_child->get_source_cluster().get_size() == source_size) and (K_child->get_source_cluster().get_offset() == source_offset))) {
                                sh.push_back(H_child.get());
                                sh.push_back(K_child.get());
                            }
                        }
                    }
                }

            } else if (K_childs.size() == 0) {
                if (K->is_low_rank()) {
                    auto K_lr                      = K->get_low_rank_data();
                    Matrix<CoefficientPrecision> U = K_lr->Get_U();
                    Matrix<CoefficientPrecision> V = K_lr->Get_V();
                    Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
                    for (int i = 0; i < V.nb_rows(); ++i) {
                        for (int j = 0; j < source_size; ++j) {
                            V_restr(i, j) = V(i, j + source_offset - K->get_source_cluster().get_offset());
                        }
                    }
                    for (int k = 0; k < H_childs.size(); ++k) {
                        auto &H_child = H_childs[k];
                        if ((H_child->get_target_cluster().get_size() == target_size) and (H_child->get_target_cluster().get_offset() == target_offset)) {
                            // Matrix<CoefficientPrecision> U_restr( H_child->get_source_cluster().get_size() , U.nb_cols());
                            // for (int i= 0 ; i < H_child->get_source_cluster().get_size() ; ++i ) {
                            //     for (int j = 0 ; j < U.nb_cols() ; ++j ){
                            //         U(i,j ) = U_restr(i+ H_child->get_source_cluter().get_offset() -K->get_target_cluster().get_offset() , j);
                            //     }
                            // }
                            // H_child*colonne de U pour construire HU
                            // on copie la colonne j
                            Matrix<CoefficientPrecision> HU(target_size, U.nb_cols());
                            for (int j = 0; j < U.nb_cols(); ++j) {
                                std::vector<CoefficientPrecision> U_col_j(H_child->get_source_cluster().get_size());
                                for (int i = 0; i < H_child->get_source_cluster().get_offset(); ++i) {
                                    U_col_j[i] = U(i + H_child->get_source_cluster().get_offset() - K->get_target_cluster().get_offset(), j);
                                }
                                std::vector<CoefficientPrecision> H_U_col_j(target_size, 0.0);
                                H_child->add_vector_product('N', 1.0, U_col_j.data(), 0.0, H_U_col_j.data());
                                for (int i = 0; i < target_size; ++i) {
                                    HU(i, j) = H_U_col_j[i];
                                }
                            }
                            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(HU, V_restr);
                        }
                    }
                } else { // K dense
                    // normalement si c'est ni lr et que ca a pas de fils c'est dense
                    Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*K, K_dense.data());
                    for (int k = 0; k < H_childs.size(); ++k) {
                        auto &H_child = H_childs[k];
                        if ((H_child->get_target_cluster().get_size() == target_size) and (H_child->get_target_cluster().get_offset() == target_offset)) {
                            Matrix<CoefficientPrecision> K_restr(H_child->get_source_cluster().get_size(), source_size);
                            CoefficientPrecision *ptr_K = new CoefficientPrecision[H_child->get_source_cluster().get_size() * source_size];
                            for (int i = 0; i < H_child->get_source_cluster().get_size(); ++i) {
                                for (int j = 0; j < source_size; ++j) {
                                    // K_restr(i, j) = K_dense(i + H_child->get_source_cluster().get_offset() - K->get_target_cluster().get_offset(), j + source_offset - K->get_source_cluster().get_size());
                                    ptr_K[i * source_size + j] = K_dense(i + H_child->get_source_cluster().get_offset() - K->get_target_cluster().get_offset(), j + source_offset - K->get_source_cluster().get_size());
                                }
                            }
                            K_restr.assign(H_child->get_source_cluster().get_size(), source_size, ptr_K, true);
                            // soit on la stocke en lr soit en full mais il vaut mieux pas la mettre dans sh comme elle a pas de fils
                            Matrix<CoefficientPrecision> H_child_dense(H_child->get_target_cluster().get_size(), H_child->get_source_cluster().get_size());
                            copy_to_dense(*H_child, H_child_dense.data());
                            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lr(H_child_dense, K_restr);
                            sr.push_back(lr), of.push_back(target_offset);
                            of.push_back(source_offset);
                        }
                    }
                }
            } else if (H_childs.size() == 0) {
                if (H->is_low_rank()) {
                    auto H_lr                      = H->get_low_rank_data();
                    Matrix<CoefficientPrecision> U = H_lr->Get_U();
                    Matrix<CoefficientPrecision> V = H_lr->Get_V();
                    Matrix<CoefficientPrecision> U_restr(target_size, U.nb_cols());
                    CoefficientPrecision *ptr_U = new CoefficientPrecision[target_size * U.nb_cols()];
                    for (int i = 0; i < target_size; ++i) {
                        for (int j = 0; j < U.nb_cols(); ++j) {
                            // U_restr(i, j) = U(i + target_offset - H->get_target_cluster().get_size(), j);
                            ptr_U[i * U.nb_cols() + j] = U(i + target_offset - H->get_target_cluster().get_size(), j);
                        }
                    }
                    U_restr.assign(target_size, U.nb_cols(), ptr_U, true);
                    for (int k = 0; k < K_childs.size(); ++k) {
                        auto &K_child = K_childs[k];
                        // lmigne de V fois K pour avoir la ligne de VK
                        // ON COPIE LA LIGNE I
                        if ((K_child->get_source_cluster().get_size() == source_size) and (K_child->get_source_cluster().get_offset() == source_offset)) {
                            Matrix<CoefficientPrecision> VK(V.nb_rows(), source_size);
                            CoefficientPrecision *ptr_VK = new CoefficientPrecision[V.nb_rows() * source_size];
                            for (int i = 0; i < V.nb_rows(); ++i) {
                                std::vector<CoefficientPrecision> V_row_i(K_child->get_source_cluster().get_size());
                                for (int j = 0; j < K_child->get_target_cluster().get_size(); ++j) {
                                    V_row_i[j] = V(i, j + K_child->get_target_cluster().get_offset() - H->get_source_cluster().get_size());
                                }
                                std::vector<CoefficientPrecision> V_row_i_K(source_size, 0.0);
                                K_child->add_vector_product('T', 1.0, V_row_i.data(), 0.0, V_row_i_K.data());
                                for (int j = 0; j < source_size; ++j) {
                                    // VK(i, j) = V_row_i_K[j];
                                    ptr_VK[i * source_size + j] = V_row_i_K[j];
                                }
                            }
                            VK.assign(V.nb_rows(), source_size, ptr_VK, true);
                            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(U_restr, VK);
                            sr.push_back(HK);
                            of.push_back(target_offset);
                            of.push_back(source_offset);
                        }
                    }
                } else { // H dense
                    Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    copy_to_dense(*H, H_dense.data());
                    for (int k = 0; k < K_childs.size(); ++k) {
                        auto &K_child = K_childs[k];
                        // HMatrix<CoefficientPrecision, CoordinatePrecision> *K_child = K_childs[k].get();
                        if ((K_child->get_source_cluster().get_size() == source_size) and (K_child->get_source_cluster().get_offset() == source_offset)) {
                            Matrix<CoefficientPrecision> H_restr(target_size, K_child->get_target_cluster().get_size());
                            CoefficientPrecision *ptr_H = new CoefficientPrecision[target_size * K_child->get_target_cluster().get_size()];
                            for (int i = 0; i < target_size; ++i) {
                                for (int j = 0; j < K_child->get_target_cluster().get_size(); ++j) {
                                    // H_restr(i, j) = H_dense(i + target_offset - H->get_target_cluster().get_offset(), j + K_child->get_target_cluster().get_offset() - H->get_source_cluster().get_offset());
                                    ptr_H[i * K_child->get_target_cluster().get_size() + j] = H_dense(i + target_offset - H->get_target_cluster().get_offset(), j + K_child->get_target_cluster().get_offset() - H->get_source_cluster().get_offset());
                                }
                            }
                            H_restr.assign(target_size, K_child->get_target_cluster().get_size(), ptr_H, true);
                            std::cout << target_size << ',' << K_child->get_target_cluster().get_size() << ',' << source_size << ',' << K_child->get_source_cluster().get_size() << std::endl;
                            std::cout << "teeeeest" << std::endl;
                            // Matrix<CoefficientPrecision> K_child_dense(target_size, K_child->get_target_cluster().get_size());
                            // std::cout << "size matrice " << K_child->get_target_cluster().get_size() << ',' << source_size << std::endl;
                            Matrix<CoefficientPrecision> K_child_dense(K_child->get_target_cluster().get_size(), source_size);
                            std::cout << "test ok" << std::endl;
                            //  Matrix<CoefficientPrecision> K_child_dense(10, 10);
                            copy_to_dense(*K_child, K_child_dense.data());
                            LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(H_restr, K_child_dense);
                            sr.push_back(HK);
                            of.push_back(target_offset);
                            of.push_back(source_offset);
                        }
                    }
                }

            } else {
                std::cout << "no hmat " << std::endl;
                // normalement la aucune n'est une Hmat
                if ((H->is_low_rank()) and (K->is_low_rank())) {
                    auto H_lr                        = H->get_low_rank_data();
                    auto K_lr                        = K->get_low_rank_data();
                    Matrix<CoefficientPrecision> H_U = H_lr->Get_U();
                    Matrix<CoefficientPrecision> H_V = H_lr->Get_V();
                    Matrix<CoefficientPrecision> K_U = K_lr->Get_U();
                    Matrix<CoefficientPrecision> K_V = K_lr->Get_V();
                    Matrix<CoefficientPrecision> V   = H_V * (K_U * K_V);
                    Matrix<CoefficientPrecision> U_restr(target_size, H_U.nb_cols());
                    Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
                    CoefficientPrecision *ptr_U = new CoefficientPrecision[target_size * H_U.nb_cols()];
                    CoefficientPrecision *ptr_V = new CoefficientPrecision[V.nb_rows() * source_size];
                    for (int i = 0; i < target_size; ++i) {
                        for (int j = 0; j < H_U.nb_cols(); ++j) {
                            // U_restr(i, j) = H_U(i + target_offset - H->get_target_cluster().get_offset(), j);
                            ptr_U[i * H_U.nb_cols() + j] = H_U(i + target_offset - H->get_target_cluster().get_offset(), j);
                        }
                    }
                    U_restr.assign(target_size, H_U.nb_cols(), ptr_U, true);
                    for (int i = 0; i < V.nb_rows(); ++i) {
                        for (int j = 0; j < source_size; ++j) {
                            // V_restr(i, j) = V(i, j + source_offset - K->get_source_cluster().get_offset());
                            ptr_V[i * V.nb_rows() + j] = V(i, j + source_offset - K->get_source_cluster().get_offset());
                        }
                    }
                    V_restr.assign(V.nb_rows(), source_size, ptr_V, true);
                    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(U_restr, V_restr);
                    sr.push_back(HK);
                    of.push_back(target_offset);
                    of.push_back(source_offset);
                } else if (H->is_low_rank()) {
                    auto H_lr                      = H->get_low_rank_data();
                    Matrix<CoefficientPrecision> U = H_lr->Get_U();
                    Matrix<CoefficientPrecision> V = H_lr->Get_V();
                    Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*K, K_dense.data());
                    Matrix<CoefficientPrecision> K_restr(K->get_target_cluster().get_size(), source_size);
                    CoefficientPrecision *ptr_K = new CoefficientPrecision[K->get_target_cluster().get_size() * source_size];
                    for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
                        for (int j = 0; j < source_size; ++j) {
                            // K_restr(i, j) = K_dense(i, j + source_offset - K->get_source_cluster().get_offset());
                            ptr_K[i * source_size + j] = K_dense(i, j + source_offset - K->get_source_cluster().get_offset());
                        }
                    }
                    K_restr.assign(K->get_target_cluster().get_size(), source_size, ptr_K, true);
                    Matrix<CoefficientPrecision> U_restr(target_size, U.nb_cols());
                    CoefficientPrecision *ptr_U = new CoefficientPrecision[target_size * U.nb_cols()];
                    for (int i = 0; i < target_size; ++i) {
                        for (int j = 0; j < U.nb_cols(); ++j) {
                            // U_restr(i, j) = U(i + target_offset - H->get_target_cluster().get_offset(), j);
                            ptr_U[i * U.nb_cols() + j] = U(i + target_offset - H->get_target_cluster().get_offset(), j);
                        }
                    }
                    U_restr.assign(target_size, U.nb_cols(), ptr_U, true);
                    Matrix<CoefficientPrecision> VK = V * K_restr;
                    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(U_restr, VK);
                    sr.push_back(HK);
                    of.push_back(target_offset);
                    of.push_back(source_offset);

                } else if (K->is_low_rank()) {
                    auto K_lr                      = K->get_low_rank_data();
                    Matrix<CoefficientPrecision> U = K_lr->Get_U();
                    Matrix<CoefficientPrecision> V = K_lr->Get_V();
                    Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    copy_to_dense(*H, H_dense.data());
                    Matrix<CoefficientPrecision> H_restr(target_size, H->get_source_cluster().get_size());
                    CoefficientPrecision *ptr_H = new CoefficientPrecision[target_size * H->get_source_cluster().get_size()];
                    for (int i = 0; i < target_size; ++i) {
                        for (int j = 0; j < H->get_source_cluster().get_size(); ++j) {
                            // H_restr(i, j) = H_dense(i + target_offset - H->get_source_cluster().get_offset(), j);
                            ptr_H[i * H->get_source_cluster().get_size() + j] = H_dense(i + target_offset - H->get_source_cluster().get_offset(), j);
                        }
                    }
                    H_restr.assign(target_size, H->get_source_cluster().get_size(), ptr_H, true);
                    Matrix<CoefficientPrecision> V_restr(V.nb_rows(), source_size);
                    CoefficientPrecision *ptr_V = new CoefficientPrecision[V.nb_rows() * source_size];
                    for (int i = 0; i < V.nb_rows(); ++i) {
                        for (int j = 0; j < source_size; ++j) {
                            // V_restr(i, j) = V(i, j + source_offset - K->get_source_cluster().get_offset());
                            ptr_V[i * source_size + j] = V(i, j + source_offset - K->get_source_cluster().get_offset());
                        }
                    }
                    V_restr.assign(V.nb_rows(), source_size, ptr_V, true);
                    Matrix<CoefficientPrecision> HU = H_restr * U;
                    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(HU, V_restr);
                    sr.push_back(HK);
                    of.push_back(target_offset);
                    of.push_back(source_offset);
                }

                else { // les deux sont dense
                    Matrix<CoefficientPrecision> H_dense(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> K_dense(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
                    copy_to_dense(*H, H_dense.data());
                    copy_to_dense(*K, K_dense.data());
                    Matrix<CoefficientPrecision> H_restr(target_size, H->get_source_cluster().get_size());
                    Matrix<CoefficientPrecision> K_restr(K->get_target_cluster().get_size(), source_size);
                    CoefficientPrecision *ptr_H = new CoefficientPrecision[target_size * H->get_source_cluster().get_size()];
                    CoefficientPrecision *ptr_K = new CoefficientPrecision[K->get_target_cluster().get_size() * source_size];
                    for (int i = 0; i < target_size; ++i) {
                        for (int j = 0; j < H->get_source_cluster().get_size(); ++j) {
                            // H_restr(i, j) = H_dense(i + target_offset - H->get_target_cluster().get_offset(), j);
                            ptr_H[i * target_size + j] = H_dense(i + target_offset - H->get_target_cluster().get_offset(), j);
                        }
                    }
                    for (int i = 0; i < K->get_target_cluster().get_size(); ++i) {
                        for (int j = 0; j < source_size; ++j) {
                            // K_restr(i, j) = K_dense(i, j + source_offset - K->get_source_cluster().get_offset());
                            ptr_K[i * source_size + j] = K_dense(i, j + source_offset - K->get_source_cluster().get_offset());
                        }
                    }
                    H_restr.assign(target_size, H->get_source_cluster().get_size(), ptr_H, true);
                    K_restr.assign(K->get_target_cluster().get_size(), source_size, ptr_K, true);
                    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> HK(H_restr, K_restr);
                    sr.push_back(HK);
                    of.push_back(target_offset);
                    of.push_back(source_offset);
                }
            }
        }
        SumExpression res(sr, sh, of);
        return res;
    }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        // Matrix<CoefficientPrecision> result(M, N);
        // new CoefficientPrecision *ptr(M * N);
        // result.assign(M, N, ptr, false);
        auto H = *this;
        for (int k = 0; k < M; ++k) {
            for (int l = 0; l < N; ++l) {
                std::vector<CoefficientPrecision> xl(nc, 0.0);
                xl[l + col_offset] = 1.0;
                // std::vector<CoefficientPrecision> y(nr, 0.0);
                std::vector<CoefficientPrecision> y = H.prod(xl);
                ptr[k + M * l]                      = y[k + row_offset];
            }
        }
    }
    // lowrank * hmat = low rank et ca restrict le voit pas donc il faut faire un tri
    // void Sort(){
    //   std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> sr = Sr;
    //   std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision>*> sh;
    //   std::vector<int > of = offset;
    //   if ( Sh.size() > 0){
    //     for (int k =0 ; k < Sh.size()/2 ; ++k ){
    //       auto H = Sh[2*k] ; auto K = Sh[2*k+1];
    //       std::cout << " flag du sort : is low rank :"<<H->is_low_rank() << ',' << K->is_low_rank()<< std::endl;
    //       if ( (H->is_low_rank() )  and !(K->is_low_rank()) ){
    //         std::cout<< 1 << std::endl;
    //         auto lr = H->get_low_rank_data() ; auto U = lr->Get_U() ; auto V = lr->Get_V();
    //         // on récupére les lignes de V et on fait du (matrice vecteur)  transp pour construire V'
    //         std::cout <<" 1 lr mat ok " << std::endl;
    //         Matrix<CoefficientPrecision> Vp(V.nb_rows(), K->get_source_cluster().get_size());
    //         for (int i =0 ; i < V.nb_rows(); ++i){
    //           std::vector<CoefficientPrecision> x(V.nb_cols());
    //           for (int j =0 ; j < V.nb_cols() ; ++j ){
    //             x[j] = V(i,j);
    //           }
    //           std::vector<CoefficientPrecision> y (K->get_source_cluster().get_size() , 0);
    //           std::cout << " serait ce ici " << std::endl;
    //           std::cout<<"----------------------------------"<< std::endl ;
    //           std::cout<< "on fait ligne de V * K"<< std::endl;
    //           std::cout << "dimp de  K " <<K->get_target_cluster().get_size() << ',' << K->get_source_cluster().get_size() << std::endl;
    //           std::cout << "dim de V"  << V.nb_rows() << ',' << V.nb_cols() << std::endl;
    //           std::cout<< "taille de x et y " << x.size() << ',' << y.size() << std::endl;
    //           std::cout <<"___________________________________________" << std::endl;
    //           K->add_vector_product('T',1.0, x.data(), 0.0, y.data());
    //           std::cout<<"toujours pas " << std::endl;
    //           std::cout<< "on écrit la restriction" << std::endl;
    //           for (int j =0 ; j< K->get_source_cluster().get_size() ; ++j ){
    //             Vp(i,j) = y[j];
    //           }
    //           }
    //           std::cout<<"restriction ok " << std::endl;
    //           int of1 = H->get_target_cluster().get_offset() ; int of2 = K->get_source_cluster().get_offset();
    //           LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lrp ( U , Vp);
    //           sr.push_back(lrp) ; of.push_back(of1) ; of.push_back(of2);
    //         std::cout << "1 ok " << std::endl;
    //         }
    //       else if (!(H->is_low_rank() )  and (K->is_low_rank()) ){
    //         std::cout<< 1 << std::endl;
    //         auto lr = K->get_low_rank_data() ; auto U = lr->Get_U() ; auto V = lr->Get_V();
    //         // On récupere les collonnes de U et on fait hmat vevteur
    //         Matrix<CoefficientPrecision> Up(H->get_target_cluster().get_size() , U.nb_cols());
    //         for (int j =0 ; j < U.nb_cols() ; ++j){
    //           std::vector<CoefficientPrecision> x(U.nb_rows());
    //           for (int i =0 ; i < U.nb_rows() ; ++i ){
    //             x[i] = U(i,j);
    //           }
    //           std::vector<CoefficientPrecision> y ( H->get_target_cluster().get_size() , 0);
    //           std::cout << " serait ce ici " << std::endl;
    //           std::cout<<"----------------------------------"<< std::endl ;
    //           std::cout<< "on fait H*colonnes de U"<< std::endl;
    //           std::cout << "dimp de  H " <<H->get_target_cluster().get_size() << ',' << H->get_source_cluster().get_size() << std::endl;
    //           std::cout << "dim de U"  << U.nb_rows() << ',' << U.nb_cols() << std::endl;
    //           std::cout<< "taille de x et y " << x.size() << ',' << y.size() << std::endl;
    //           std::cout <<"___________________________________________" << std::endl;
    //           H->add_vector_product('N',1.0, x.data(), 0.0, y.data());
    //           std::cout<<"toujours pas " << std::endl;
    //           std::cout<< "on écrit la restriction" << std::endl;
    //           for (int i =0 ; i < H->get_target_cluster().get_size() ; ++ i ){
    //             Up(i,j) = y[i];
    //           }
    //         }
    //           std::cout<<"restriction ok " << std::endl;
    //           int of1 = H->get_target_cluster().get_offset() ; int of2 = K->get_source_cluster().get_offset();
    //           LowRankMatrix<CoefficientPrecision, CoordinatePrecision> lrp ( Up , V);
    //           std::cout<< "assemblage lrmat ok " << std::endl;
    //           sr.push_back(lrp) ; of.push_back(of1) ; of.push_back(of2);
    //           std::cout<< "actualistion ok " << std::endl;

    //         }
    //       else if ( (H->is_low_rank() ) and (K->is_low_rank() )){
    //         auto Ua = H->get_low_rank_data()->Get_U()  ; auto Va = H->get_low_rank_data()->Get_V();
    //         auto Ub = K->get_low_rank_data()->Get_U()  ; auto Vb = K->get_low_rank_data()->Get_V();
    //         std::cout<< "est ce que ca pète " << std::endl;
    //         auto V = Va * (Ub * Vb);
    //         std::cout << "ca n'a pas péter " << std::endl;
    //         LowRankMatrix<CoefficientPrecision,CoordinatePrecision> lrk(Ua , V);
    //         sr.push_back(lrk) ; of.push_back(H->get_target_cluster().get_offset() )  ; of.push_back(K->get_source_cluster().get_offset());
    //       }
    //       else if ( (H->is_dense()) and (K->is_dense())){
    //         Matrix<CoefficientPrecision> fh (H->get_target_cluster().get_size() , H->get_source_cluster().get_size() );
    //         Matrix<CoefficientPrecision> fk(K->get_target_cluster().get_size() , K->get_source_cluster().get_size() );
    //         copy_to_dense(*H,fh.data()) ; copy_to_dense(*K,fk.data());
    //         LowRankMatrix<CoefficientPrecision> lrk (fh,fk);
    //         sr.push_back(lrk) ; of.push_back(H->get_target_cluster().get_offset()) ; of.push_back(K->get_source_cluster().get_offset());
    //         }

    //       else{
    //         std::cout<< "?" << std::endl;
    //         sh.push_back(H) ; sh.push_back(K);
    //         std::cout<< "waaaaaaaaaaaaaaaaaaaaaaaaaaaat" << std::endl;
    //       }
    //       }
    //       std::cout<< "c'est la fin"<< std::endl;
    //       Sr = sr ; Sh =sh ; offset = of ; }
    //       std::cout<< "c'est fini" << std::endl;
    //   }

    // Sort :  on parcourq sh et on garde si il y en a aucun lr

    // Problème quand on fait hmat lrmat

    void Sort() {
        std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> sr = Sr;
        std::vector<int> of                                                      = offset;
        std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> sh;
        // On reagrde toute les paires de hmat et on garde que celles qui toutes les deux hmat
        if (Sh.size() > 0) {
            // On parcours les paires
            for (int k = 0; k < Sh.size() / 2; ++k) {
                auto H = Sh[2 * k];
                auto K = Sh[2 * k + 1];
                if (!(H->is_low_rank()) and !(K->is_low_rank())) {
                    sh.push_back(H);
                    sh.push_back(K);
                } else {
                    // On doit différencier les cas
                    // les deux sont lr
                    if ((H->is_low_rank()) and (K->is_low_rank())) {
                        auto uh = H->get_low_rank_data()->Get_U();
                        auto vh = H->get_low_rank_data()->Get_V();
                        auto uk = K->get_low_rank_data()->Get_U();
                        auto vk = K->get_low_rank_data()->Get_V();
                        LowRankMatrix<CoefficientPrecision> lrk(uh, vh * (uk * vk));
                        sr.push_back(lrk);
                        of.push_back(H->get_target_cluster().get_offset());
                        of.push_back(K->get_source_cluster().get_offset());
                    }
                    // celle de gauche est lr -> lr( U , V*K) , on fait V* K en faisant K^T *(lignes de V)^T
                    else if ((H->is_low_rank()) and !(K->is_low_rank())) {
                        std::cout << " transp begin" << std::endl;
                        auto uh = H->get_low_rank_data()->Get_U();
                        auto vh = H->get_low_rank_data()->Get_V();
                        std::cout << " u v ok" << std::endl;
                        Matrix<CoefficientPrecision> M(vh.nb_rows(), K->get_target_cluster().get_size());
                        // on extrait chaque lignes
                        for (int i = 0; i < vh.nb_rows(); ++i) {
                            std::vector<CoefficientPrecision> x(vh.nb_cols()); // la ligne i
                            for (int j = 0; j < vh.nb_cols(); ++j) {
                                x[j] = vh(i, j);
                            }
                            std::vector<CoordinatePrecision> y(K->get_source_cluster().get_size(), 0.0); // y = ligne i * K
                            // std::cout<< "y size "<<y.size() << ',' << "x size " << x.size() << " dim de K "<< K->get_target_cluster().get_size() <<',' << K->get_source_cluster().get_size() << std::endl;
                            std::cout << " prod begin" << std::endl;
                            std::cout << K->get_target_cluster().get_size() << ',' << K->get_source_cluster().get_size() << " x et y " << x.size() << ',' << y.size() << std::endl;
                            K->add_vector_product('T', 1.0, x.data(), 0.0, y.data());
                            std::cout << " produit ok" << std::endl;
                            // maintenant on reporte la ligne y dans M
                            for (int j = 0; j < K->get_source_cluster().get_size(); ++j) {
                                M(i, j) = y[j];
                            }
                        }
                        LowRankMatrix<CoefficientPrecision> lrk(uh, M);
                        sr.push_back(lrk);
                        of.push_back(H->get_target_cluster().get_offset());
                        of.push_back(K->get_source_cluster().get_offset());
                        std::cout << " transp end " << std::endl;
                    }
                    // celle de  droite est lr -> lr ( H*U , V) , on fait H*U en faisant H*colonnes U
                    else if (!(H->is_low_rank()) and (K->is_low_rank())) {
                        std::cout << "normal begin " << std::endl;
                        auto uk = K->get_low_rank_data()->Get_U();
                        auto vk = K->get_low_rank_data()->Get_V();
                        Matrix<CoefficientPrecision> M(H->get_target_cluster().get_size(), uk.nb_cols());
                        // on extrait chaque colonnes
                        for (int j = 0; j < uk.nb_cols(); ++j) {
                            std::vector<CoefficientPrecision> x(uk.nb_rows(), 0); // la colonne j
                            for (int i = 0; i < uk.nb_rows(); ++i) {
                                x[i] = uk(i, j);
                            }
                            std::vector<CoefficientPrecision> y(H->get_target_cluster().get_size(), 0); // la colonne j
                            H->add_vector_product('N', 1.0, x.data(), 0.0, y.data());
                            std::cout << "produit ok " << std::endl;
                            // Maitenant on reporte la colonne y dans M
                            for (int i = 0; i < H->get_target_cluster().get_size(); ++i) {
                                M(i, j) = y[i];
                            }
                        }
                        LowRankMatrix<CoefficientPrecision> lrk(M, vk);
                        sr.push_back(lrk);
                        of.push_back(H->get_target_cluster().get_offset());
                        of.push_back(K->get_source_cluster().get_offset());
                        std::cout << " normal end " << std::endl;
                    }
                }
            }
            Sr     = sr;
            Sh     = sh;
            offset = of;
        }
    }

    Matrix<CoefficientPrecision> Evaluate() {
        Matrix<CoefficientPrecision> res(nr, nc);
        for (int k = 0; k < Sr.size(); ++k) {
            auto lra = Sr[k]; //  auto lrb = Sr[2*k+1];
            // Matrix<CoefficientPrecision> U,V,W,X;
            auto U = lra.Get_U();
            auto V = lra.Get_V(); // W = lrb.Get_U() ; X = lrb.Get_V();
            res    = res + U * V;
        }
        for (int k = 0; k < Sh.size() / 2; ++k) {
            auto H = Sh[2 * k];
            auto K = Sh[2 * k + 1];
            Matrix<CoefficientPrecision> Hh(H->get_target_cluster().get_size(), H->get_source_cluster().get_size());
            Matrix<CoefficientPrecision> Kk(K->get_target_cluster().get_size(), K->get_source_cluster().get_size());
            copy_to_dense(*H, Hh.data());
            copy_to_dense(*K, Kk.data());
            res = res + Hh * Kk;
        }
        return res;
    }

    // J'en fait une de bourrin
    //  LowRankMatrix<CoefficientPrecision, CoordinatePrecision> Troncature(const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s,std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> LR_Generator,int rk_max =-1, CoefficientPrecision epsilon = 0.001){
    //    Matrix<CoefficientPrecision> full = this->Evaluate; Matrix<CoefficientPrecision> U,V;
    //    LR_Generator->copy_low_rank_approximation(full,t,s,rk_max,epsilon,rk_max,U,V );
    //    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> tronc(U,V);
    //    return tronc;

    // }
    // HMult mais avce troncature de bourrin

    // bloc courant = L
    //  pour utiliser cette fonction : sumexpr( H,K) ; Hmat( cluster total, cluster total) L ; sumexpr.Hmult(L)

    // void Hmult( HMatrix<CoefficientPrecision, CoordinatePrecision>* L , sumexpr){
    //   std::cout << "on appelle hmult sur le bloc de taille "<< L->get_target_cluster().get_size()<<'x'<< L->get_source_cluster().get_size() << " et d'offset " << L->get_target_cluster().get_offset() <<'x'<< L->get_source_cluster().get_offset() << std::endl;
    //   auto& t =  L->get_target_cluster() ; auto& s = L->get_source_cluster();
    //   auto& tt= t.get_children() ; auto& ss = s.get_children();

    //   const auto &target_cluster       = L->get_target_cluster();
    //   const auto &source_cluster       = L->get_source_cluster();
    //   const auto &target_children = target_cluster.get_children();
    //   const auto &source_children = source_cluster.get_children();

    //   if (!target_cluster.is_leaf() && !source_cluster.is_leaf()){
    //   for (const auto& target_child : target_children){
    //     for (const auto& source_child : source_children){
    //       HMatrix<CoefficientPrecision, CoordinatePrecision>* Lk = L->add_child(target_child.get(),source_child.get());
    //       SumExpression srestr = this->Restrict(sz1,sz2,of1,of2);
    //       //sort
    //       // call recursive avec srestr et Lk

    //     }

    //   }}
    //   else{
    //     if() // admissibility{
    //       // Lk.compute_low_rank_data(sumeprx,...)
    //     }else{ //evaluate
    //       //Lk.compute_dense_data(sumexpr,...)
    //     }
    //   }

    //   // si les cluster ont des fils on fait la restriction sur eux et on rappele
    //   if ( (tt.size() > 0 ) and ( ss.size() > 0)){
    //     for (int k =0 ; k < tt.size() ; ++k ){
    //       auto& tk = tt[k]; int of1 = tk->get_offset() ; int sz1 = tk->get_size() ;
    //       for (int l =0 ; l < ss.size() ; ++l ){
    //         auto& sl = ss[l]; int of2 = sl->get_offset() ; int sz2 = sl->get_size() ;
    //         SumExpression srestr = this->Restrict(sz1,sz2,of1,of2);
    //         srestr.Sort();
    //         std::cout << "?!" << std::endl;
    //         std::cout << "taille des cluster : " << sz1 << ',' << sz2 << std::endl;
    //         std::cout<<"_________________________________"<< std::endl;
    //         std::cout<< "c'est le constructeur ?" << std::endl;
    //         HMatrix<CoefficientPrecision, CoordinatePrecision> Lk (*L,&*tk , &*sl);
    //         HMatrix* Lk = L->add_child(tk.get(),)
    //         std::cout << "a bah non " << std::endl;
    //         std::cout<<"_________________________________"<< std::endl;
    //         srestr.Hmult(&Lk);
    //       }
    //     }
    //   }
    //   else if (tt.size()> 0){
    //     for (int k =0 ; k < tt.size() ; ++k ){
    //       auto& tk = tt[k]; int of1 = tk->get_offset() ; int sz1 =  tk->get_size();
    //       int of2 = s.get_offset() ; int sz2 = s.get_size() ;
    //       SumExpression srestr = this->Restrict(sz1,sz2,of1,of2);
    //       std::cout<< "avant le sort" << std::endl;
    //       std::cout << srestr.get_sr().size() << ',' << srestr.get_sh().size() << std::endl;
    //       std::cout << "après le sort " << std::endl;
    //       srestr.Sort();
    //       std::cout << srestr.get_sr().size() << ',' << srestr.get_sh().size() << std::endl;
    //       HMatrix<CoefficientPrecision, CoordinatePrecision> Lk (*L,&*tk , &s);
    //       srestr.Hmult(&Lk);
    //     }
    //   }
    //   else if (ss.size() >0) {
    //     for (int l =0 ; l< ss.size() ; ++l ){
    //       auto& sl = ss[l] ; int of1 = t.get_offset() ; int sz1 = t.get_size();
    //       int of2 = sl->get_offset() ;int sz2 = sl->get_size() ;
    //       SumExpression srestr = this->Restrict(sz1,sz2,of1,of2);
    //       std::cout<< "avant le sort" << std::endl;
    //       std::cout << srestr.get_sr().size() << ',' << srestr.get_sh().size() << std::endl;
    //       std::cout << "après le sort " << std::endl;
    //       srestr.Sort();
    //       std::cout << srestr.get_sr().size() << ',' << srestr.get_sh().size() << std::endl;
    //       HMatrix<CoefficientPrecision, CoordinatePrecision> Lk (*L,&t , &*sl);
    //       srestr.Hmult(&Lk);

    //     }
    //   }
    //   else if ( (ss.size() ==0) and (tt.size()==0)){
    //     std::cout<< "feuilles"<< std::endl;}
    //     //donc la on est sur une feuille normalement
    //     // si c'est pas admissible evaluate
    //     //auto adm = L->m_tree_data->m_admissibility_condition; -> j'arrive pas a la chopper donc je vais la mettre en dure pour l'instant
    //    //bool admissible = 2 * std::min(L->get_target_cluster().get_radius(), L->get_source_cluster().get_radius()) < L->get_eta() * std::max((norm2(L->get_target_cluster().get_center() - L->get_source_cluster().get_center()) - L->get_target_cluster().get_radius() - L->get_source_cluster().get_radius()), 0.);
    //     // Matrix<CoefficientPrecision> val = this->Evaluate();
    //     // MatrixGenerator<CoefficientPrecision> Val( val);
    //     // if (admissible ){
    //     //   //troncature
    //     //   //LowRankMatrix<CoefficientPrecision, CoordinatePrecision> tronc =this->Troncature( L->get_target_cluster() , L->get_source_cluster() , L->get_lr() , L->get_epsilon() , L->get_rk() );
    //     //   L->compute_low_rank_data(Val,*(L->get_lr()), L->get_rk() , L->get_epsilon());
    //     // }
    //     // else{
    //     //   //Matrix<CoefficientPrecision> val = this->Evaluate();
    //     //   L->compute_dense_data(Val);
    //     // }
    //   //}
    // }

    // append_R(Block<T> *, Block<T> *);
    // append_H(Block<T> *, Block<T> *);
};
} // namespace htool
#endif
