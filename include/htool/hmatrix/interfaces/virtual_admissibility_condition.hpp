#ifndef HTOOL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP
#define HTOOL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP

#include "../../basic_types/vector.hpp"
#include "../../clustering/cluster_node.hpp"

namespace htool {

template <typename CoordinatePrecision>
class VirtualAdmissibilityCondition {
  public:
    virtual bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const = 0;
    virtual ~VirtualAdmissibilityCondition() {}
};

// Rjasanow - Steinbach (3.15) p111 Chap Approximation of Boundary Element Matrices
template <typename CoordinatePrecision>
class RjasanowSteinbach final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
  public:
    bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
        bool admissible = 2 * std::min(target.get_radius(), source.get_radius()) < eta * std::max((norm2(target.get_center() - source.get_center()) - target.get_radius() - source.get_radius()), 0.);
        return admissible;
    }
};

template <typename CoordinatePrecision>
class Directional final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
  private:
    std::vector<CoordinatePrecision> convection;

  public:
    Directional(std::vector<CoordinatePrecision> &convection0) { convection = convection0; }
    bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
        bool admissible_elliptique = 2 * std::min(target.get_radius(), source.get_radius()) < eta * std::max((norm2(target.get_center() - source.get_center()) - target.get_radius() - source.get_radius()), 0.);
        std::vector<std::vector<CoordinatePrecision>> Perp(convection.size() - 1);
        if (convection.size() == 2) {
            std::vector<CoordinatePrecision> perp(2);
            perp[0] = -convection[1];
            perp[1] = convection[0];
            for (int i = 0; i < perp.size(); ++i) {
                perp[i] = perp[i] / norm2(perp);
            }
            Perp[0] = perp;
        } else if (convection.size() == 3) {
            std::vector<CoordinatePrecision> perp1(3);
            std::vector<CoordinatePrecision> perp2(3);
            perp1[0] = -convection[0] * convection[2];
            perp1[1] = -convection[1] * convection[2];
            perp1[2] = convection[0] * convection[0] + convection[1] * convection[1];
            perp2[0] = -convection[1];
            perp2[1] = convection[0];
            for (int i = 0; i < perp1.size(); ++i) {
                perp1[i] = perp1[i] / norm2(perp1);
                perp2[i] = perp2[i] / norm2(perp2);
            }
            Perp[0] = perp1;
            Perp[1] = perp2;
        }
        auto direction = target.get_center() - source.get_center();
        std::vector<CoordinatePrecision> projection(convection.size());
        for (int i = 0; i < Perp.size(); ++i) {
            auto e_i  = Perp[i];
            double dd = 0.0;
            for (int kk = 0; kk < e_i.size(); ++kk) {
                dd += (e_i[kk] * direction[kk]);
            }
            std::vector<CoordinatePrecision> temp(e_i.size());
            for (int kk = 0; kk < e_i.size(); ++kk) {
                temp[kk] = e_i[kk] * dd;
            }
            projection = projection + temp;
        }
        bool admissible_directional = norm2(projection) > (target.get_radius() + source.get_radius());
        return ((admissible_directional));
    }
};

template <typename CoordinatePrecision>
class Hybride final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
  private:
    std::vector<CoordinatePrecision> convection;

  public:
    Hybride(std::vector<CoordinatePrecision> &convection0) { convection = convection0; }
    bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
        bool admissible_elliptique = 2 * std::min(target.get_radius(), source.get_radius()) < eta * std::max((norm2(target.get_center() - source.get_center()) - target.get_radius() - source.get_radius()), 0.);
        std::vector<std::vector<CoordinatePrecision>> Perp(convection.size() - 1);
        if (convection.size() == 2) {
            std::vector<CoordinatePrecision> perp(2);
            perp[0] = -convection[1];
            perp[1] = convection[0];
            for (int i = 0; i < perp.size(); ++i) {
                perp[i] = perp[i] / norm2(perp);
            }
            Perp[0] = perp;
        } else if (convection.size() == 3) {
            std::vector<CoordinatePrecision> perp1(3);
            std::vector<CoordinatePrecision> perp2(3);
            perp1[0] = -convection[0] * convection[2];
            perp1[1] = -convection[1] * convection[2];
            perp1[2] = convection[0] * convection[0] + convection[1] * convection[1];
            perp2[0] = -convection[1];
            perp2[1] = convection[0];
            for (int i = 0; i < perp1.size(); ++i) {
                perp1[i] = perp1[i] / norm2(perp1);
                perp2[i] = perp2[i] / norm2(perp2);
            }
            Perp[0] = perp1;
            Perp[1] = perp2;
        }
        auto direction = target.get_center() - source.get_center();
        std::vector<CoordinatePrecision> projection(convection.size());
        for (int i = 0; i < Perp.size(); ++i) {
            auto e_i  = Perp[i];
            double dd = 0.0;
            for (int kk = 0; kk < e_i.size(); ++kk) {
                dd += (e_i[kk] * direction[kk]);
            }
            std::vector<CoordinatePrecision> temp(e_i.size());
            for (int kk = 0; kk < e_i.size(); ++kk) {
                temp[kk] = e_i[kk] * dd;
            }
            projection = projection + temp;
        }
        bool admissible_directional = norm2(projection) > (target.get_radius() + source.get_radius());
        return ((admissible_directional) and (admissible_elliptique));
    }
};

template <typename CoordinatePrecision>
class directional_plus final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
  private:
    std::vector<CoordinatePrecision> convection;

  public:
    directional_plus(std::vector<CoordinatePrecision> &convection0) { convection = convection0; }
    bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
        std::vector<std::vector<CoordinatePrecision>> Perp(convection.size() - 1);
        bool admissible_directional;
        if (convection.size() == 2) {
            std::vector<CoordinatePrecision> perp(2);
            perp[0] = -convection[1];
            perp[1] = convection[0];
            for (int i = 0; i < perp.size(); ++i) {
                perp[i] = perp[i] / norm2(perp);
            }
            Perp[0] = perp;
        } else if (convection.size() == 3) {
            std::vector<CoordinatePrecision> perp1(3);
            std::vector<CoordinatePrecision> perp2(3);
            perp1[0] = -convection[0] * convection[2];
            perp1[1] = -convection[1] * convection[2];
            perp1[2] = convection[0] * convection[0] + convection[1] * convection[1];
            perp2[0] = -convection[1];
            perp2[1] = convection[0];
            for (int i = 0; i < perp1.size(); ++i) {
                perp1[i] = perp1[i] / norm2(perp1);
                perp2[i] = perp2[i] / norm2(perp2);
            }
            Perp[0] = perp1;
            Perp[1] = perp2;
        }
        auto direction = target.get_center() - source.get_center();
        double test    = 0.0;
        for (int k = 0; k < 3; ++k) {
            test += convection[k] * direction[k];
        }
        if (test > 0) {
            std::vector<CoordinatePrecision> projection(convection.size());
            for (int i = 0; i < Perp.size(); ++i) {
                auto e_i  = Perp[i];
                double dd = 0.0;
                for (int kk = 0; kk < e_i.size(); ++kk) {
                    dd += (e_i[kk] * direction[kk]);
                }
                std::vector<CoordinatePrecision> temp(e_i.size());
                for (int kk = 0; kk < e_i.size(); ++kk) {
                    temp[kk] = e_i[kk] * dd;
                }
                projection = projection + temp;
            }
            admissible_directional = norm2(projection) > (target.get_radius() + source.get_radius());
        } else {
            admissible_directional = 2 * std::min(target.get_radius(), source.get_radius()) < eta * std::max((norm2(target.get_center() - source.get_center()) - target.get_radius() - source.get_radius()), 0.);
        }
        return (admissible_directional);
    }
};

// template <typename CoordinatePrecision>
// class vrai_directional final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
//   private:
//     std::vector<CoordinatePrecision> convection;
//     std::vector<CoordinatePrecision> points;

//   public:
//     vrai_directional(std::vector<CoordinatePrecision> &convection0) { convection = convection0; }
//     bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
//         std::vector<std::vector<CoordinatePrecision>> Perp(convection.size() - 1);
//         bool admissible_directional;
//         if (convection.size() == 2) {
//             std::vector<CoordinatePrecision> perp(2);
//             perp[0] = -convection[1];
//             perp[1] = convection[0];
//             for (int i = 0; i < perp.size(); ++i) {
//                 perp[i] = perp[i] / norm2(perp);
//             }
//             Perp[0] = perp;
//         } else if (convection.size() == 3) {
//             std::vector<CoordinatePrecision> perp1(3);
//             std::vector<CoordinatePrecision> perp2(3);
//             perp1[0] = -convection[0] * convection[2];
//             perp1[1] = -convection[1] * convection[2];
//             perp1[2] = convection[0] * convection[0] + convection[1] * convection[1];
//             perp2[0] = -convection[1];
//             perp2[1] = convection[0];
//             for (int i = 0; i < perp1.size(); ++i) {
//                 perp1[i] = perp1[i] / norm2(perp1);
//                 perp2[i] = perp2[i] / norm2(perp2);
//             }
//             Perp[0] = perp1;
//             Perp[1] = perp2;
//         }
//         auto direction = target.get_center() - source.get_center();
//         double test    = 0.0;
//         for (int k = 0; k < 3; ++k) {
//             test += convection[k] * direction[k];
//         }
//         if (test > 0) {
//             std::vector<CoordinatePrecision> projection(convection.size());
//             for (int i = 0; i < Perp.size(); ++i) {
//                 auto e_i  = Perp[i];
//                 double dd = 0.0;
//                 for (int kk = 0; kk < e_i.size(); ++kk) {
//                     dd += (e_i[kk] * direction[kk]);
//                 }
//                 std::vector<CoordinatePrecision> temp(e_i.size());
//                 for (int kk = 0; kk < e_i.size(); ++kk) {
//                     temp[kk] = e_i[kk] * dd;
//                 }
//                 projection = projection + temp;
//             }
//             // calcul rayon:
//             auto permutation = target.get_permutation();

//             for (int k = 0; k < target.size(); ++k) {
//                 std::vector<CoordinatePrecision> X(convection.size());
//                 for (int i = 0; i < convection.size(); ++i) {
//                     X[i] = points[convection.size() * permutation[k + cluster->get_offset()] + i];
//                 }
//                 for (int l = 0; l < convection.size(); ++l) {
//                     std::vector<CoordinatePrecision> Y(convection.size());
//                     for (int i = 0; i < convection.size(); ++i) {
//                         Y[i] = points[convection.size() * permutation[k + cluster->get_offset()] + i];
//                     }
//                 }
//             }
//             admissible_directional = norm2(projection) > (target.get_radius() + source.get_radius());
//         } else {
//             admissible_directional = 2 * std::min(target.get_radius(), source.get_radius()) < eta * std::max((norm2(target.get_center() - source.get_center()) - target.get_radius() - source.get_radius()), 0.);
//         }
//         return (admissible_directional);
//     }
// };

// template <typename CoordinatePrecision>
// class directional_heavy final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
//   private:
//     std::vector<CoordinatePrecision> convection;
//     std::vector<CoordinatePrecision> coordinate;

//   public:
//     directional_heavy(std::vector<CoordinatePrecision> &convection0) { convection = convection0; }
//     bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
//         std::vector<std::vector<CoordinatePrecision>> Perp(convection.size() - 1);
//         bool admissible_directional;
//         if (convection.size() == 2) {
//             std::vector<CoordinatePrecision> perp(2);
//             perp[0] = -convection[1];
//             perp[1] = convection[0];
//             for (int i = 0; i < perp.size(); ++i) {
//                 perp[i] = perp[i] / norm2(perp);
//             }
//             Perp[0] = perp;
//         } else if (convection.size() == 3) {
//             std::vector<CoordinatePrecision> perp1(3);
//             std::vector<CoordinatePrecision> perp2(3);
//             perp1[0] = -convection[0] * convection[2];
//             perp1[1] = -convection[1] * convection[2];
//             perp1[2] = convection[0] * convection[0] + convection[1] * convection[1];
//             perp2[0] = -convection[1];
//             perp2[1] = convection[0];
//             for (int i = 0; i < perp1.size(); ++i) {
//                 perp1[i] = perp1[i] / norm2(perp1);
//                 perp2[i] = perp2[i] / norm2(perp2);
//             }
//             Perp[0] = perp1;
//             Perp[1] = perp2;
//         }
//         auto direction = target.get_center() - source.get_center();
//         double test    = 0.0;
//         for (int k = 0; k < 3; ++k) {
//             test += convection[k] * direction[k];
//         }
//         if (test > 0) {
//             double target_radius = 0.0;
//             double source_radius = 0.0;
//             // calcul du radius
//             for (int k = 0; k < target.get_size(); ++k) {
//                 auto coord = target.m_tree_data.m_coordinates;
//                 std::vector<CoordinatePrecision> X;
//                 for (int jj = 0; jj < target.get_center().size(); ++jj) {
//                     X.push_back(target.m_tree_data->coordinates[target.get_center().size() * target.m_tree_data->permutation[k + target.get_offset()] + jj]);
//                 }
//                 for (int l = k + 1; l < target.size(); ++l) {
//                     std::vector<CoordinatePrecision> Y;
//                     for (int jj = 0; jj < target.get_center().size(); ++jj) {
//                         Y.push_back(target.m_tree_data->coordinates[target.get_center().size() * target.m_tree_data->permutation[k + target.get_offset()] + jj]);
//                     }
//                     std::vector<CoordinatePrecision> projete(target.get_center().size());
//                     for (int i = 0; i < Perp.size(); ++i) {
//                         auto e_i  = Perp[i];
//                         double dd = 0.0;
//                         std::vector<CoordinatePrecision> temp(e_i.size());
//                         auto direction = X - Y;
//                         for (int jj = 0; jj < e_i.size(); ++jj) {
//                             dd += (e_i[jj] * direction[jj]);
//                         }
//                         for (int kk = 0; kk < e_i.size(); ++kk) {
//                             temp[kk] = e_i[kk] * dd;
//                         }
//                         projete = projete + temp;
//                     }
//                     target_radius = std::max(norm2(projete), target_radius);
//                 }
//             }
//             for (int k = 0; k < source.get_size(); ++k) {
//                 auto coord = source.m_tree_data->m_coordinates;
//                 std::vector<CoordinatePrecision> X;
//                 for (int jj = 0; jj < source.get_center().size(); ++jj) {
//                     X.push_back(source.m_tree_data->coordinates[source.get_center().size() * source.m_tree_data->permutation[k + source.get_offset()] + jj]);
//                 }
//                 for (int l = k + 1; l < source.size(); ++l) {
//                     std::vector<CoordinatePrecision> Y;
//                     for (int jj = 0; jj < source.get_center().size(); ++jj) {
//                         Y.push_back(source.m_tree_data->coordinates[source.get_center().size() * source.m_tree_data->permutation[k + source.get_offset()] + jj]);
//                     }
//                     std::vector<CoordinatePrecision> projete(source.get_center().size());
//                     for (int i = 0; i < Perp.size(); ++i) {
//                         auto e_i  = Perp[i];
//                         double dd = 0.0;
//                         std::vector<CoordinatePrecision> temp(e_i.size());
//                         auto direction = X - Y;
//                         for (int jj = 0; jj < e_i.size(); ++jj) {
//                             dd += (e_i[jj] * direction[jj]);
//                         }
//                         for (int kk = 0; kk < e_i.size(); ++kk) {
//                             temp[kk] = e_i[kk] * dd;
//                         }
//                         projete = projete + temp;
//                     }
//                     source_radius = std::max(norm2(projete), source_radius);
//                 }
//             }
//             std::vector<CoordinatePrecision> projection(convection.size());
//             for (int i = 0; i < Perp.size(); ++i) {
//                 auto e_i  = Perp[i];
//                 double dd = 0.0;
//                 for (int kk = 0; kk < e_i.size(); ++kk) {
//                     dd += (e_i[kk] * direction[kk]);
//                 }
//                 std::vector<CoordinatePrecision> temp(e_i.size());
//                 for (int kk = 0; kk < e_i.size(); ++kk) {
//                     temp[kk] = e_i[kk] * dd;
//                 }
//                 projection = projection + temp;
//             }
//             admissible_directional = norm2(projection) > (target_radius + source_radius);
//         } else {
//             admissible_directional = 2 * std::min(target.get_radius(), source.get_radius()) < eta * std::max((norm2(target.get_center() - source.get_center()) - target.get_radius() - source.get_radius()), 0.);
//         }
//         return (admissible_directional);
//     }
// };

// template <typename CoordinatePrecision>
// class Leborne final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
//   private:
//     std::vector<CoordinatePrecision> b;
//     CoordinatePrecision alpha;
//     // il doit il y avoir une meilleure méthode mais la je vais faire du brute force
//   public:
//     Leborne(const std::vector<CoordinatePrecision> &w, const CoordinatePrecision &alpha0) : b(w),
//                                                                                             alpha(alpha0) {}
//     bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
//         // radius de target:
//         std::vector<CoordinatePrecision> target_center = target.get_center();
//         std::vector<CoordinatePrecision> source_center = source.get_center();
//         int space_dim                                  = target_center.size() - 1;
//         std::vector<std::vector<CoordinatePrecision>> b_perp(space_dim);
//         bool admissible = false;
//         if (target_center.size() == 2) {
//             std::vector<CoordinatePrecision> bperp(2, 0.0);
//             bperp[0]  = -b[1];
//             bperp[1]  = b[0];
//             b_perp[0] = bperp;
//         } else if (target_center.size() == 3) {
//             std::vector<CoordinatePrecision> bperp1(3, 0.0);
//             bperp1[0] = -b[0] * b[2];
//             bperp1[1] = -b[1] * b[2];
//             bperp1[2] = b[0] * b[0] + b[1] * b[1];
//             b_perp[0] = bperp1;
//             std::vector<CoordinatePrecision> bperp2(3, 0.0);
//             bperp2[0] = -b[1];
//             bperp2[1] = b[0];
//             b_perp[1] = bperp2;
//         }
//         for (int i = 0; i < b_perp.size(); ++i) {
//             auto bk   = b_perp[i];
//             double nr = norm2(bk);
//             for (int k = 0; k < bk.size(); ++k) {
//                 bk[k] = bk[k] / nr;
//             }
//         }
//         double nrb = norm2(b);
//         std::vector<CoordinatePrecision> btemp(b.size());
//         for (int k = 0; k < b.size(); ++k) {
//             btemp[k] = b[k] / nrb;
//         }
//         double rayon_target = 0.0;
//         for (int k = 0; k < target.get_size(); ++k) {
//             std::vector<CoordinatePrecision> u(target_center.size(), 0);
//             for (int p = 0; p < target_center.size(); p++) {
//                 u[p] = target.coordinates[target_center.size() * target->m_tree_data->permutation[k + target->get_offset()] + p] - target->get_center()[p];
//             }
//             double nrk = alpha * ddprod(btemp, u) * ddprod(btemp, u);
//             for (int l = 0; l < b_perp.size(); ++l) {
//                 auto bk = b_perp[k];
//                 nrk += ddprod(bk, u) * ddprod(bk, u);
//             }
//             rayon_target = std::max(rayon_target, sqrt(nrk));
//         }
//         double rayon_source = 0.0;
//         for (int k = 0; k < source.get_size(); ++k) {
//             std::vector<CoordinatePrecision> u(source_center.size(), 0);
//             for (int p = 0; p < source_center.size(); p++) {
//                 u[p] = source->coordinates[source_center.size() * source->m_tree_data->permutation[k + source->get_offset()] + p] - source->get_center()[p];
//             }
//             double nrk = alpha * ddprod(b, u) * ddprod(b, u);
//             for (int l = 0; l < b_perp.size(); ++l) {
//                 auto bk = b_perp[k];
//                 nrk += ddprod(bk, u) * ddprod(bk, u);
//             }
//             rayon_source = std::max(rayon_source, sqrt(nrk));
//         }
//         double distance = 1e15;
//         for (int k = 0; k < target.get_size(); ++k) {
//             std::vector<CoordinatePrecision> u(target_center.size(), 0);
//             for (int p = 0; p < source_center.size(); p++) {
//                 u[p] = source->coordinates[source_center.size() * target->m_tree_data->permutation[k + target->get_offset()] + p] - target->get_center()[p];
//             }
//             for (int l = 0; l < source.get_size(); ++l) {
//                 std::vector<CoordinatePrecision> v(source_center.size(), 0);
//                 for (int p = 0; p < source_center.size(); p++) {
//                     v[p] = source->coordinates[target_center.size() * source->m_tree_data->permutation[k + source->get_offset()] + p] - source->get_center()[p];
//                 }
//                 double distance_uv = alpha * (ddprod(b, u - v)) * (ddprod(b, u - v));
//                 for (int i = 0; i < b_perp.size(); ++i) {
//                     auto bk = b_perp[i];
//                     distance_uv += (ddprod(bk, u - v) * ddprod(bk, u - v));
//                 }
//                 distance_uv = sqrt(distance_uv);
//                 distance    = std::min(distance, distance_uv);
//             }
//         }
//         admissible = 2 * std::min(rayon_target, rayon_source) < eta * std::max(distance, 0.0);

//         return admissible;
//     }
// };

} // namespace htool
#endif
