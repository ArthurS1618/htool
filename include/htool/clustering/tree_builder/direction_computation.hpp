#ifndef HTOOL_CLUSTERING_TREE_BUILDER_DIRECTION_COMPUTATION_HPP
#define HTOOL_CLUSTERING_TREE_BUILDER_DIRECTION_COMPUTATION_HPP

#include "../../misc/evp.hpp"
#include "../../misc/logger.hpp"
#include "../cluster_node.hpp"

template <typename T>
T norme_directional(std::vector<T> x, std::vector<T> direction, T alpha) {
    std::vector<std::vector<T>> direction_perp;
    if (x.size() == 2) {
        std::vector<T> perp;
        perp[0] = -direction[1];
        perp[1] = direction[0];
        perp    = (1 / norm2(perp)) * perp;
        direction_perp.push_back(perp);
    } else if (x.size() == 3) {
        std::vector<T> perp1, perp2;
        perp1[0] = -direction[1];
        perp1[1] = direction[0];
        perp1    = (1 / norm2(perp1)) * perp1;
        direction_perp.push_back(perp1);
        perp2[0] = -direction[2] * direction[0];
        perp2[1] = -direction[2] * direction[1];
        perp2[2] = direction[0] * direction[0] + direction[1] * direction[1];
        perp2    = (1 / norm2(perp2)) * perp2;
        direction_perp.push_back(perp2);
    }
    double norme = alpha * dprod(direction, x) * dprod(direction, x);
    for (int k = 0; k < x.size() - 1; ++k) {
        norme += dprod(x, direction_perp[k]) * dprod(x, direction_perp[k]);
    }
    return sqrt(norme);
}
namespace htool {

template <typename T>
class ComputeLargestExtent {
  public:
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const, const T *const weights) {
        if (spatial_dimension != 2 && spatial_dimension != 3) {
            htool::Logger::get_instance().log(Logger::LogLevel::ERROR, "clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
            // throw std::logic_error("[Htool error] clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
        }

        Matrix<T> cov(spatial_dimension, spatial_dimension);
        std::vector<T> direction(spatial_dimension, 0);

        for (int j = 0; j < cluster->get_size(); j++) {
            std::vector<T> u(spatial_dimension, 0);
            for (int p = 0; p < spatial_dimension; p++) {
                u[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p] - cluster->get_center()[p];
            }

            for (int p = 0; p < spatial_dimension; p++) {
                for (int q = 0; q < spatial_dimension; q++) {
                    cov(p, q) += weights[permutation[j + cluster->get_offset()]] * u[p] * u[q];
                }
            }
        }
        if (spatial_dimension == 2) {
            direction = solve_EVP_2(cov);
        } else if (spatial_dimension == 3) {
            direction = solve_EVP_3(cov);
        }
        return direction;
    }
};

template <typename T>
class ComputeBoundingBox {
  public:
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const, const T *const) {

        // min max for each axis
        std::vector<T> min_point(spatial_dimension, std::numeric_limits<T>::max());
        std::vector<T> max_point(spatial_dimension, std::numeric_limits<T>::min());
        for (int j = 0; j < cluster->get_size(); j++) {
            std::vector<T> u(spatial_dimension, 0);
            for (int p = 0; p < spatial_dimension; p++) {
                if (min_point[p] > coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p]) {
                    min_point[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p];
                }
                if (max_point[p] < coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p]) {
                    max_point[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p];
                }
                u[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p] - cluster->get_center()[p];
            }
        }

        // Direction of largest extent
        T max_distance(std::numeric_limits<T>::min());
        int dir_axis = 0;
        for (int p = 0; p < spatial_dimension; p++) {
            if (max_distance < max_point[p] - min_point[p]) {
                max_distance = max_point[p] - min_point[p];
                dir_axis     = p;
            }
        }
        std::vector<T> direction(spatial_dimension, 0);
        direction[dir_axis] = 1;
        return direction;
    }
};

// j'arrive que a les mettre en dure : il faut changer b et alpha a chaques fois
template <typename T>
class Mydriection {
  public:
    static std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
        std::vector<T> direction(3, 0.0);
        double alpha = 0.00001;
        if (spatial_dimension != 2 && spatial_dimension != 3) {
            throw std::logic_error("[Htool error] clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
        }
        if (spatial_dimension == 3) {
            std::vector<T> convection(3, 0.0);
            // on prend b = 1,1
            convection[0] = 1.0;
            convection[1] = 1.0;
            std::vector<T> b1(3, 0.0);
            b1[0] = -convection[1];
            b1[1] = convection[0];
            std::vector<T> b2(3, 0.0);
            b2[0] = -convection[2] * convection[0];
            b2[1] = -convection[2] * convection[1];
            b2[2] = convection[0] * convection[0] + convection[1] * convection[1];
            for (int k = 0; k < 3; ++k) {
                b1[k]         = b1[k] * (1 / norm2(b1));
                b2[k]         = b2[k] * (1 / norm2(b2));
                convection[k] = convection[k] * (1 / norm2(convection));
            }
            double max_b      = 0.0;
            double max_b_perp = 0.0;
            for (int k = 0; k < cluster->get_size(); ++k) {
                std::vector<T> u(spatial_dimension, 0);
                for (int p = 0; p < spatial_dimension; p++) {
                    u[p] = coordinates[spatial_dimension * permutation[k + cluster->get_offset()] + p] - cluster->get_center()[p];
                }
                for (int l = 0; l < cluster->get_size(); ++l) {
                    std::vector<T> v(spatial_dimension, 0);
                    for (int p = 0; p < spatial_dimension; ++p) {
                        v[p] = coordinates[spatial_dimension * permutation[l + cluster->get_offset()] + p] - cluster->get_center()[p];
                    }
                    T nr_b     = alpha * abs(dprod(u - v, convection));
                    max_b      = std::max(nr_b, max_b);
                    T nr_bperp = abs(dprod(u - v, b1)) + abs(dprod(u - v, b2));
                    max_b_perp = std::max(nr_bperp, max_b_perp);
                }
            }
            if (max_b < max_b_perp) {
                for (int l = 0; l < spatial_dimension; ++l) {
                    direction[l] = (1 / norm2(b1 + b2)) * (b1[l] + b2[l]);
                }
            } else {
                direction = convection;
            }
        }
        return direction;
    }
};

template <typename T>
class constdir {
  private:
    std::vector<T> convection;

  public:
    constdir(const std::vector<T> &c) : convection(c) {}
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
        return convection;
    }
};

template <typename T>
class DirectionConvection {
  private:
    std::vector<T> convection;

  public:
    DirectionConvection(const std::vector<T> &c) : convection(c) {}
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
        std::vector<std::vector<T>> bases;
        bases.push_back(convection);
        if (convection.size() == 2) {
            std::vector<T> perp(2);
            perp[0] = -convection[1];
            perp[1] = convection[0];
            perp    = (1.0 / norm2(perp)) * perp;
            bases.push_back(perp);
        } else if (convection.size() == 3) {
            std::vector<T> perp1(3);
            perp1[0] = -convection[1];
            perp1[1] = convection[0];
            std::vector<T> perp2(3);
            perp2[0] = -convection[2] * convection[0];
            perp2[1] = -convection[2] * convection[1];
            perp2[2] = convection[0] * convection[0] + convection[1] * convection[1];
            perp1    = (1.0 / norm2(perp1)) * perp1;
            perp2    = (1.0 / norm2(perp2)) * perp2;
            bases.push_back(perp1);
            bases.push_back(perp2);
        }
        if (spatial_dimension != 2 && spatial_dimension != 3) {
            throw std::logic_error("[Htool error] clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
        }

        Matrix<T> cov(spatial_dimension, spatial_dimension);
        std::vector<T> direction(spatial_dimension, 0);

        for (int j = 0; j < cluster->get_size(); j++) {
            std::vector<T> v(spatial_dimension, 0);
            for (int p = 0; p < spatial_dimension; p++) {
                v[p] = coordinates[spatial_dimension * permutation[j + cluster->get_offset()] + p] - cluster->get_center()[p];
            }
            std::vector<T> u(bases.size() - 1);
            for (int i = 0; i < bases.size() - 1; +i) {
                T prod = dprod(bases[i + 1], v);
                u[i]   = bases[i + 1] * prod;
            }

            for (int p = 0; p < spatial_dimension - 1; p++) {
                for (int q = 0; q < spatial_dimension - 1; q++) {
                    cov(p, q) += weights[permutation[j + cluster->get_offset()]] * u[p] * u[q];
                }
            }
        }
        if (spatial_dimension == 2) {
            direction = solve_EVP_2(cov);
        } else if (spatial_dimension == 3) {
            direction = solve_EVP_3(cov);
        } else {
            throw std::logic_error("[Htool error] clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
        }
        return direction;
    }
};

template <typename T>
class FixedDirection {
  public:
    FixedDirection(const std::vector<T> &dir) : direction_(dir) {}
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
        return direction_;
    }

  private:
    std::vector<T> direction_;
};
template <typename T>
class dumb_direction_10 {
  private:
  public:
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
        std::vector<T> conv(3, 0.0);
        conv[0] = 1.0;
        std::vector<T> perp1(3, 0.0);
        std::vector<T> perp2(3, 0.0);
        perp1[0] = -conv[1];
        perp1[1] = conv[0];
        perp2[0] = -conv[0] * conv[2];
        perp2[1] = -conv[1] * conv[2];
        perp2[2] = conv[0] * conv[0] + conv[1] * conv[1];
        // perp1               = mult(perp1, 1.0 / norm2(perp1));
        // perp2               = mult(perp2, 1.0 / norm2(perp2));
        std::vector<T> perp = perp1 + perp2;
        // perp                = mult(perp, 1.0 / norm2(perp));
        return perp;
    }
};

template <typename T>
class dumb_direction_11 {
  public:
    std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
        std::vector<T> conv(3, 0.0);
        conv[0] = 1.0;
        conv[1] = 1.0;
        std::vector<T> perp1(3, 0.0);
        std::vector<T> perp2(3, 0.0);
        perp1[0] = -conv[1];
        perp1[1] = conv[0];
        perp2[0] = -conv[0] * conv[2];
        perp2[1] = -conv[1] * conv[2];
        perp2[2] = conv[0] * conv[0] + conv[1] * conv[1];
        // perp1               = mult(perp1, 1.0 / norm2(perp1));
        // perp2               = mult(perp2, 1.0 / norm2(perp2));
        std::vector<T> perp = perp1 + perp2;
        // perp                = mult(perp, 1.0 / norm2(perp));
        return perp;
    }
};
// template <typename T>
// class dumb_direction_factory {
//   private:
//     std::vector<T> convection_;

//   public:
//     dumb_direction_factory(const std::vector<T> &c) : convection_(c) {}

//     dumb_direction<T> create() const {
//         return dumb_direction<T>(convection_);
//     }
// };
// template <typename T>
// class ConvectionDirection {
//   private:
//     static std::vector<T> default_convection;
//     static T default_alpha;
//     const std::vector<T> &convection;
//     const T alpha;

//   public:
//     ConvectionDirection(std::vector<T> w = default_convection, T alpha0 = default_alpha) : convection(w), alpha(alpha0) {}

//     static std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
//         std::vector<T> direction(spatial_dimension);
//         if (spatial_dimension != 2 && spatial_dimension != 3) {
//             throw std::logic_error("[Htool error] clustering not define for spatial dimension !=2 and !=3"); // LCOV_EXCL_LINE
//         }
//         if (spatial_dimension == 3) {
//             std::vector<T> max_conv();
//             std::vector<T> min_conv();
//             std::vector<T> b1(3, 0.0);
//             b1[0] = -convection[1];
//             b1[1] = convection[0];
//             std::vector<T> b2(3, 0.0);
//             b2[0]             = -convection[2] * convection[0];
//             b2[1]             = -convection[2] * convection[1];
//             b2[2]             = convection[0] * convection[0] + convection[1] * convection[1];
//             b1                = (1 / norm2(b1)) * b1;
//             b2                = (1 / norm2(b2)) * b2;
//             double max_b      = 0.0;
//             double max_b_perp = 0.0;
//             for (int k = 0; k < cluster.get_size(); ++k) {
//                 std::vector<T> u(spatial_dimension, 0);
//                 for (int p = 0; p < spatial_dimension; p++) {
//                     u[p] = coordinates[spatial_dimension * permutation[k + cluster->get_offset()] + p] - cluster->get_center()[p];
//                 }
//                 for (int l = 0; l < cluster.get_size(); ++l) {
//                     std::vector<T> v(spatial_dimension, 0);
//                     for (int p = 0; p < spatial_dimension; ++p) {
//                         v[p] = coordinates[spatial_dimension * permutation[l + cluster->get_offset()] + p] - cluster->get_center()[p];
//                     }
//                     T nr_b          = alpha * abs(dprod(u - v, convection));
//                     max_conv        = max(nr_b, max_conv);
//                     double nr_bperp = abs(dprod(u - v, b1)) + abs(dprod(u - v, b2));
//                     min_conv        = max(nr_bperp, min_conv);
//                 }
//             }
//             if (max_conv > min_conv) {
//                 direction = (1 / norm2(b1 + b2)) * (b1 + b2);
//             } else {
//                 direction = convection;
//             }
//         }
//         return direction;
//     }
// };
// template <typename T>
// class ArthurDirection {
//   private:
//     std::vector<T> &convection;
//     T alpha;

//   public:
//     ArthurDirection(std::vector<T> w, T alpha0) : convection(w), alpha(alpha0) {}

//     static std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
//         std::vector<T> direction(spatial_dimension);
//         if (spatial_dimension != 2 && spatial_dimension != 3) {
//             throw std::logic_error("[Htool error] clustering not defined for spatial dimension !=2 and !=3");
//         }
//         if (spatial_dimension == 3) {
//             std::vector<T> max_conv(spatial_dimension, 0.0);
//             std::vector<T> min_conv(spatial_dimension, 0.0);
//             std::vector<T> b1(spatial_dimension, 0.0);
//             b1[0] = -convection[1];
//             b1[1] = convection[0];
//             std::vector<T> b2(spatial_dimension, 0.0);
//             b2[0]             = -convection[2] * convection[0];
//             b2[1]             = -convection[2] * convection[1];
//             b2[2]             = convection[0] * convection[0] + convection[1] * convection[1];
//             b1                = (1 / norm2(b1)) * b1;
//             b2                = (1 / norm2(b2)) * b2;
//             double max_b      = 0.0;
//             double max_b_perp = 0.0;
//             for (int k = 0; k < cluster->get_size(); ++k) {
//                 std::vector<T> u(spatial_dimension, 0);
//                 for (int p = 0; p < spatial_dimension; p++) {
//                     u[p] = coordinates[spatial_dimension * permutation[k + cluster->get_offset()] + p] - cluster->get_center()[p];
//                 }
//                 for (int l = 0; l < cluster->get_size(); ++l) {
//                     std::vector<T> v(spatial_dimension, 0);
//                     for (int p = 0; p < spatial_dimension; ++p) {
//                         v[p] = coordinates[spatial_dimension * permutation[l + cluster->get_offset()] + p] - cluster->get_center()[p];
//                     }
//                     T nr_b          = alpha * abs(dprod(u - v, default_convection));
//                     max_conv        = max(nr_b, max_conv);
//                     double nr_bperp = abs(dprod(u - v, b1)) + abs(dprod(u - v, b2));
//                     min_conv        = max(nr_bperp, min_conv);
//                 }
//             }
//             if (max_conv > min_conv) {
//                 direction = (1 / norm2(b1 + b2)) * (b1 + b2);
//             } else {
//                 direction = convection;
//             }
//         }
//         return direction;
//     }
// };
// template <typename T>
// class ConvectionDirection {
//   private:
//     std::vector<T> convection;
//     T alpha;

//   public:
//     ConvectionDirection(std::vector<T> w, T alpha0) : alpha(alpha0) { convection = w; }

//     std::vector<T> compute_direction(const Cluster<T> *cluster, const std::vector<int> &permutation, int spatial_dimension, const T *const coordinates, const T *const radii, const T *const weights) {
//         std::vector<T> direction(spatial_dimension);
//         std::vector<T> Convection(spatial_dimension, 0.0);
//         if (spatial_dimension != 2 && spatial_dimension != 3) {
//             throw std::logic_error("[Htool error] clustering not defined for spatial dimension !=2 and !=3");
//         }
//         if (spatial_dimension == 3) {
//             Convection[0] = 1.0;
//             std::vector<T> b1(spatial_dimension, 0.0);
//             b1[0] = -Convection[1];
//             b1[1] = Convection[0];
//             std::vector<T> b2(spatial_dimension, 0.0);
//             b2[0] = -Convection[2] * Convection[0];
//             b2[1] = -Convection[2] * Convection[1];
//             b2[2] = Convection[0] * Convection[0] + Convection[1] * Convection[1];
//             for (int l = 0; l < spatial_dimension; ++l) {
//                 b1[l] = (1 / norm2(b1)) * b1[l];
//                 b2[l] = (1 / norm2(b2)) * b2[l];
//             }
//             double max_b      = 0.0;
//             double max_b_perp = 0.0;
//             for (int k = 0; k < cluster->get_size(); ++k) {
//                 std::vector<T> u(spatial_dimension, 0);
//                 for (int p = 0; p < spatial_dimension; p++) {
//                     u[p] = coordinates[spatial_dimension * permutation[k + cluster->get_offset()] + p] - cluster->get_center()[p];
//                 }
//                 for (int l = 0; l < cluster->get_size(); ++l) {
//                     std::vector<T> v(spatial_dimension, 0);
//                     for (int p = 0; p < spatial_dimension; ++p) {
//                         v[p] = coordinates[spatial_dimension * permutation[l + cluster->get_offset()] + p] - cluster->get_center()[p];
//                     }
//                     T nr_b          = alpha * abs(dprod(u - v, convection));
//                     max_b           = std::max(nr_b, max_b);
//                     double nr_bperp = abs(dprod(u - v, b1)) + abs(dprod(u - v, b2));
//                     max_b_perp      = std::max(nr_bperp, max_b_perp);
//                 }
//             }
//             if (max_b > max_b_perp) {
//                 for (int l = 0; l < spatial_dimension; ++l) {
//                     direction[l] = (1 / norm2(b1 + b2)) * (b1[l] + b2[l]);
//                 }
//             } else {
//                 direction = convection;
//             }
//         }
//         return direction;
//     }
//};
} // namespace htool

#endif
