#ifndef HTOOL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP
#define HTOOL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP

#include "../../basic_types/vector.hpp"
#include "../../clustering/cluster_node.hpp"
#include "../../matrix/matrix.hpp"
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
class Bande final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
  private:
    std::vector<CoordinatePrecision> convection;
    std::vector<int> coordinates;

  public:
    Bande(std::vector<CoordinatePrecision> &convection0, std::vector<int> coord) {
        convection  = convection0;
        coordinates = coord;
    }

    bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
        bool res;
        std::vector<std::vector<CoordinatePrecision>> bases;
        bases.push_back(convection);
        // base orthonormée
        std::vector<CoordinatePrecision> perp1(3);
        std::vector<CoordinatePrecision> perp2(3);
        perp1[0] = -convection[1];
        perp1[1] = convection[0];
        perp1    = mult(1.0 / norm2(perp1), perp1);
        perp2[0] = -convection[0] * convection[2];
        perp2[1] = -convection[1] * convection[2];
        perp2[2] = convection[0] * convection[0] + convection[1] * convection[1];
        perp2    = mult(1.0 / norm2(perp2), perp2);
        bases.push_back(perp1);
        bases.push_back(perp2);
        Matrix<CoordinatePrecision> cov_target(2, 2);
        Matrix<CoordinatePrecision> cov_source(2, 2);
        double target_radius = 0;
        double source_radius = 0;
        // calcul du rayon dans le plan perp :
        for (int j = 0; j < target.get_size(); ++j) {
            std::vector<CoordinatePrecision> v(3, 0);
            // x - center
            for (int p = 0; p < 3; ++p) {
                v[p] = coordinates[3 * target.get_permutation()[j + target.get_offset()] + p] - target.get_center()[p];
            }
            // projection(x - center)
            std::vector<CoordinatePrecision> u(2, 0.0);
            for (int i = 0; i < bases.size() - 1; ++i) {
                CoordinatePrecision prod = dprod(bases[i + 1], v);
                u[i]                     = prod;
            }
            // matrice de covariance
            for (int p = 0; p < 2; p++) {
                for (int q = 0; q < 2; q++) {
                    cov_target(p, q) += u[p] * u[q];
                }
            }
        }
        for (int j = 0; j < source.get_size(); j++) {
            std::vector<CoordinatePrecision> v(3, 0);
            // x - center *
            for (int p = 0; p < 3; p++) {
                v[p] = coordinates[3 * source.get_permutation()[j + source.get_offset()] + p] - source.get_center()[p];
                // std::cout << v[p] << std::endl;
            }
            // projection(x - center)
            std::vector<CoordinatePrecision> u(2, 0.0);
            for (int i = 0; i < bases.size() - 1; ++i) {
                CoordinatePrecision prod = dprod(bases[i + 1], v);
                u[i]                     = prod;
            }
            // matrice de covariance
            for (int p = 0; p < 2; p++) {
                for (int q = 0; q < 2; q++) {
                    cov_source(p, q) += u[p] * u[q];
                    // std::cout << cov_source(p, q);
                }
            }
        }
        std::vector<CoordinatePrecision> direction_t = solve_EVP_2(cov_target);
        std::vector<CoordinatePrecision> direction_s = solve_EVP_2(cov_source);
        source_radius                                = norm2(mult(direction_s[0], bases[1]) + mult(direction_s[1], bases[2]));
        target_radius                                = norm2(mult(direction_t[0], bases[1]) + mult(direction_t[1], bases[2]));
        CoordinatePrecision c                        = norm2(mult(dprod(target.get_center(), bases[1]), bases[1]) + mult(dprod(target.get_center(), bases[2]), bases[2]) - (mult(dprod(source.get_center(), bases[1]), bases[1]) + mult(dprod(source.get_center(), bases[2]), bases[2])));
        res                                          = 2 * target.get_radius() < eta * c;

        return res;
    }
};
} // namespace htool
#endif
