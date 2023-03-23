#ifndef HTOOL_GENERATOR_HPP
#define HTOOL_GENERATOR_HPP

#include <cassert>
#include <iterator>

namespace htool {

template <typename CoefficientPrecision>
class VirtualGenerator {

  public:
    virtual void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const = 0;
    // virtual void copy_submatrix(int M, int N, const int *rows, const int *cols, CoefficientPrecision *ptr) const = 0;

    VirtualGenerator(){};
    VirtualGenerator(const VirtualGenerator &)            = default;
    VirtualGenerator &operator=(const VirtualGenerator &) = default;
    VirtualGenerator(VirtualGenerator &&)                 = default;
    VirtualGenerator &operator=(VirtualGenerator &&)      = default;
    virtual ~VirtualGenerator(){};
};

template <typename CoefficientPrecision>
class MatrixGenerator : public VirtualGenerator<CoefficientPrecision> {
    int space_dim;
    Matrix<CoefficientPrecision> mat;

  public:
    // Constructor
    MatrixGenerator(const Matrix<CoefficientPrecision> &M) : mat(M) {}

    // Virtual function to overload
    CoefficientPrecision get_coef(const int &k, const int &j) const {
        return mat(k, j);
    }

    // Virtual function to overload
    void copy_submatrix(int M, int N, int rows, int cols, CoefficientPrecision *ptr) const override {
        // const int rr = rows.get_perm_data() ;  int* cc= cols.get_perm_data();
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                // cout<< "ici" << k<<','<< j << ',' <<rows[j]<< ',' << cols[k]<< ','<< mat.nb_rows()<< ',' << mat.nb_cols() << endl;
                ptr[j + M * k] = mat(j + rows, k + cols);
            }
        }
    }
    // Matrix vector product
    // std::vector<CoefficientPrecision> operator*(std::vector<CoefficientPrecision> a) {
    //     std::vector<CoefficientPrecision> result(nr, 0);
    //     for (int j = 0; j < nr; j++) {
    //         for (int k = 0; k < nc; k++) {
    //             result[j] += this->get_coef(j, k) * a[k]; }}
    //     return result;}

    // Frobenius norm
    double norm() {
        double norm = 0;
        for (int j = 0; j < mat.nb_rows(); j++) {
            for (int k = 0; k < mat.nb_cols(); k++) {
                norm += pow(this->get_coef(j, k), 2);
            }
        }
        return sqrt(norm);
    }
};

// template <typename CoefficientPrecision>
// class VirtualGeneratorWithPermutation : public VirtualGenerator<CoefficientPrecision> {

//   protected:
//     const std::vector<int> &m_target_permutation;
//     const std::vector<int> &m_source_permutation;

//   public:
//     VirtualGeneratorWithPermutation(const std::vector<int> &target_permutation, const std::vector<int> &source_permutation) : m_target_permutation(target_permutation), m_source_permutation(source_permutation) {}

//     virtual void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
//         copy_submatrix(M, N, m_target_permutation.data() + row_offset, m_source_permutation.data() + col_offset, ptr);
//     };

//     virtual void copy_submatrix(int M, int N, const int *rows, const int *cols, CoefficientPrecision *ptr) const = 0;
// };

} // namespace htool

#endif
