#ifndef HTOOL_BASIC_TYPES_MATRIX_HPP
#define HTOOL_BASIC_TYPES_MATRIX_HPP

#include "../misc/logger.hpp"
#include "../misc/misc.hpp"
#include "../wrappers/wrapper_blas.hpp"
#include "../wrappers/wrapper_lapack.hpp"
#include "vector.hpp"
#include <cassert>
#include <functional>
#include <iterator>

namespace htool {

template <typename T>
class Matrix {

  protected:
    int m_number_of_rows, m_number_of_cols;
    T *m_data;
    bool m_is_owning_data;

  public:
    Matrix() : m_number_of_rows(0), m_number_of_cols(0), m_data(nullptr) {}
    Matrix(int nbr, int nbc) : m_number_of_rows(nbr), m_number_of_cols(nbc), m_is_owning_data(true) {
        m_data = new T[nbr * nbc];
        std::fill_n(m_data, nbr * nbc, 0);
    }
    Matrix(const Matrix &rhs) : m_number_of_rows(rhs.m_number_of_rows), m_number_of_cols(rhs.m_number_of_cols), m_is_owning_data(true) {
        m_data = new T[rhs.m_number_of_rows * rhs.m_number_of_cols]();

        std::copy_n(rhs.m_data, rhs.m_number_of_rows * rhs.m_number_of_cols, m_data);
    }
    Matrix &operator=(const Matrix &rhs) {
        if (&rhs == this) {
            return *this;
        }
        if (m_number_of_rows * m_number_of_cols == rhs.m_number_of_cols * rhs.m_number_of_rows) {
            std::copy_n(rhs.m_data, m_number_of_rows * m_number_of_cols, m_data);
            m_number_of_rows = rhs.m_number_of_rows;
            m_number_of_cols = rhs.m_number_of_cols;
            m_is_owning_data = true;
        } else {
            m_number_of_rows = rhs.m_number_of_rows;
            m_number_of_cols = rhs.m_number_of_cols;
            if (m_is_owning_data)
                delete[] m_data;
            m_data = new T[m_number_of_rows * m_number_of_cols]();
            std::copy_n(rhs.m_data, m_number_of_rows * m_number_of_cols, m_data);
            m_is_owning_data = true;
        }
        return *this;
    }
    Matrix(Matrix &&rhs) : m_number_of_rows(rhs.m_number_of_rows), m_number_of_cols(rhs.m_number_of_cols), m_data(rhs.m_data), m_is_owning_data(rhs.m_is_owning_data) {
        rhs.m_data = nullptr;
    }

    Matrix &operator=(Matrix &&rhs) {
        if (this != &rhs) {
            if (m_is_owning_data)
                delete[] m_data;
            m_number_of_rows = rhs.m_number_of_rows;
            m_number_of_cols = rhs.m_number_of_cols;
            m_data           = rhs.m_data;
            m_is_owning_data = rhs.m_is_owning_data;
            rhs.m_data       = nullptr;
        }
        return *this;
    }

    ~Matrix() {
        if (m_data != nullptr && m_is_owning_data)
            delete[] m_data;
    }

    void operator=(const T &z) {
        if (m_number_of_rows * m_number_of_cols > 0)
            std::fill_n(m_data, m_number_of_rows * m_number_of_cols, z);
    }

    /////////////////////
    // ARTHUR: formatage matrice row major pour utiliser hmatrice*matrice
    // question Matrix ou void ? -> est ce qu'on a besoin de la matrice après l'opération ?
    std::vector<T> to_row_major() const {
        std::vector<T> M_row_major(this->nb_rows() * this->nb_cols(), 0.0);
        for (int k = 0; k < this->nb_rows(); ++k) {
            for (int l = 0; l < this->nb_cols(); ++l) {
                M_row_major[k * this->nb_cols() + l] = this->data()[k + this->nb_rows() * l];
            }
        }
        return M_row_major;
    }

    void to_col_major() {
        auto temp = this->data();
        for (int k = 0; k < m_number_of_rows; ++k) {
            for (int l = 0; l < m_number_of_cols; ++l) {
                this->data()[l * m_number_of_rows + k] = temp[k * m_number_of_cols + l];
            }
        }
    }
    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A(j,k)_ returns the entry of _A_ located
    jth row and kth column. Modification of the entries
    are allowed.
    */
    T &operator()(const int &j, const int &k) {
        return m_data[j + k * m_number_of_rows];
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A(j,k)_ returns the entry of _A_ located
    jth row and kth column. Modification of the
    entries are forbidden.
    */
    const T &operator()(const int &j, const int &k) const {
        return m_data[j + k * m_number_of_rows];
    }

    //! ### Access operator
    /*!
     */

    T *data() {
        return m_data;
    }
    T *data() const {
        return m_data;
    }

    void assign(int number_of_rows, int number_of_cols, T *ptr, bool owning_data) {
        if (m_number_of_rows * m_number_of_cols > 0 && m_is_owning_data)
            delete[] m_data;

        m_number_of_rows = number_of_rows;
        m_number_of_cols = number_of_cols;
        m_data           = ptr;
        m_is_owning_data = owning_data;
    }

    int nb_cols() const { return m_number_of_cols; }
    int nb_rows() const { return m_number_of_rows; }
    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_stridedslice(i,j,k)_ returns the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_. Modification forbidden
    */
    //// TRansposé -----------> a ne jamais utiliser c'est juste pour les tests

    Matrix transp(const Matrix &M) const {
        Matrix res(M.nb_cols(), M.nb_rows());
        for (int k = 0; k < M.nb_rows(); ++k) {
            for (int l = 0; l < M.nb_cols(); ++l) {
                res(l, k) = M(k, l);
            }
        }
        return res;
    }

    std::vector<T> get_stridedslice(int start, int length, int stride) const {
        std::vector<T> result;
        result.reserve(length);
        const T *pos = &m_data[start];
        for (int i = 0; i < length; i++) {
            result.push_back(*pos);
            pos += stride;
        }
        return result;
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_row(j)_ returns the jth row of _A_.
    */

    std::vector<T> get_row(int row) const {
        return this->get_stridedslice(row, m_number_of_cols, m_number_of_rows);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_col(j)_ returns the jth col of _A_.
    */

    std::vector<T> get_col(int col) const {
        return this->get_stridedslice(col * m_number_of_rows, m_number_of_rows, 1);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_stridedslice(i,j,k,a)_ puts a in the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_.
    */

    void set_stridedslice(int start, int length, int stride, const std::vector<T> &a) {
        assert(length == a.size());
        T *pos = &m_data[start];
        for (int i = 0; i < length; i++) {
            *pos = a[i];
            pos += stride;
        }
    }

    Matrix get_block(int row, int col, int row_offset, int col_offset) const {
        Matrix<T> res(row, col);
        auto temp = *this;
        if (row == m_number_of_rows && col == m_number_of_cols) {
            res = *this;
        } else {
            if (row == 1) {
                auto roww = this->get_row(row);
                std::vector<T> r(col);
                std::copy(roww.begin() + col_offset, roww.begin() + col + col_offset, r.begin());
                res.assign(row, col, r.data(), false);
            } else if (col == 1) {
                auto coll = this->get_col(col);
                std::vector<T> r(row);
                std::copy(coll.begin() + row_offset, coll.begin() + row + row_offset, r.begin());
                res.assign(row, col, r.data(), false);
            } else {
                for (int k = 0; k < row; ++k) {
                    for (int l = 0; l < col; ++l) {
                        // res(k, l) = m_data[(k + row_offset) + (l + col_offset) * this->nb_rows()];
                        // if (k + row_offset > this->nb_rows() or l + col_offset > this->nb_cols()) {
                        //     std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                        //     std::cout << row << ',' << row_offset << ',' << m_number_of_rows << '!' << col << ',' << col_offset << ',' << m_number_of_cols << std::endl;
                        // }
                        res(k, l) = temp(k + row_offset, l + col_offset);
                    }
                }
            }
        }
        return res;
    }

    void copy_submatrix(int row, int col, int row_offset, int col_offset, T *ptr) {
        // on va chercher les colonnes
        for (int l = 0; l < col; ++l) {
            std::copy(data() + (l + col_offset) * m_number_of_rows + row_offset, data() + (l + col_offset) * m_number_of_rows + row_offset + row, ptr + l * row);
        }
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_row(i,a)_ puts a in the ith row of _A_.
    */
    void set_row(int row, const std::vector<T> &a) {
        set_stridedslice(row, m_number_of_cols, m_number_of_rows, a);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_row(i,a)_ puts a in the row of _A_.
    */
    void set_col(int col, const std::vector<T> &a) {
        set_stridedslice(col * m_number_of_rows, m_number_of_rows, 1, a);
    }

    void set_size(int nr0, int nc0) {
        m_number_of_rows = nr0;
        m_number_of_cols = nc0;
    }

    //! ### Modifies the size of the matrix
    /*!
    Changes the size of the matrix so that
    the number of rows is set to _nbr_ and
    the number of columns is set to _nbc_.
    */
    void resize(const int nbr, const int nbc, T value = 0) {
        if (m_data != nullptr && m_is_owning_data) {
            delete[] m_data;
        }
        m_data           = new T[nbr * nbc];
        m_number_of_rows = nbr;
        m_number_of_cols = nbc;
        std::fill_n(m_data, nbr * nbc, value);
    }

    //! ### Matrix-scalar product
    /*!
     */

    friend Matrix
    operator*(const Matrix &A, const T &a) {
        Matrix R(A.m_number_of_rows, A.m_number_of_cols);
        for (int i = 0; i < A.m_number_of_rows; i++) {
            for (int j = 0; j < A.m_number_of_cols; j++) {
                R(i, j) = A(i, j) * a;
            }
        }
        return R;
    }
    friend Matrix operator*(const T &a, const Matrix &A) {
        return A * a;
    }

    //! ### Matrix sum
    /*!
     */

    Matrix operator+(const Matrix &A) const {
        assert(m_number_of_rows == A.m_number_of_rows && m_number_of_cols == A.m_number_of_cols);
        Matrix R(A.m_number_of_rows, A.m_number_of_cols);
        for (int i = 0; i < A.m_number_of_rows; i++) {
            for (int j = 0; j < A.m_number_of_cols; j++) {
                R(i, j) = m_data[i + j * m_number_of_rows] + A(i, j);
            }
        }
        return R;
    }

    //! ### Matrix -
    /*!
     */

    Matrix operator-(const Matrix &A) const {
        assert(m_number_of_rows == A.m_number_of_rows && m_number_of_cols == A.m_number_of_cols);
        Matrix R(A.m_number_of_rows, A.m_number_of_cols);
        for (int i = 0; i < A.m_number_of_rows; i++) {
            for (int j = 0; j < A.m_number_of_cols; j++) {
                R(i, j) = m_data[i + j * m_number_of_rows] - A(i, j);
            }
        }
        return R;
    }

    //! ### Matrix-std::vector product
    /*!
     */

    std::vector<T> operator*(const std::vector<T> &rhs) const {
        std::vector<T> lhs(m_number_of_rows);
        this->mvprod(rhs.data(), lhs.data(), 1);
        return lhs;
    }

    //! ### Matrix-Matrix product
    /*!
     */
    Matrix operator*(const Matrix &B) const {
        assert(m_number_of_cols == B.m_number_of_rows);
        Matrix R(m_number_of_rows, B.m_number_of_cols);
        this->mvprod(&(B.m_data[0]), &(R.m_data[0]), B.m_number_of_cols);
        return R;
    }

    //! ### Interface with blas gemm
    /*!
     */
    void mvprod(const T *in, T *out, const int &mu = 1) const {
        int nr  = m_number_of_rows;
        int nc  = m_number_of_cols;
        T alpha = 1;
        T beta  = 0;
        int lda = nr;

        if (mu == 1) {
            char n   = 'N';
            int incx = 1;
            int incy = 1;
            Blas<T>::gemv(&n, &nr, &nc, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
        } else {
            char transa = 'N';
            char transb = 'N';
            int M       = nr;
            int N       = mu;
            int K       = nc;
            int ldb     = nc;
            int ldc     = nr;
            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    void add_vector_product(char trans, T alpha, const T *in, T beta, T *out) const {
        int nr   = m_number_of_rows;
        int nc   = m_number_of_cols;
        int lda  = nr;
        int incx = 1;
        int incy = 1;
        Blas<T>::gemv(&trans, &nr, &nc, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
    }

    void add_matrix_product(char trans, T alpha, const T *in, T beta, T *out, int mu) const {
        int nr      = m_number_of_rows;
        int nc      = m_number_of_cols;
        char transa = trans;
        char transb = 'N';
        int lda     = nr;
        int M       = nr;
        int N       = mu;
        int K       = nc;
        int ldb     = nc;
        int ldc     = nr;
        if (transa != 'N') {
            M   = nc;
            N   = mu;
            K   = nr;
            ldb = nr;
            ldc = nc;
        }

        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
    }

    void add_matrix_product_row_major(char trans, T alpha, const T *in, T beta, T *out, int mu) const {
        int nr      = m_number_of_rows;
        int nc      = m_number_of_cols;
        char transa = 'N';
        char transb = 'T';
        int M       = mu;
        int N       = nr;
        int K       = nc;
        int lda     = mu;
        int ldb     = nr;
        int ldc     = mu;
        if (trans != 'N') {
            transb = 'N';
            N      = nc;
            K      = nr;
        }
        if (trans == 'C' && is_complex<T>()) {
            std::vector<T> conjugate_in(nr * mu);
            T conjugate_alpha = conj_if_complex<T>(alpha);
            T conjugate_beta  = conj_if_complex<T>(beta);
            std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return conj_if_complex<T>(c); });
            conj_if_complex<T>(out, nc * mu);
            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &conjugate_alpha, conjugate_in.data(), &lda, m_data, &ldb, &conjugate_beta, out, &ldc);
            conj_if_complex<T>(out, nc * mu);
            return;
        }
        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, m_data, &ldb, &beta, out, &ldc);
    }

    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_vector_product_symmetric(char, T alpha, const T *in, T beta, T *out, char UPLO, char) const {
        int nr  = m_number_of_rows;
        int lda = nr;

        if (nr) {
            int incx = 1;
            int incy = 1;
            Blas<T>::symv(&UPLO, &nr, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_vector_product_symmetric(char trans, T alpha, const T *in, T beta, T *out, char UPLO, char symmetry) const {
        int nr = m_number_of_rows;
        if (nr) {
            int lda  = nr;
            int incx = 1;
            int incy = 1;
            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symv(&UPLO, &nr, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                Blas<T>::hemv(&UPLO, &nr, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
                Blas<T>::symv(&UPLO, &nr, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &incx, &conjugate_beta, out, &incy);
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
            } else if (symmetry == 'H' && trans == 'T') {
                std::vector<T> conjugate_in(nr);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
                Blas<T>::hemv(&UPLO, &nr, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &incx, &conjugate_beta, out, &incy);
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });

            } else {
                htool::Logger::get_instance().log(Logger::LogLevel::ERROR, "Invalid arguments for add_vector_product_symmetric: " + std::string(1, trans) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE
                // throw std::invalid_argument("[Htool error] Invalid arguments for add_vector_product_symmetric");               // LCOV_EXCL_LINE
            }
        }
    }

    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric(char, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char) const {
        int nr  = m_number_of_rows;
        int lda = nr;

        if (nr) {
            char side = 'L';
            int M     = nr;
            int N     = mu;
            int ldb   = m_number_of_cols;
            int ldc   = nr;
            Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric(char trans, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char symmetry) const {
        int nr = m_number_of_rows;

        if (nr) {
            int lda   = nr;
            char side = 'L';
            int M     = nr;
            int N     = mu;
            int ldb   = m_number_of_cols;
            int ldc   = nr;

            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                Blas<T>::hemm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, m_number_of_cols * mu);
                Blas<T>::symm(&side, &UPLO, &M, &N, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, m_number_of_cols * mu);
            } else if (symmetry == 'H' && trans == 'T') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr * mu, out, [](const T &c) { return std::conj(c); });
                Blas<T>::hemm(&side, &UPLO, &M, &N, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                std::transform(out, out + nr * mu, out, [](const T &c) { return std::conj(c); });
            } else {
                htool::Logger::get_instance().log(Logger::LogLevel::ERROR, "Invalid arguments for add_matrix_product_symmetric: " + std::string(1, trans) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE

                // throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + symmetry + ")");                                            // LCOV_EXCL_LINE
            }
        }
    }

    //! ### Special mvprod with row major input and output
    /*!
     */
    void mvprod_row_major(const T *in, T *out, const int &mu, char transb, char op = 'N') const {
        int nr  = m_number_of_rows;
        int nc  = m_number_of_cols;
        T alpha = 1;
        T beta  = 0;
        int lda = nr;

        if (mu == 1) {
            int incx = 1;
            int incy = 1;
            Blas<T>::gemv(&op, &nr, &nc, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
        } else {
            lda         = mu;
            char transa = 'N';
            int M       = mu;
            int N       = nr;
            int K       = nc;
            int ldb     = nr;
            int ldc     = mu;

            if (op == 'T' || op == 'C') {
                transb = 'N';
                N      = nc;
                K      = nr;
            }

            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, m_data, &ldb, &beta, out, &ldc);
        }
    }

    // void add_matrix_product_row_major(T alpha, const T *in, T beta, T *out, const int &mu, char transb, char op = 'N') const {
    //     int nr = this->nr;
    //     int nc = this->nc;

    //     if (nr && nc) {
    //         if (mu == 1) {
    //             int lda  = nr;
    //             int incx = 1;
    //             int incy = 1;
    //             Blas<T>::gemv(&op, &nr, &nc, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
    //         } else {
    //             int lda     = mu;
    //             char transa = 'N';
    //             int M       = mu;
    //             int N       = nr;
    //             int K       = nc;
    //             int ldb     = nr;
    //             int ldc     = mu;

    //             if (op == 'T' || op == 'C') {
    //                 transb = 'N';
    //                 N      = nc;
    //                 K      = nr;
    //             }

    //             Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, mat, &ldb, &beta, out, &ldc);
    //         }
    //     }
    // }

    // see https://stackoverflow.com/questions/6972368/stdenable-if-to-conditionally-compile-a-member-function for why  Q template parameter
    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric_row_major(char, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char) const {
        int nr = m_number_of_rows;

        if (nr) {
            int lda   = nr;
            char side = 'R';
            int M     = mu;
            int N     = nr;
            int ldb   = mu;
            int ldc   = mu;

            Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric_row_major(char trans, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char symmetry) const {
        int nr = m_number_of_rows;

        if (nr) {
            int lda   = nr;
            char side = 'R';
            int M     = mu;
            int N     = nr;
            int ldb   = mu;
            int ldc   = mu;

            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'H' && trans == 'T') {
                Blas<T>::hemm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, m_number_of_cols * mu);
                Blas<T>::symm(&side, &UPLO, &M, &N, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, m_number_of_cols * mu);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, m_number_of_cols * mu);
                Blas<T>::hemm(&side, &UPLO, &M, &N, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, m_number_of_cols * mu);
            } else {
                htool::Logger::get_instance().log(Logger::LogLevel::ERROR, "Invalid arguments for add_matrix_product_symmetric_row_major: " + std::string(1, trans) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE
                // throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + symmetry + ")"); // LCOV_EXCL_LINE
            }
        }
    }

    void scale(T alpha) {
        std::transform(m_data, m_data + m_number_of_rows * m_number_of_cols, m_data, std::bind(std::multiplies<T>(), std::placeholders::_1, alpha));
    }

    //! ### Looking for the entry of maximal modulus
    /*!
    Returns the number of row and column of the entry
    of maximal modulus in the matrix _A_.
    */
    friend std::pair<int, int> argmax(const Matrix<T> &M) {
        int p = std::max_element(M.data(), M.data() + M.nb_cols() * M.nb_rows(), [](T a, T b) { return std::abs(a) < std::abs(b); }) - M.data();
        return std::pair<int, int>(p % M.m_number_of_rows, (int)p / M.m_number_of_rows);
    }

    //! ### Looking for the entry of maximal modulus
    /*!
    Save a Matrix in a file (bytes)
    */
    int matrix_to_bytes(const std::string &file) {
        std::ofstream out(file, std::ios::out | std::ios::binary | std::ios::trunc);

        if (!out) {
            std::cout << "Cannot open file." << std::endl; // LCOV_EXCL_LINE
            return 1;                                      // LCOV_EXCL_LINE
        }
        int rows = m_number_of_rows;
        int cols = m_number_of_cols;
        out.write((char *)(&rows), sizeof(int));
        out.write((char *)(&cols), sizeof(int));
        out.write((char *)m_data, rows * cols * sizeof(T));

        out.close();
        return 0;
    }

    //! ### Looking for the entry of maximal modulus
    /*!
    Load a matrix from a file (bytes)
    */
    int bytes_to_matrix(const std::string &file) {
        std::ifstream in(file, std::ios::in | std::ios::binary);

        if (!in) {
            std::cout << "Cannot open file." << std::endl; // LCOV_EXCL_LINE
            return 1;                                      // LCOV_EXCL_LINE
        }

        int rows = 0, cols = 0;
        in.read((char *)(&rows), sizeof(int));
        in.read((char *)(&cols), sizeof(int));
        if (m_number_of_rows != 0 && m_number_of_cols != 0 && m_is_owning_data)
            delete[] m_data;
        m_data           = new T[rows * cols];
        m_number_of_rows = rows;
        m_number_of_cols = cols;
        m_is_owning_data = true;
        in.read((char *)&(m_data[0]), rows * cols * sizeof(T));

        in.close();
        return 0;
    }

    int print(std::ostream &os, const std::string &delimiter) const {
        int rows = m_number_of_rows;

        if (m_number_of_cols > 0) {
            for (int i = 0; i < rows; i++) {
                std::vector<T> row = this->get_row(i);
                std::copy(row.begin(), row.end() - 1, std::ostream_iterator<T>(os, delimiter.c_str()));
                os << row.back();
                os << '\n';
            }
        }
        return 0;
    }

    int csv_save(const std::string &file, const std::string &delimiter = ",") const {
        std::ofstream os(file);
        try {
            if (!os) {
                htool::Logger::get_instance().log(Logger::LogLevel::WARNING, "Cannot create file " + file); // LCOV_EXCL_LINE
                // throw std::string("Cannot create file " + file);
            }
        } catch (std::string const &error) {
            std::cerr << error << std::endl;
            return 1;
        }

        this->print(os, delimiter);

        os.close();
        return 0;
    }

    // friend Matrix LU(const Matrix &M) {
    //     Matrix mat = M;
    //     for (int i = 1; i < M.nb_cols(); ++i) {
    //         for (int j = 0; j < M.nb_cols(); ++j) {
    //             if (std::abs(mat(i, j)) < 1e-15) {
    //                 mat(i, j) = mat(i, j) / mat(j, j);
    //                 for (int k = j + 1; k < M.nb_cols(); ++k) {
    //                     if (std::abs(mat(i, k)) < 1e-15) {
    //                         mat(i, k) = mat(i, k) - mat(i, j) * mat(j, k);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     return mat;
    // }
    /// trunc row --> return the k first row
    // Matrix trunc_row(int k) {
    //     Matrix<T> res(k, m_number_of_cols);
    //     for (int l = 0; l < m_number_of_cols; ++l) {
    //         std::vector<T> el(k)
    //     }
    //     return res;
    // }

    // // trunc col -------> crazy fast
    // friend Matrix trunc_col(const int &l)  {
    //     Matrix<T> res(m_number_of_rows, l);
    //     std::copy(data.begin(), data.begin() + l * m_number_of_rows, res.data().begin());
    //     return res;
    // }

    ///// extraire les k-1 premières colonnes
    /* Attention col(10) renvoie les 10 premières col0, col1, ..., col9*/
    Matrix<T> trunc_col(const int &l) {
        Matrix<T> res(m_number_of_rows, l);
        std::copy(this->data(), this->data() + l * m_number_of_rows, res.data());
        return res;
    }

    // extraire les k-1 premières lignes   -> vraiment pas dingue , plus interresant que get_çblock si k << lda
    // Matrix<T> trunc_row(const int &k) {
    //     Matrix<T> res(k, m_number_of_cols);
    //     for (int l = 0; l < k; ++l) {
    //         std::vector<T> el(m_number_of_rows);
    //         el[l] = 1;
    //         std::vector<T> rowtemp(m_number_of_cols);
    //         this->add_vector_product('T', 1.0, el.data(), 1.0, rowtemp.data());
    //         std::copy(rowtemp.begin(), rowtemp.begin() + m_number_of_cols, res.data() + l * m_number_of_cols);
    //     }
    //     return res;
    // }

    Matrix<T> trunc_row(const int &k) {
        Matrix<T> res(k, m_number_of_cols);
        for (int l = 0; l < k; ++l) {
            res.set_row(l, this->get_row(l));
        }
        return res;
    }
};

// template <typename T>
// Matrix<T> trunc_row(const Matrix<T> *M, const int &k) {
//     Matrix<T> res(k, M->nb_cols());
//     for (int l = 0; l < k; ++l) {
//         std::vector<T> el(M->nb_rows());
//         el[l] = 1;
//         std::vector<T> rowtemp(M->nb_cols());
//         M->add_vector_product('T', 1.0, el.data(), 1.0, rowtemp.data());
//         std::copy(rowtemp.begin(), rowtemp.begin() + M->nb_cols(), res.data() + l * M->nb_cols());
//     }
//     return res;
// }

// trunc col -------> crazy fast

/// concaténation
// V1 V2 -> (V1^T, V2^T)
template <typename T>
Matrix<T> conc_row_T(const Matrix<T> &A, const Matrix<T> &B) {
    Matrix<T> row_conc(A.nb_cols(), A.nb_rows() + B.nb_rows());
    for (int i = 0; i < A.nb_rows(); ++i) {
        std::vector<T> ei(A.nb_rows(), 0);
        ei[i] = 1.0;
        std::vector<T> rowtemp(A.nb_cols()); // colonne of conc
        A.add_vector_product('T', 1.0, ei.data(), 1.0, rowtemp.data());
        std::copy(rowtemp.begin(), rowtemp.begin() + A.nb_cols(), row_conc.data() + i * A.nb_cols());
    }
    for (int i = 0; i < B.nb_rows(); ++i) {
        std::vector<T> ei(B.nb_rows(), 0);
        ei[i] = 1.0;
        std::vector<T> rowtemp(B.nb_cols());
        B.add_vector_product('T', 1.0, ei.data(), 1.0, rowtemp.data());
        std::copy(rowtemp.begin(), rowtemp.begin() + B.nb_cols(), row_conc.data() + (i + A.nb_rows()) * B.nb_cols());
    }

    return row_conc;
}

// U1 U2 -> (U1, U2)
template <typename T>
Matrix<T> conc_col(const Matrix<T> &A, const Matrix<T> &B) {
    Matrix<T> col_conc(A.nb_rows(), A.nb_cols() + B.nb_cols());
    for (int j = 0; j < A.nb_cols(); ++j) {
        std::copy(A.data() + j * A.nb_rows(), A.data() + (j + 1) * A.nb_rows(), col_conc.data() + j * A.nb_rows());
    }
    for (int j = 0; j < B.nb_cols(); ++j) {
        std::copy(B.data() + j * B.nb_rows(), B.data() + (j + 1) * B.nb_rows(), col_conc.data() + (j + A.nb_cols()) * B.nb_rows());
    }
    return col_conc;
}
// template <typename T>
// std::vector<T> get_col(const Matrix<T> *M, const int &l) {
//     std::vector<T> col(M->m_number_of_rows, 0.0);
//     std::copy(M->data.begin() + l * M->m_number_of_rows, M->data.begin() + (l + 1) * M->m_number_of_rows, col.data().begin());
// }

// template <typename T>
// Matrix<T> get_range_col(const Matrix<T> *M, const int &size_l, const int &begin, const int &end) {
//     Matrix<T> res(M->m_number_of_rows, size_l);
//     std::copy(M->data.begin() + begin * M->m_number_of_rows, M->data.begin() + (end + 1) * M->m_number_of_rows, res.data().begin());
// }

//////////////////// ARTHUR : A = PLU avec LAPACK
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

/// A =SVD -> lapack ------------> je le fait pour renvoyer S et écrire U et V sur des matrices d'entrées !!!!! MARCHE QUE AVEC DES MATRICES CARRE sinon il faut changer lda, ldb etc
template <typename T>
std::vector<T> compute_svd(const Matrix<T> &A, Matrix<T> &U, Matrix<T> &V) {
    int N = A.nb_cols();
    std::vector<T> S(N, 0.0);
    int info;
    int lwork = 10 * N;
    std::vector<T> rwork(lwork);
    std::vector<T> work(lwork);
    Lapack<T>::gesvd("A", "A", &N, &N, A.data(), &N, S.data(), U.data(), &N, V.data(), &N, work.data(), &lwork, rwork.data(), &info);
    return S;
}
//! ### Computation of the Frobenius norm
/*!
Computes the Frobenius norm of the input matrix _A_.
*/
template <typename T>
double normFrob(const Matrix<T> &A) {
    double norm = 0;
    for (int j = 0; j < A.nb_rows(); j++) {
        for (int k = 0; k < A.nb_cols(); k++) {
            norm = norm + std::pow(std::abs(A(j, k)), 2);
        }
    }
    return sqrt(norm);
}

template <typename T>
Matrix<T> LU(const Matrix<T> &M) {
    Matrix<T> mat = M;
    int n         = mat.nb_cols();
    for (int j = 0; j < n; ++j) {
        for (int i = j + 1; i < n; ++i) {
            mat(i, j) = mat(i, j) / mat(j, j);
            for (int k = j + 1; k < n; ++k) {
                mat(i, k) = mat(i, k) - mat(i, j) * mat(j, k);
            }
        }
    }
    return mat;
}

template <typename T>
std::pair<Matrix<T>, Matrix<T>> get_lu(const Matrix<T> &M) {
    Matrix<T> L(M.nb_rows(), M.nb_cols());
    Matrix<T> U(M.nb_rows(), M.nb_cols());
    for (int k = 0; k < M.nb_rows(); ++k) {
        L(k, k) = 1.0, U(k, k) = M(k, k);
        for (int l = 0; l < k; ++l) {
            L(k, l) = M(k, l);
        }
        for (int l = k + 1; l < M.nb_rows(); ++l) {
            U(k, l) = M(k, l);
        }
    }
    std::pair<Matrix<T>, Matrix<T>> res;
    res.first  = L;
    res.second = U;
    return res;
}

template <typename CoefficientPrecision>
void ACA(const Matrix<CoefficientPrecision> &A, const int &offset_t, const int &offset_s, const CoefficientPrecision &epsilon, const int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) {

    int n1, n2;
    int i1;
    int i2;
    // const double *x1;

    if (offset_t >= offset_s) {

        n1 = A.nb_rows();
        n2 = A.nb_cols();
        i1 = offset_t;
        i2 = offset_s;
        // x1        = xt;
        // cluster_1 = &t;
    } else {
        n1 = A.nb_cols();
        n2 = A.nb_rows();
        i1 = offset_s;
        i2 = offset_t;
        // x1        = xs;
        // cluster_1 = &s;
    }

    //// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
    // double dist = 1e30;
    int I1 = 0;
    // for (int i = 0; i < n1; i++) {
    //     double aux_dist = std::sqrt(std::inner_product(x1 + (cluster_1->get_space_dim() * i1[i]), x1 + (cluster_1->get_space_dim() * i1[i]) + cluster_1->get_space_dim(), cluster_1->get_ctr().begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }));

    //     if (dist > aux_dist) {
    //         dist = aux_dist;
    //         I1   = i;
    //     }
    // }
    // Partial pivot
    int I2      = 0;
    int q       = 0;
    int reqrank = rank;
    std::vector<std::vector<CoefficientPrecision>> uu, vv;
    std::vector<bool> visited_1(n1, false);
    std::vector<bool> visited_2(n2, false);

    underlying_type<CoefficientPrecision> frob = 0;
    underlying_type<CoefficientPrecision> aux  = 0;

    underlying_type<CoefficientPrecision> pivot, tmp;
    CoefficientPrecision coef;
    int incx(1), incy(1);
    std::vector<CoefficientPrecision> u1(n2), u2(n1);
    // Either we have a required rank
    // Or it is negative and we have to check the relative error between two iterations.
    // But to do that we need a least two iterations.
    while (((reqrank > 0) && (q < std::min(reqrank, std::min(n1, n2)))) || ((reqrank < 0) && (q == 0 || sqrt(aux / frob) > epsilon))) {

        // Next current rank
        q += 1;

        if (q * (n1 + n2) > (n1 * n2)) { // the next current rank would not be advantageous
            q = -1;
            break;
        } else {
            std::fill(u1.begin(), u1.end(), CoefficientPrecision(0));
            if (offset_t >= offset_s) {
                // u1 = A.get_block(1, n2, i1 + I1, i2);
                auto uu = A.get_row(i1 + I1);
                std::copy(uu.begin() + i2, uu.begin() + i2 + n2, u1.begin());
            } else {
                auto uu = A.get_col(i1 + I1);
                std::copy(uu.begin() + i2, uu.begin() + i2 + n2, u1.begin());
                // A.copy_submatrix(n2, 1, i2, i1 + I1, u1.data());
            }

            for (int j = 0; j < uu.size(); j++) {
                coef = -uu[j][I1];
                Blas<CoefficientPrecision>::axpy(&(n2), &(coef), vv[j].data(), &incx, u1.data(), &incy);
            }

            pivot = 0.;
            tmp   = 0;
            for (int k = 0; k < n2; k++) {
                if (visited_2[k])
                    continue;
                tmp = std::abs(u1[k]);
                if (tmp < pivot)
                    continue;
                pivot = tmp;
                I2    = k;
            }
            visited_1[I1]              = true;
            CoefficientPrecision gamma = CoefficientPrecision(1.) / u1[I2];

            //==================//
            // Look for a line
            if (std::abs(u1[I2]) > 1e-15) {
                std::fill(u2.begin(), u2.end(), CoefficientPrecision(0));
                if (offset_t >= offset_s) {
                    // A.copy_submatrix(n1, 1, i1, i2 + I2, u2.data());
                    auto vv = A.get_col(i2 + I2);
                    std::copy(vv.begin() + i1, vv.begin() + i1 + n1, u2.begin());
                } else {
                    auto vv = A.get_row(i2 + I2);
                    std::copy(vv.begin() + i1, vv.begin() + i1 + n1, u2.begin());
                    // A.copy_submatrix(1, n1, i2 + I2, i1, u2.data());
                }
                for (int k = 0; k < uu.size(); k++) {
                    coef = -vv[k][I2];
                    Blas<CoefficientPrecision>::axpy(&(n1), &(coef), uu[k].data(), &incx, u2.data(), &incy);
                }
                u2 *= gamma;
                pivot = 0.;
                tmp   = 0;
                for (int k = 0; k < n1; k++) {
                    if (visited_1[k])
                        continue;
                    tmp = std::abs(u2[k]);
                    if (tmp < pivot)
                        continue;
                    pivot = tmp;
                    I1    = k;
                }
                visited_2[I2] = true;

                // Test if no given rank
                if (reqrank < 0) {
                    // Error estimator
                    CoefficientPrecision frob_aux = 0.;
                    aux                           = std::abs(Blas<CoefficientPrecision>::dot(&(n1), u2.data(), &incx, u2.data(), &incx)) * std::abs(Blas<CoefficientPrecision>::dot(&(n2), u1.data(), &incx, u1.data(), &incx));

                    // aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
                    for (int j = 0; j < uu.size(); j++) {
                        frob_aux += Blas<CoefficientPrecision>::dot(&(n2), u1.data(), &incx, vv[j].data(), &incy) * Blas<CoefficientPrecision>::dot(&(n1), u2.data(), &(incx), uu[j].data(), &(incy));
                    }
                    // frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
                    frob += aux + 2 * std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
                                                           //==================//
                }
                // Matrix<T> M=A.get_submatrix(this->ir,this->ic);
                // uu.push_back(M.get_col(J));
                // vv.push_back(M.get_row(I)/M(I,J));
                // New cross added
                uu.push_back(u2);
                vv.push_back(u1);

            } else {
                q -= 1;
                if (q == 0) { // corner case where first row is zero, ACA fails, we build a dense block instead
                    q = -1;
                }
                htool::Logger::get_instance().log(Logger::LogLevel::WARNING, "ACA found a zero row in a " + std::to_string(A.nb_rows()) + "x" + std::to_string(A.nb_cols()) + " block. Final rank is " + std::to_string(q)); // LCOV_EXCL_LINE
                // std::cout << "[Htool warning] ACA found a zero row in a " + std::to_string(t.get_size()) + "x" + std::to_string(s.get_size()) + " block. Final rank is " + std::to_string(q) << std::endl;
                break;
            }
        }
    }

    // Final rank
    int rankk = q;
    if (rankk > 0) {
        U.resize(A.nb_rows(), rankk);
        V.resize(rankk, A.nb_cols());

        if (offset_t >= offset_s) {
            for (int k = 0; k < rankk; k++) {
                std::move(uu[k].begin(), uu[k].end(), U.data() + k * A.nb_rows());
                for (int j = 0; j < A.nb_cols(); j++) {
                    V(k, j) = vv[k][j];
                }
            }
        } else {
            for (int k = 0; k < rankk; k++) {
                std::move(vv[k].begin(), vv[k].end(), U.data() + k * A.nb_rows());
                for (int j = 0; j < A.nb_cols(); j++) {
                    V(k, j) = uu[k][j];
                }
            }
        }
    }
}
template <typename T>
void plus(const int size, T *ptr_a, T *ptr_b, T *ptr_c) {
    std::transform(ptr_a, ptr_a + size, ptr_b, ptr_c, [](double a, double b) { return a + b; });
}
template <typename T>
void moins(const int size, T *ptr_a, T *ptr_b, T *ptr_c) {
    std::transform(ptr_a, ptr_a + size, ptr_b, ptr_c, [](double a, double b) { return a - b; });
}

} // namespace htool

#endif
