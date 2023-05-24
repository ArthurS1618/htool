#ifndef HTOOL_BASIC_TYPES_MATRIX_HPP
#define HTOOL_BASIC_TYPES_MATRIX_HPP

#include "../misc/misc.hpp"
#include "../wrappers/wrapper_blas.hpp"
#include "vector.hpp"
#include <cassert>
#include <functional>
#include <iterator>

namespace htool {

template <typename T>
class Matrix {

  protected:
    int nr, nc;
    T *mat;
    bool owning_data;

  public:
    Matrix() : nr(0), nc(0), mat(nullptr) {}
    Matrix(int nbr, int nbc) : nr(nbr), nc(nbc), owning_data(true) {
        this->mat = new T[nbr * nbc];
        std::fill_n(this->mat, nbr * nbc, 0);
    }
    Matrix(const Matrix &rhs) : nr(rhs.nr), nc(rhs.nc), owning_data(true) {
        mat = new T[rhs.nr * rhs.nc]();

        std::copy_n(rhs.mat, rhs.nr * rhs.nc, mat);
    }
    Matrix &operator=(const Matrix &rhs) {
        if (&rhs == this) {
            return *this;
        }
        if (this->nr * this->nc == rhs.nc * rhs.nr) {
            std::copy_n(rhs.mat, this->nr * this->nc, mat);
            this->nr          = rhs.nr;
            this->nc          = rhs.nc;
            this->owning_data = true;
        } else {
            this->nr = rhs.nr;
            this->nc = rhs.nc;
            if (owning_data)
                delete[] mat;
            mat = new T[this->nr * this->nc]();
            std::copy_n(rhs.mat, this->nr * this->nc, mat);
            this->owning_data = true;
        }
        return *this;
    }
    Matrix(Matrix &&rhs) : nr(rhs.nr), nc(rhs.nc), mat(rhs.mat), owning_data(rhs.owning_data) {
        rhs.mat = nullptr;
    }

    Matrix &operator=(Matrix &&rhs) {
        if (this != &rhs) {
            if (owning_data)
                delete[] this->mat;
            this->nr          = rhs.nr;
            this->nc          = rhs.nc;
            this->mat         = rhs.mat;
            this->owning_data = rhs.owning_data;
            rhs.mat           = nullptr;
        }
        return *this;
    }

    ~Matrix() {
        if (mat != nullptr && owning_data)
            delete[] mat;
    }

    void operator=(const T &z) {
        if (this->nr * this->nc > 0)
            std::fill_n(this->mat, this->nr * this->nc, z);
    }
    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A(j,k)_ returns the entry of _A_ located
    jth row and kth column. Modification of the entries
    are allowed.
    */
    T &operator()(const int &j, const int &k) {
        return this->mat[j + k * this->nr];
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A(j,k)_ returns the entry of _A_ located
    jth row and kth column. Modification of the
    entries are forbidden.
    */
    const T &operator()(const int &j, const int &k) const {
        return this->mat[j + k * this->nr];
    }

    //! ### Access operator
    /*!
     */

    T *data() {
        return this->mat;
    }
    T *data() const {
        return this->mat;
    }

    void assign(int nr, int nc, T *ptr, bool owning_data) {
        if (this->nr * this->nc > 0 && this->owning_data)
            delete[] this->mat;

        this->nr          = nr;
        this->nc          = nc;
        this->mat         = ptr;
        this->owning_data = owning_data;
    }

    int nb_cols() const { return nc; }
    int nb_rows() const { return nr; }
    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_stridedslice(i,j,k)_ returns the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_. Modification forbidden
    */
    //// TRansposé

    Matrix transp(const Matrix &M) {
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
        const T *pos = &mat[start];
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
        return this->get_stridedslice(row, this->nc, this->nr);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_col(j)_ returns the jth col of _A_.
    */

    std::vector<T> get_col(int col) const {
        return this->get_stridedslice(col * this->nr, this->nr, 1);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_stridedslice(i,j,k,a)_ puts a in the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_.
    */

    void set_stridedslice(int start, int length, int stride, const std::vector<T> &a) {
        assert(length == a.size());
        T *pos = &mat[start];
        for (int i = 0; i < length; i++) {
            *pos = a[i];
            pos += stride;
        }
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_row(i,a)_ puts a in the ith row of _A_.
    */
    void set_row(int row, const std::vector<T> &a) {
        set_stridedslice(row, this->nc, this->nr, a);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_row(i,a)_ puts a in the row of _A_.
    */
    void set_col(int col, const std::vector<T> &a) {
        set_stridedslice(col * this->nr, this->nr, 1, a);
    }

    void set_size(int nr0, int nc0) {
        this->nr = nr0;
        this->nc = nc0;
    }

    //! ### Modifies the size of the matrix
    /*!
    Changes the size of the matrix so that
    the number of rows is set to _nbr_ and
    the number of columns is set to _nbc_.
    */
    void resize(const int nbr, const int nbc, T value = 0) {
        if (mat != nullptr && owning_data) {
            delete[] mat;
        }
        mat = new T[nbr * nbc];
        nr  = nbr;
        nc  = nbc;
        std::fill_n(mat, nbr * nbc, value);
    }

    //! ### Matrix-scalar product
    /*!
     */

    friend Matrix
    operator*(const Matrix &A, const T &a) {
        Matrix R(A.nr, A.nc);
        for (int i = 0; i < A.nr; i++) {
            for (int j = 0; j < A.nc; j++) {
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
        assert(this->nr == A.nr && this->nc == A.nc);
        Matrix R(A.nr, A.nc);
        for (int i = 0; i < A.nr; i++) {
            for (int j = 0; j < A.nc; j++) {
                R(i, j) = this->mat[i + j * this->nr] + A(i, j);
            }
        }
        return R;
    }

    //! ### Matrix -
    /*!
     */

    Matrix operator-(const Matrix &A) const {
        assert(this->nr == A.nr && this->nc == A.nc);
        Matrix R(A.nr, A.nc);
        for (int i = 0; i < A.nr; i++) {
            for (int j = 0; j < A.nc; j++) {
                R(i, j) = this->mat[i + j * this->nr] - A(i, j);
            }
        }
        return R;
    }

    //! ### Matrix-std::vector product
    /*!
     */

    std::vector<T> operator*(const std::vector<T> &rhs) const {
        std::vector<T> lhs(this->nr);
        this->mvprod(rhs.data(), lhs.data(), 1);
        return lhs;
    }

    //! ### Matrix-Matrix product
    /*!
     */
    Matrix operator*(const Matrix &B) const {
        assert(this->nc == B.nr);
        Matrix R(this->nr, B.nc);
        this->mvprod(&(B.mat[0]), &(R.mat[0]), B.nc);
        return R;
    }

    //! ### Interface with blas gemm
    /*!
     */
    void mvprod(const T *in, T *out, const int &mu = 1) const {
        int nr  = this->nr;
        int nc  = this->nc;
        T alpha = 1;
        T beta  = 0;
        int lda = nr;

        if (mu == 1) {
            char n   = 'N';
            int incx = 1;
            int incy = 1;
            Blas<T>::gemv(&n, &nr, &nc, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
        } else {
            char transa = 'N';
            char transb = 'N';
            int M       = nr;
            int N       = mu;
            int K       = nc;
            int ldb     = nc;
            int ldc     = nr;
            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    void add_vector_product(char trans, T alpha, const T *in, T beta, T *out) const {
        int nr   = this->nr;
        int nc   = this->nc;
        int lda  = nr;
        int incx = 1;
        int incy = 1;
        Blas<T>::gemv(&trans, &nr, &nc, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
    }

    void add_matrix_product(char trans, T alpha, const T *in, T beta, T *out, int mu) const {
        int nr      = this->nr;
        int nc      = this->nc;
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

        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
    }

    void add_matrix_product_row_major(char trans, T alpha, const T *in, T beta, T *out, int mu) const {
        int nr      = this->nr;
        int nc      = this->nc;
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
            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &conjugate_alpha, conjugate_in.data(), &lda, mat, &ldb, &conjugate_beta, out, &ldc);
            conj_if_complex<T>(out, nc * mu);
            return;
        }
        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, mat, &ldb, &beta, out, &ldc);
    }

    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_vector_product_symmetric(char trans, T alpha, const T *in, T beta, T *out, char UPLO, char) const {
        int nr  = this->nr;
        int lda = nr;

        if (nr) {
            int incx = 1;
            int incy = 1;
            Blas<T>::symv(&UPLO, &nr, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_vector_product_symmetric(char trans, T alpha, const T *in, T beta, T *out, char UPLO, char symmetry) const {
        int nr = this->nr;
        if (nr) {
            int lda  = nr;
            int incx = 1;
            int incy = 1;
            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symv(&UPLO, &nr, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                Blas<T>::hemv(&UPLO, &nr, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
                Blas<T>::symv(&UPLO, &nr, &conjugate_alpha, mat, &lda, conjugate_in.data(), &incx, &conjugate_beta, out, &incy);
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
            } else if (symmetry == 'H' && trans == 'T') {
                std::vector<T> conjugate_in(nr);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
                Blas<T>::hemv(&UPLO, &nr, &conjugate_alpha, mat, &lda, conjugate_in.data(), &incx, &conjugate_beta, out, &incy);
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });

            } else {
                throw std::invalid_argument("[Htool error] Invalid arguments for add_vector_product_symmetric"); // LCOV_EXCL_LINE
            }
        }
    }

    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric(char trans, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char symmetry) const {
        int nr  = this->nr;
        int lda = nr;

        if (nr) {
            char side = 'L';
            int M     = nr;
            int N     = mu;
            int ldb   = nc;
            int ldc   = nr;
            Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric(char trans, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char symmetry) const {
        int nr = this->nr;

        if (nr) {
            int lda   = nr;
            char side = 'L';
            int M     = nr;
            int N     = mu;
            int ldb   = nc;
            int ldc   = nr;

            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                Blas<T>::hemm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, nc * mu);
                Blas<T>::symm(&side, &UPLO, &M, &N, &conjugate_alpha, mat, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, nc * mu);
            } else if (symmetry == 'H' && trans == 'T') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr * mu, out, [](const T &c) { return std::conj(c); });
                Blas<T>::hemm(&side, &UPLO, &M, &N, &conjugate_alpha, mat, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                std::transform(out, out + nr * mu, out, [](const T &c) { return std::conj(c); });
            } else {
                throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + symmetry + ")"); // LCOV_EXCL_LINE
            }
        }
    }

    //! ### Special mvprod with row major input and output
    /*!
     */
    void mvprod_row_major(const T *in, T *out, const int &mu, char transb, char op = 'N') const {
        int nr  = this->nr;
        int nc  = this->nc;
        T alpha = 1;
        T beta  = 0;
        int lda = nr;

        if (mu == 1) {
            int incx = 1;
            int incy = 1;
            Blas<T>::gemv(&op, &nr, &nc, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
        } else {
            int lda     = mu;
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

            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, mat, &ldb, &beta, out, &ldc);
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
        int nr = this->nr;

        if (nr) {
            int lda   = nr;
            char side = 'R';
            int M     = mu;
            int N     = nr;
            int ldb   = mu;
            int ldc   = mu;

            Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric_row_major(char trans, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char symmetry) const {
        int nr = this->nr;

        if (nr) {
            int lda   = nr;
            char side = 'R';
            int M     = mu;
            int N     = nr;
            int ldb   = mu;
            int ldc   = mu;

            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'H' && trans == 'T') {
                Blas<T>::hemm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, nc * mu);
                Blas<T>::symm(&side, &UPLO, &M, &N, &conjugate_alpha, mat, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, nc * mu);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, nc * mu);
                Blas<T>::hemm(&side, &UPLO, &M, &N, &conjugate_alpha, mat, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, nc * mu);
            } else {
                throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + symmetry + ")"); // LCOV_EXCL_LINE
            }
        }
    }

    void scale(T alpha) {
        std::transform(mat, mat + nr * nc, mat, std::bind(std::multiplies<T>(), std::placeholders::_1, alpha));
    }

    //! ### Looking for the entry of maximal modulus
    /*!
    Returns the number of row and column of the entry
    of maximal modulus in the matrix _A_.
    */
    friend std::pair<int, int> argmax(const Matrix<T> &M) {
        int p = std::max_element(M.data(), M.data() + M.nb_cols() * M.nb_rows(), [](T a, T b) { return std::abs(a) < std::abs(b); }) - M.data();
        return std::pair<int, int>(p % M.nr, (int)p / M.nr);
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
        int rows = this->nr;
        int cols = this->nc;
        out.write((char *)(&rows), sizeof(int));
        out.write((char *)(&cols), sizeof(int));
        out.write((char *)mat, rows * cols * sizeof(T));

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
        if (this->nr != 0 && this->nc != 0 && owning_data)
            delete[] mat;
        mat               = new T[rows * cols];
        this->nr          = rows;
        this->nc          = cols;
        this->owning_data = true;
        in.read((char *)&(mat[0]), rows * cols * sizeof(T));

        in.close();
        return 0;
    }

    int print(std::ostream &os, const std::string &delimiter) const {
        int rows = this->nr;

        if (this->nc > 0) {
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
                throw std::string("Cannot create file " + file);
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
};

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

} // namespace htool

#endif
