#ifndef HTOOL_LAPACK_HPP
#define HTOOL_LAPACK_HPP

#include "../misc/define.hpp"

#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
#    define HTOOL_LAPACK_F77(func) func
#else
#    define HTOOL_LAPACK_F77(func) func##_
#endif

#define HTOOL_GENERATE_EXTERN_LAPACK(C, T, U, SYM, ORT) \
    void HTOOL_LAPACK_F77(C##SYM##gv)(const int *, const char *, const char *, const int *, T *, const int *, T *, const int *, T *, T *, const int *, int *);

// const int *itype, const char *jobz, const char *UPLO, const int *n, T *A, const int *lda, T *B, const int *ldb, T *W, T *work, const int *lwork, U *, int *info)
#define HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(C, T, B, U)                                                                                                                                     \
    void HTOOL_LAPACK_F77(B##gesvd)(const char *, const char *, const int *, const int *, U *, const int *, U *, U *, const int *, U *, const int *, U *, const int *, int *);               \
    void HTOOL_LAPACK_F77(C##gesvd)(const char *, const char *, const int *, const int *, T *, const int *, U *, T *, const int *, T *, const int *, T *, const int *, U *, int *);          \
    void HTOOL_LAPACK_F77(B##ggev)(const char *, const char *, const int *, U *, const int *, U *, const int *, U *, U *, U *, U *, const int *, U *, const int *, U *, const int *, int *); \
    void HTOOL_LAPACK_F77(C##ggev)(const char *, const char *, const int *, T *, const int *, T *, const int *, T *, T *, T *, const int *, T *, const int *, T *, const int *, U *, int *); \
    void HTOOL_LAPACK_F77(B##sygv)(const int *, const char *, const char *, const int *, U *, const int *, U *, const int *, U *, U *, const int *, int *);                                  \
    void HTOOL_LAPACK_F77(C##hegv)(const int *, const char *, const char *, const int *, T *, const int *, T *, const int *, U *, T *, const int *, U *, int *);

#if !defined(PETSC_HAVE_BLASLAPACK)
#    ifndef _MKL_H_
#        ifdef __cplusplus
extern "C" {
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(z, std::complex<double>, d, double)
}
#        else
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(c, void, s, float)
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(z, void, d, double)
#        endif // __cplusplus
#    endif     // _MKL_H_
#endif

#ifdef __cplusplus
namespace htool {
/* Class: Lapack
 *
 *  A class that wraps some LAPACK routines for dense linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template <class K>
struct Lapack {
    /* Function: gesvd
     *  computes the singular value decomposition (SVD). */
    static void gesvd(const char *, const char *, const int *, const int *, K *, const int *, underlying_type<K> *, K *, const int *, K *, const int *, K *, const int *, underlying_type<K> *, int *);
    /* Function: ggev
     *  Computes the eigenvalues and (optionally) the eigenvectors of a nonsymmetric generalized eigenvalue problem. */
    static void ggev(const char *, const char *, const int *, K *, const int *, K *, const int *, K *, K *, K *, K *, const int *, K *, const int *, K *, const int *, underlying_type<K> *, int *);
    /* Function: gv
     *  Computes the eigenvalues and (optionally) the eigenvectors of a hermitian/symetric generalized eigenvalue problem. */
    static void gv(const int *, const char *, const char *, const int *, K *, const int *, K *, const int *, underlying_type<K> *, K *, const int *, underlying_type<K> *, int *);
};

#    define HTOOL_GENERATE_LAPACK_COMPLEX(C, T, B, U)                                                                                                                                                                                                           \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<U>::gesvd(const char *jobu, const char *jobvt, const int *m, const int *n, U *a, const int *lda, U *s, U *u, const int *ldu, U *vt, const int *ldvt, U *work, const int *lwork, U *, int *info) {                                    \
            HTOOL_LAPACK_F77(B##gesvd)                                                                                                                                                                                                                          \
            (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<T>::gesvd(const char *jobu, const char *jobvt, const int *m, const int *n, T *a, const int *lda, U *s, T *u, const int *ldu, T *vt, const int *ldvt, T *work, const int *lwork, U *rwork, int *info) {                               \
            HTOOL_LAPACK_F77(C##gesvd)                                                                                                                                                                                                                          \
            (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);                                                                                                                                                                         \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<U>::ggev(const char *jobvl, const char *jobvr, const int *n, U *a, const int *lda, U *b, const int *ldb, U *alphar, U *alphai, U *beta, U *vl, const int *ldvl, U *vr, const int *ldvr, U *work, const int *lwork, U *, int *info) { \
            HTOOL_LAPACK_F77(B##ggev)                                                                                                                                                                                                                           \
            (jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);                                                                                                                                                     \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<T>::ggev(const char *jobvl, const char *jobvr, const int *n, T *a, const int *lda, T *b, const int *ldb, T *alpha, T *, T *beta, T *vl, const int *ldvl, T *vr, const int *ldvr, T *work, const int *lwork, U *rwork, int *info) {   \
            HTOOL_LAPACK_F77(C##ggev)                                                                                                                                                                                                                           \
            (jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);                                                                                                                                                       \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<U>::gv(const int *itype, const char *jobz, const char *uplo, const int *n, U *a, const int *lda, U *b, const int *ldb, U *w, U *work, const int *lwork, U *, int *info) {                                                            \
            HTOOL_LAPACK_F77(B##sygv)                                                                                                                                                                                                                           \
            (itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info);                                                                                                                                                                                       \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<T>::gv(const int *itype, const char *jobz, const char *uplo, const int *n, T *a, const int *lda, T *b, const int *ldb, U *w, T *work, const int *lwork, U *rwork, int *info) {                                                       \
            HTOOL_LAPACK_F77(C##hegv)                                                                                                                                                                                                                           \
            (itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, info);                                                                                                                                                                                \
        }

HTOOL_GENERATE_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_LAPACK_COMPLEX(z, std::complex<double>, d, double)
} // namespace htool
#endif // __cplusplus
#endif // HTOOL_LAPACK_HPP
