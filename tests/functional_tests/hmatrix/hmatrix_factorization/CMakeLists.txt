#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

set(types "double")
list(APPEND types "complex_double")

foreach(type ${types})
    add_executable(Test_hmatrix_factorization_${type} test_hmatrix_factorization_${type}.cpp)
    target_link_libraries(Test_hmatrix_factorization_${type} htool)
    add_dependencies(build-tests-hmatrix-factorization Test_hmatrix_factorization_${type})
    add_test(NAME Test_hmatrix_factorization_${type} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Test_hmatrix_factorization_${type})
    set_tests_properties(Test_hmatrix_factorization_${type} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=4)
endforeach()

#=== ARTHUR ===#
#== Je rajoute ca pour avoir un test ou on fait varier la taille en spécifiant en argv ==#

add_executable(Test_hmatrix_factorization_comparaison test_hmatrix_factorization_comparaison.cpp)
target_link_libraries(Test_hmatrix_factorization_comparaison htool)
add_dependencies(build-tests-hmatrix-factorization Test_hmatrix_factorization_comparaison)
add_test(Test_hmatrix_factorization_comparaison Test_hmatrix_factorization_comparaison)
