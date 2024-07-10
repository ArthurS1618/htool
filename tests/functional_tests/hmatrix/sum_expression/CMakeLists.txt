#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_sumexpr test_sum_expression.cpp)
target_link_libraries(Test_sumexpr htool)
add_dependencies(build-tests-sum-expression Test_sumexpr)
add_test(Test_sumexpr Test_sumexpr)