#=============================================================================#
#=========================== Executables =====================================#
#=============================================================================#

add_executable(Test_sum_expression test_sum_expression.cpp)
target_link_libraries(Test_sum_expression htool)
add_dependencies(build-tests-sum-expression Test_sum_expression)
add_test(Test_sum_expression Test_sum_expression)