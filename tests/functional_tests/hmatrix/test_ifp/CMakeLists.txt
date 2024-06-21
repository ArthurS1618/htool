add_executable(Test_air_quality test_air_quality.cpp)
target_link_libraries(Test_air_quality htool)
add_dependencies(build-tests-hmatrix-air-quality Test_air_quality)
add_test(Test_air_quality Test_air_quality)


add_executable(Test_freefem test_freefem.cpp)
target_link_libraries(Test_freefem htool)
add_dependencies(build-tests-hmatrix-air-quality Test_freefem)
add_test(Test_freefem Test_freefem)