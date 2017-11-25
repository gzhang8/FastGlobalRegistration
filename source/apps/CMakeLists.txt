find_package(PCL 1.7 REQUIRED COMPONENTS io registration search features segmentation surface)

include_directories(${PCL_INCLUDE_DIRS})

add_executable(faGlobalRegistration faGlobalRegistration.cpp ../FastGlobalRegistration/app.cpp)
target_link_libraries(faGlobalRegistration
  ${Boost_LIBRARIES}

  ${PCL_LIBRARIES})
