# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug

# Include any dependencies generated for this target.
include CMakeFiles/test_graph.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_graph.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_graph.dir/flags.make

CMakeFiles/test_graph.dir/tests/test_graph.cpp.o: CMakeFiles/test_graph.dir/flags.make
CMakeFiles/test_graph.dir/tests/test_graph.cpp.o: ../tests/test_graph.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_graph.dir/tests/test_graph.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_graph.dir/tests/test_graph.cpp.o -c /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/tests/test_graph.cpp

CMakeFiles/test_graph.dir/tests/test_graph.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_graph.dir/tests/test_graph.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/tests/test_graph.cpp > CMakeFiles/test_graph.dir/tests/test_graph.cpp.i

CMakeFiles/test_graph.dir/tests/test_graph.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_graph.dir/tests/test_graph.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/tests/test_graph.cpp -o CMakeFiles/test_graph.dir/tests/test_graph.cpp.s

CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.o: CMakeFiles/test_graph.dir/flags.make
CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.o: ../third_party/googletest/googletest/src/gtest_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.o -c /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/third_party/googletest/googletest/src/gtest_main.cc

CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/third_party/googletest/googletest/src/gtest_main.cc > CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.i

CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/third_party/googletest/googletest/src/gtest_main.cc -o CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.s

# Object files for target test_graph
test_graph_OBJECTS = \
"CMakeFiles/test_graph.dir/tests/test_graph.cpp.o" \
"CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.o"

# External object files for target test_graph
test_graph_EXTERNAL_OBJECTS =

test_graph: CMakeFiles/test_graph.dir/tests/test_graph.cpp.o
test_graph: CMakeFiles/test_graph.dir/third_party/googletest/googletest/src/gtest_main.cc.o
test_graph: CMakeFiles/test_graph.dir/build.make
test_graph: lacam/liblacam.a
test_graph: lib/libgtest.a
test_graph: CMakeFiles/test_graph.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test_graph"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_graph.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_graph.dir/build: test_graph

.PHONY : CMakeFiles/test_graph.dir/build

CMakeFiles/test_graph.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_graph.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_graph.dir/clean

CMakeFiles/test_graph.dir/depend:
	cd /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug/CMakeFiles/test_graph.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_graph.dir/depend

