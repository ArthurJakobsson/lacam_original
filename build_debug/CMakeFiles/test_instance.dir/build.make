# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/arthur/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/arthur/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug

# Include any dependencies generated for this target.
include CMakeFiles/test_instance.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_instance.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_instance.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_instance.dir/flags.make

CMakeFiles/test_instance.dir/tests/test_instance.cpp.o: CMakeFiles/test_instance.dir/flags.make
CMakeFiles/test_instance.dir/tests/test_instance.cpp.o: /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/tests/test_instance.cpp
CMakeFiles/test_instance.dir/tests/test_instance.cpp.o: CMakeFiles/test_instance.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_instance.dir/tests/test_instance.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_instance.dir/tests/test_instance.cpp.o -MF CMakeFiles/test_instance.dir/tests/test_instance.cpp.o.d -o CMakeFiles/test_instance.dir/tests/test_instance.cpp.o -c /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/tests/test_instance.cpp

CMakeFiles/test_instance.dir/tests/test_instance.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_instance.dir/tests/test_instance.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/tests/test_instance.cpp > CMakeFiles/test_instance.dir/tests/test_instance.cpp.i

CMakeFiles/test_instance.dir/tests/test_instance.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_instance.dir/tests/test_instance.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/tests/test_instance.cpp -o CMakeFiles/test_instance.dir/tests/test_instance.cpp.s

CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o: CMakeFiles/test_instance.dir/flags.make
CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o: /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/third_party/googletest/googletest/src/gtest_main.cc
CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o: CMakeFiles/test_instance.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o -MF CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o.d -o CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o -c /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/third_party/googletest/googletest/src/gtest_main.cc

CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/third_party/googletest/googletest/src/gtest_main.cc > CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.i

CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/third_party/googletest/googletest/src/gtest_main.cc -o CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.s

# Object files for target test_instance
test_instance_OBJECTS = \
"CMakeFiles/test_instance.dir/tests/test_instance.cpp.o" \
"CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o"

# External object files for target test_instance
test_instance_EXTERNAL_OBJECTS =

test_instance: CMakeFiles/test_instance.dir/tests/test_instance.cpp.o
test_instance: CMakeFiles/test_instance.dir/third_party/googletest/googletest/src/gtest_main.cc.o
test_instance: CMakeFiles/test_instance.dir/build.make
test_instance: lacam/liblacam.a
test_instance: lib/libgtest.a
test_instance: /home/arthur/Home/MAPF_ML/libtorch/lib/libtorch.so
test_instance: /home/arthur/Home/MAPF_ML/libtorch/lib/libc10.so
test_instance: /home/arthur/Home/MAPF_ML/libtorch/lib/libc10.so
test_instance: /home/arthur/Home/MAPF_ML/libtorch/lib/libkineto.a
test_instance: CMakeFiles/test_instance.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable test_instance"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_instance.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_instance.dir/build: test_instance
.PHONY : CMakeFiles/test_instance.dir/build

CMakeFiles/test_instance.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_instance.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_instance.dir/clean

CMakeFiles/test_instance.dir/depend:
	cd /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug /home/arthur/Home/MAPF_ML/Github/ml-mapf/lacam/lacam_original/build_debug/CMakeFiles/test_instance.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_instance.dir/depend

