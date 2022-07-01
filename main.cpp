#include <argparse/argparse.hpp>
#include <iostream>
#include <planner.hpp>

int main(int argc, char* argv[])
{
  argparse::ArgumentParser program("fast_mapf", "0.1.0");
  program.add_argument("-m", "--map").help("map file").required();
  program.add_argument("-i", "--scen").help("scenario file").required();
  program.add_argument("-N", "--num").help("number of agents").required();
  program.add_argument("-v", "--verbose")
      .help("verbose")
      .default_value(std::string("0"));
  program.add_argument("-t", "--time_limit_sec")
      .help("time limit sec")
      .default_value(std::string("10"));

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // read instance
  const auto verbose = std::stoi(program.get<std::string>("verbose"));
  const auto time_limit_sec =
      std::stoi(program.get<std::string>("time_limit_sec"));
  const auto scen_name = program.get<std::string>("scen");
  const auto map_name = program.get<std::string>("map");
  const auto N = std::stoi(program.get<std::string>("num"));
  const auto ins = Instance(scen_name, map_name, N);
  if (!is_valid_instance(ins, verbose)) return 1;

  // solve
  const auto deadline = Deadline(time_limit_sec * 1000);
  const auto solution = solve(ins, &deadline);
  const auto comp_time_ms = deadline.elapsed_ms();

  // failure
  if (solution.empty()) {
    info(1, verbose, "failed to solve");
    return 0;
  }
  // check feasibility
  if (!is_feasible_solution(ins, solution, verbose)) return 1;
  // metrics
  info(1, verbose, "solved: ", comp_time_ms, "ms");

  return 0;
}
