#include <argparse/argparse.hpp>
#include <lacam.hpp>
#include <torch/script.h>


int main(int argc, char* argv[])
{
  // arguments parser
  argparse::ArgumentParser program("lacam", "0.1.0");
  program.add_argument("-m", "--map").help("map file").required();
  program.add_argument("-i", "--scen")
      .help("scenario file")
      .default_value(std::string(""));
  program.add_argument("-N", "--num").help("number of agents").required();
  program.add_argument("-s", "--seed")
      .help("seed")
      .default_value(std::string("0"));
  program.add_argument("-v", "--verbose")
      .help("verbose")
      .default_value(std::string("0"));
  program.add_argument("-t", "--time_limit_sec")
      .help("time limit sec")
      .default_value(std::string("10"));
  program.add_argument("-o", "--outputcsv")
      .help("output csv")
      .default_value(std::string("./build/result.csv"));
  program.add_argument("-p", "--outputpaths")
      .help("output agent paths")
      .default_value(std::string("./build/result_paths.txt"));
  program.add_argument("-l", "--log_short")
      .default_value(false)
      .implicit_value(true);
  program.add_argument("-M", "--model")
      .help("model file")
      .default_value(std::string("./models/91_val_paris.pt"));
  program.add_argument("-k", "--kval")
      .default_value(std::string("4")); //TODO <-- this could cause problems
  program.add_argument("-n", "--neural_flag")
      .default_value(std::string("true")); //TODO <-- this could cause problems
  program.add_argument("--force_goal_wait")
      .help("Whether to force agents to wait at their goal via PIBT").required();
  program.add_argument("--relative_last_action")
      .help("Include a relative last action input to the NN").required();
  program.add_argument("--target_indicator")
      .help("Include an indicator if at target to the NN").required();
  program.add_argument("--neural_random")
      .help("Whether to randomize outputs of neural instead of picking arg max").required();
  program.add_argument("--prioritized_helpers")
      .help("Whether to have earlier agents ignore later ones by not including them in helper bds and locs").required();
  program.add_argument("--just_pibt")
      .help("Whether to run just pibt").required();
  program.add_argument("--tie_breaking")
      .help("Whether to run tie_breaking metric").required();
  program.add_argument("--r_weight")
      .help("r_weight for weighted combination").required();
  program.add_argument("--h_type")
      .help("heuristic type, one of [perfect, manhattan, noisy]").required();
  program.add_argument("--mult_noise")
      .help("heuristic multi_noise, [0,1]").required();
  program.add_argument("--initial_ordering")
      .help("Initial ordering of agents [bd, random, inverse]").required();
  program.add_argument("--adaptive_priorities")
      .help("Whether to use adaptive priorities or keep them constant").required();
    

  try {
    program.parse_known_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  // setup instance
  const auto verbose = std::stoi(program.get<std::string>("verbose"));
  const auto time_limit_sec =
      std::stoi(program.get<std::string>("time_limit_sec"));
  const auto scen_name = program.get<std::string>("scen");
  const auto seed = std::stoi(program.get<std::string>("seed"));
  auto MT = std::mt19937(seed);
  const auto map_name = program.get<std::string>("map");
  const auto output_csv = program.get<std::string>("outputcsv");
  const auto output_agent_paths = program.get<std::string>("outputpaths");
  const auto log_short = program.get<bool>("log_short");
  const auto model_name = program.get<std::string>("model");
  const auto k = std::stoi(program.get<std::string>("kval"));
  const auto N = std::stoi(program.get<std::string>("num"));
  const auto ins = scen_name.size() > 0 ? Instance(scen_name, map_name, N)
                                        : Instance(map_name, &MT, N);
  bool neural_flag = program.get<std::string>("neural_flag") == "true" ||
                      program.get<std::string>("neural_flag") == "True";
  bool force_goal_wait = program.get<std::string>("force_goal_wait") == "true" ||
                      program.get<std::string>("force_goal_wait") == "True";
  bool relative_last_action = program.get<std::string>("relative_last_action") == "true" ||
                      program.get<std::string>("relative_last_action") == "True";
  bool target_indicator = program.get<std::string>("target_indicator") == "true" ||
                      program.get<std::string>("target_indicator") == "True";
  bool neural_random = program.get<std::string>("neural_random") == "true" ||
                      program.get<std::string>("neural_random") == "True";
  bool prioritized_helpers = program.get<std::string>("prioritized_helpers") == "true" ||
                      program.get<std::string>("prioritized_helpers") == "True";
  bool just_pibt = program.get<std::string>("just_pibt") == "true" ||
                      program.get<std::string>("just_pibt") == "True";
  bool tie_breaking = program.get<std::string>("tie_breaking") == "true" ||
                      program.get<std::string>("tie_breaking") == "True";
  const std::string h_type = program.get<std::string>("h_type");
  const double mult_noise = std::stod(program.get<std::string>("mult_noise"));
  const std::string initial_ordering = program.get<std::string>("initial_ordering");
  const bool adaptive_priorities = program.get<std::string>("adaptive_priorities") == "true" ||
                      program.get<std::string>("adaptive_priorities") == "True";
  
  assert(initial_ordering == "bd" || initial_ordering == "random" || initial_ordering == "inverse");

  if (mult_noise != 0) {
    assert(h_type == "noisy");
  }

  double r_weight = std::stod(program.get<std::string>("r_weight"));
  if (tie_breaking) {
    assert(r_weight == 0);
  }
  if (r_weight != 0) {
    assert(tie_breaking == false && neural_random == false);
  }

  if (!ins.is_valid(1)) return 1;

  //setup model
  torch::jit::script::Module module;
  if (neural_flag) {
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(model_name);
    }
    catch (const c10::Error& e) {
      std::cerr << "error loading the model\n";
      return -1;
    }
  }

  // solve
  Solution solution;
  int cache_hit, loop_cnt;
  
  const auto deadline = Deadline(time_limit_sec * 1000);
  AllSolution all_solution = solveAll(ins, verbose - 1, &deadline, &MT, &module, k, neural_flag, force_goal_wait,
                                      relative_last_action, target_indicator, neural_random, prioritized_helpers, 
                                      just_pibt, tie_breaking, r_weight, h_type, mult_noise,
                                      initial_ordering, adaptive_priorities);
  const auto comp_time_ms = deadline.elapsed_ms();
  std::tie(solution, cache_hit, loop_cnt) = all_solution;
  // failure
  if (solution.empty()) info(1, verbose, "failed to solve");

  // check feasibility
  if (!is_feasible_solution(ins, solution, verbose)) {
    info(0, verbose, "invalid solution");
    return 1;
  }

  // post processing
  print_stats(verbose, ins, solution, comp_time_ms);
  make_log(ins, all_solution, output_csv, output_agent_paths, comp_time_ms, map_name, scen_name, 
          seed, initial_ordering, adaptive_priorities, log_short);
  return 0;
}
