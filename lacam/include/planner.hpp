/*
 * LaCAM algorithm
 */
#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <float.h>

// low-level search node
struct Constraint {
  std::vector<int> who;
  Vertices where;
  const int depth;
  Constraint();
  Constraint(Constraint* parent, int i, Vertex* v);  // who and where
  ~Constraint();
};

// high-level search node
struct Node {
  const Config C;
  Node* parent;

  // for low-level search
  std::vector<float> priorities;
  std::vector<int> order;
  std::queue<Constraint*> search_tree;
  std::vector<std::vector<double>>  action_ranking;

  //store predictions here
  std::vector<std::map<int,double>> predictions;

  Node(Config _C, DistTable& D, Node* _parent = nullptr);
  ~Node();
};
using Nodes = std::vector<Node*>;

// PIBT agent
struct Agent {
  const int id;
  Vertex* v_now;   // current location
  Vertex* v_next;  // next location
  Agent(int _id) : id(_id), v_now(nullptr), v_next(nullptr) {}
};
using Agents = std::vector<Agent*>;

// next location candidates, for saving memory allocation
using Candidates = std::vector<std::array<Vertex*, 5> >;

struct Planner {
  const Instance* ins;
  const Deadline* deadline;
  std::mt19937* MT;
  torch::jit::script::Module* module;
  const int verbose;

  // solver utils
  const int K;
  const int N;  // number of agents
  const int V_size;
  DistTable D;
  Candidates C_next;                // next location candidates
  std::vector<float> tie_breakers;  // random values, used in PIBT
  Agents A;
  Agents occupied_now;   // for quick collision checking
  Agents occupied_next;  // for quick collision checking
  torch::Tensor grid;
  std::vector<torch::Tensor> bd;
  int cache_hit;
  bool neural_flag; // Whether to use NN or not
  bool force_goal_wait; // Whether to force agents to wait at their goal via PIBT
  bool relative_last_action; // Whether to use relative last action or not
  bool target_indicator; // Whether to use target indicator or not
  bool neural_random; // Whether to randomize outputs of neural instead of picking arg max
  bool prioritized_helpers; // Whether to prioritize helpers for NN (so earlier agents ignore later ones)
  bool just_pibt; // Whether to just use PIBT instead of LaCAM
  bool tie_breaking; // Whether to use tie breaking combination
  double r_weight; // Weight for weighted combination of NN and PIBT

  Planner(const Instance* _ins, const Deadline* _deadline, std::mt19937* _MT,
          torch::jit::script::Module* _module, int _k = 4, int _verbose = 0, bool _neural_flag = true,
          bool _force_goal_wait = false, bool _relative_last_action = false, bool _target_indicator = false,
          bool _neural_random = false, bool _prioritized_helpers = false, 
          bool _just_pibt = false, bool _tie_breaking = false, double r_weight = 0,
          std::string h_type = "perfect", double mult_noise = 0);
  torch::Tensor slice_and_fix_pad(torch::Tensor curr_bd, int row, int col, int subtract_row, int subtract_col);
  std::vector<std::map<int, double>> createNbyFive (const Node* S);
  torch::Tensor get_map();
  torch::Tensor get_bd(int a_id);
  torch::Tensor bd_helper(int which_agent, int curr_row, int curr_col, 
                        int agent_row, int agent_col);
  at::Tensor inputs_to_torch(torch::Tensor& t_grid, torch::Tensor& t_bd,
        std::vector<torch::Tensor>& helper_bds, std::vector<std::vector<double>>& helper_loc);
  AllSolution solve();
  bool get_new_config(Node* S, Constraint* M);
  bool funcPIBT(Agent* ai,std::vector<std::map<int,double>> &preds);
  // std::vector<std::map<int, double>> getPreferencesFromPredictions(Node* S);
};

// main function
Solution solve(const Instance& ins, const int verbose = 0,
               const Deadline* deadline = nullptr, std::mt19937* MT = nullptr,
               torch::jit::script::Module* _module = nullptr, int k = 4, bool _neural_flag = true,
               bool _force_goal_wait = false, bool _relative_last_action = false, bool _target_indicator = false,
               bool _neural_random = false, bool _prioritized_helpers = false, 
               bool _just_pibt = false, bool _tie_breaking = false, double r_weight = 0,
               std::string h_type = "perfect", double mult_noise = 0);

AllSolution solveAll(const Instance& ins, const int verbose = 0,
               const Deadline* deadline = nullptr, std::mt19937* MT = nullptr,
               torch::jit::script::Module* _module = nullptr, int k = 4, bool _neural_flag = true,
               bool _force_goal_wait = false, bool _relative_last_action = false, bool _target_indicator = false,
               bool _neural_random = false, bool _prioritized_helpers = false,
               bool _just_pibt = false, bool _tie_breaking = false, double r_weight = 0,
               std::string h_type = "perfect", double mult_noise = 0);
