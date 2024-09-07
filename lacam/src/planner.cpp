#include "../include/planner.hpp"
#include <iostream>
#include <map>
#include <torch/torch.h>
#include <torch/script.h>
#include <float.h>

using torch::indexing::Slice;
namespace F = torch::nn::functional;

Constraint::Constraint() : who(std::vector<int>()), where(Vertices()), depth(0)
{
}

Constraint::Constraint(Constraint* parent, int i, Vertex* v)
    : who(parent->who), where(parent->where), depth(parent->depth + 1)
{
  who.push_back(i);
  where.push_back(v);
}

Constraint::~Constraint(){};

Node::Node(Config _C, DistTable& D, Node* _parent, const Planner* planner)
    : C(_C),
      parent(_parent),
      priorities(C.size(), 0),
      order(C.size(), 0),
      search_tree(std::queue<Constraint*>())
{
  search_tree.push(new Constraint());
  const auto N = C.size();

  // set priorities
  if (parent == nullptr) {
    // initialize
    // for (size_t i = 0; i < N; ++i) 
    //   priorities[i] = (float)D.get(i, C[i]) / N;
    if (planner->initial_ordering == "bd") {
      for (size_t i = 0; i < N; ++i) {
        priorities[i] = (float)D.get(i, C[i]) / N;
      }
    } 
    else if (planner->initial_ordering == "random") {
      std::iota(priorities.begin(), priorities.end(), 0);
      std::shuffle(priorities.begin(), priorities.end(), *(planner->MT));
      for (size_t i = 0; i < N; ++i) {
        priorities[i] = priorities[i] / N;
      }
    }
    else if (planner->initial_ordering == "inverse") {
      for (size_t i = 0; i < N; ++i) {
        priorities[i] = 1 - (float)D.get(i, C[i]) / N;
        assert(priorities[i] >= 0);
      }
    }
  } else {
    // dynamic priorities, akin to PIBT
    if (planner->adaptive_priorities) {
      for (size_t i = 0; i < N; ++i) {
        if (D.get(i, C[i]) != 0) {
          priorities[i] = parent->priorities[i] + 1;
        } else {
          priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
        }
      }
    } else {
      for (size_t i = 0; i < N; ++i) {
        priorities[i] = parent->priorities[i];
      }
    }
  }

  // set order
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](int i, int j) { return priorities[i] > priorities[j]; });
}

Node::~Node()
{
  while (!search_tree.empty()) {
    delete search_tree.front();
    search_tree.pop();
  }
}

Planner::Planner(const Instance* _ins, const Deadline* _deadline,
                 std::mt19937* _MT, torch::jit::script::Module* _module,
                 int _k, int _verbose, bool _neural_flag, bool _force_goal_wait,
                 bool relative_last_action, bool target_indicator,
                 bool _neural_random, bool _prioritized_helpers,
                 bool _just_pibt, bool _tie_breaking, double _r_weight,
                 std::string h_type, double mult_noise,
                 std::string _initial_ordering, bool _adaptive_priorities)
    : ins(_ins),
      deadline(_deadline),
      MT(_MT),
      module(_module),
      verbose(_verbose),
      K(_k),
      N(ins->N),
      V_size(ins->G.size()),
      D(DistTable(ins, _MT, h_type, mult_noise)),
      C_next(Candidates(N, std::array<Vertex*, 5>())),
      tie_breakers(std::vector<float>(V_size, 0)),
      A(Agents(N, nullptr)),
      occupied_now(Agents(V_size, nullptr)),
      occupied_next(Agents(V_size, nullptr)),
      cache_hit(0),
      neural_flag(_neural_flag),
      force_goal_wait(_force_goal_wait),
      relative_last_action(relative_last_action),
      target_indicator(target_indicator),
      neural_random(_neural_random),
      prioritized_helpers(_prioritized_helpers),
      just_pibt(_just_pibt),
      tie_breaking(_tie_breaking),
      r_weight(_r_weight),
      initial_ordering(_initial_ordering),
      adaptive_priorities(_adaptive_priorities)
{
}


// https://stackoverflow.com/questions/63466847/how-is-it-possible-to-convert-a-stdvectorstdvectordouble-to-a-torchten
/* Returns 2D torch tensor */
torch::Tensor getTensorFrom2DVecs(std::vector<std::vector<double>>& vec2D) {
    int n = vec2D.size();
    int m = vec2D[0].size();
    auto options = torch::TensorOptions().dtype(at::kDouble);//at::kFloat);
    torch::Tensor tensorAns = torch::zeros({n,m}, options);
    for (int i = 0; i < n; ++i) {
        tensorAns.slice(0,i,i+1) = torch::from_blob(vec2D[i].data(), {m}, options);
    }

    tensorAns = tensorAns.to(torch::kFloat);

    return tensorAns;
}

torch::Tensor getTensorFrom1DVec(std::vector<double>& vec1D) {
    int n = vec1D.size();
    auto options = torch::TensorOptions().dtype(at::kDouble);
    torch::Tensor tensorAns = torch::zeros({n}, options);
    tensorAns = torch::from_blob(vec1D.data(), {n}, options);
    tensorAns = tensorAns.to(torch::kFloat);
    return tensorAns;
}

at::Tensor Planner::inputs_to_torch(torch::Tensor& t_grid, torch::Tensor& t_bd,
            std::vector<torch::Tensor>& helper_bds, std::vector<std::vector<double>>& helper_loc)
{
  std::vector<torch::jit::IValue> inputs;
  // std::cout << "grid\n" << t_grid << std::endl;
  // std::cout << "\nbd\n" << t_bd << std::endl;
  // std::cout << "\nhelp_bd_1\n" << helper_bds[0] << std::endl;
  // std::cout << "\nhelp_bd_2\n" << helper_bds[1] << std::endl;
  // std::cout << "\nhelp_bd_3\n" << helper_bds[2] << std::endl;
  // std::cout << "\nhelp_bd_4\n" << helper_bds[3] << std::endl;
  torch::Tensor stacked = torch::stack({t_grid, t_bd, helper_bds[0],
                            helper_bds[1], helper_bds[2], helper_bds[3]});
  inputs.push_back(stacked.unsqueeze(0)); // (6,9,9) --> (1,6,9,9)
  // std::cout << "\nsize\n" << stacked.unsqueeze(0).sizes() << std::endl;
  std::vector<double> helper_loc_flat;
  for(int i = 0; i < helper_loc.size(); i++) {
    helper_loc_flat.insert(helper_loc_flat.end(), helper_loc[i].begin(), helper_loc[i].end());
  }
  inputs.push_back(getTensorFrom1DVec(helper_loc_flat).unsqueeze(0)); // (8+optional) -> (1,8+optional)
  // inputs.push_back(torch::flatten(getTensorFrom2DVecs(helper_loc)).unsqueeze(0)); // (8) --> (1,8)
  // std::cout << "\nhelp_locations\n" << torch::flatten(getTensorFrom2DVecs(helper_loc)).unsqueeze(0) << std::endl;
  // std::cout << "help_locations_size" << torch::flatten(getTensorFrom2DVecs(helper_loc)).unsqueeze(0).sizes() << std::endl;
  // std::cout << inputs[1] << std::endl;
  at::Tensor NN_out = (*module).forward(inputs).toTensor();
  // std::cout << NN_out.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  return NN_out;
}

torch::Tensor Planner::get_map()
{
  int width = ins->G.width;
  int height = ins->G.height;
  Vertices U = ins->G.U;

  std::vector<std::vector<double>> grd;
  grd.resize(height);
  for (int i = 0; i < height; ++i)
  {
    grd[i].resize(width);
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      if(U[width * r + c] == nullptr) // put in 1 if obstacle, else 0
      {
        grd[r][c] = 1;
      } else {
        grd[r][c] = 0;
      }
    }
  }
  torch::Tensor t_grid = getTensorFrom2DVecs(grd);
  // pad with 1's in every direction
  t_grid = F::pad(t_grid, F::PadFuncOptions({K, K, K, K}).value(1));
  return t_grid;
}

torch::Tensor Planner::get_bd(int a_id)
{
  int width = ins->G.width;
  int height = ins->G.height;
  Vertices U = ins->G.U;

  std::vector<std::vector<double>> bd;
  bd.resize(height);
  for (int i = 0; i < height; ++i)
  {
    bd[i].resize(width);
  }

  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      auto u = U[width * r + c];
      if(u == nullptr) // put in 0 if obstacle, else bd value
      {
        bd[r][c] = 0;
      } else {
        bd[r][c] = D.get(a_id, u);
      }
    }
  }
  torch::Tensor t_bd = getTensorFrom2DVecs(bd);
  // pad with 0's in every direction
  t_bd = F::pad(t_bd, F::PadFuncOptions({K, K, K, K}).value(0));
  return t_bd;
}

torch::Tensor Planner::slice_and_fix_pad(torch::Tensor curr_bd, int row, int col, int subtract_row, int subtract_col) {
  torch::Tensor loc_grid = grid.index({Slice(row, row+2*K+1),
											Slice(col, col + 2*K + 1)});
  double curr_val = curr_bd.index({subtract_row+K, subtract_col+K}).item<double>();
  // std::cout << "pre" << curr_bd.index({Slice(row, row+2*K+1),
											// Slice(col, col + 2*K + 1)}) << std::endl;
  torch::Tensor loc_bd = curr_bd.index({Slice(row, row+2*K+1),
											Slice(col, col + 2*K + 1)}) - curr_val;
  loc_bd = loc_bd * (1-loc_grid);
  // std::cout << "post" << loc_bd << std::endl;
  return loc_bd;
}

torch::Tensor Planner::bd_helper(int which_agent, int curr_row, int curr_col,
                                int agent_row, int agent_col) {
  return slice_and_fix_pad(bd[which_agent], curr_row, curr_col, agent_row, agent_col);
}

std::vector<double> help_loc_helper(std::vector<std::pair<int, std::pair<int,int>>>& dist,
                        int nth_help, int curr_size, int curr_row, int curr_col)
{
  std::vector<double> point = {0, 0};
  if(nth_help < curr_size)
  {
    point[0] = (dist[nth_help].second).first - curr_row;
    point[1] = (dist[nth_help].second).second - curr_col;
  }
  return point;
}

std::vector<float> prefix_sum_help(std::vector<double> nn_values)
{
  int len = nn_values.size();
  std::vector<float> nn_prefix_sum(len);
  for(int r_n=0; r_n<len;r_n++)
  {
    nn_prefix_sum[r_n] = nn_values[r_n];
    if(r_n!=0)
    {
      nn_prefix_sum[r_n] = nn_prefix_sum[r_n]+nn_prefix_sum[r_n-1];
    }
  }
  return nn_prefix_sum;
}

std::vector<std::map<int, double>> Planner::createNbyFive(const Node* S)
{
  std::vector<std::map<int, double>> predictions;
  predictions.resize(N);

  for(int a_outer_index = 0; a_outer_index < N; a_outer_index++) {
    int a_id = S->order[a_outer_index];
    int width = ins->G.width;
    int height = ins->G.height;
    int curr_index = A[a_id]->v_now->index;
    int curr_col = curr_index % width;
    int curr_row = (curr_index - curr_col) / width;
    torch::Tensor loc_grid = grid.index({Slice(curr_row, curr_row+2*K+1),
											Slice(curr_col, curr_col + 2*K + 1)});
    torch::Tensor loc_bd = slice_and_fix_pad(bd[a_id], curr_row, curr_col, curr_row, curr_col);
    // std::cout << loc_bd << std::endl;
    // get 4 nearest agents
    std::vector<std::pair<int, std::pair<int,int>>> locs; //hold agt id, loc

    int help_agent_max_id = N;
    if (prioritized_helpers) {
      help_agent_max_id = a_outer_index+1; // Need to include itself
    }
    for (int j = 0; j < help_agent_max_id; j++) {
      // Agents later in the order should avoid earlier ones 
      //   because those agents have better priority in PIBT
      int inner_agent_id = S->order[j];
      int help_index = A[inner_agent_id]->v_now->index;
      int help_col = help_index % width;
      int help_row = (help_index - help_col) / width;
      // std::cout << curr_x << curr_y << help_x << help_y << std::endl;
      if(abs(curr_col-help_col)<= K && abs(curr_row-help_row)<= K)
      {
        locs.push_back({inner_agent_id, {help_row, help_col}});
      }
    }
    std::sort(locs.begin(), locs.end(),
            [&](std::pair<int, std::pair<int,int>> W, std::pair<int,std::pair<int,int>> U) {
              return (abs((W.second).first - curr_row) + abs((W.second).second - curr_col)) <
                      (abs((U.second).first - curr_row) + abs((U.second).second - curr_col));
            });
    std::vector<torch::Tensor> help_bd;
    help_bd.resize(4);
    //sort distance and take indices [1][2][3][4] (0 is itself)
    std::vector<std::vector<double>> helper_loc;
    helper_loc.resize(4);
    for(int i = 1; i < 5; i++)
    {
      helper_loc[i-1] = help_loc_helper(locs, i, locs.size(), curr_row, curr_col);

      //// Get agent's bd
      if(i >= locs.size()) {
        help_bd[i-1] = torch::zeros({2*K+1, 2*K+1});
      } else {
        // locs[i] is the ith closest agent (0 is itself)
        // locs[i].first is the id of the ith closest agent
        // locs[i].second is the location of the ith closest agent
        help_bd[i-1] = bd_helper(locs[i].first, curr_row, curr_col, 
                        (locs[i].second).first, (locs[i].second).second);
      }
    }
    
    //// Add prev locations if needed
    if (relative_last_action) {
      std::vector<double> prev_relative_action = {0, 0};
      if (S->parent != nullptr) {
        int prev_index = S->parent->C[a_id]->index;
        int prev_col = prev_index % width;
        int prev_row = (prev_index - prev_col) / width;
        prev_relative_action[0] = prev_row - curr_row;
        prev_relative_action[1] = prev_col - curr_col;
      }
      helper_loc.push_back(prev_relative_action); // Append to "flat" inputs
    }
    if (target_indicator) {
      double at_goal = int(D.get(a_id, A[a_id]->v_now) == 0);
      helper_loc.push_back({at_goal});
    }

    at::Tensor NN_result = inputs_to_torch(loc_grid, loc_bd, help_bd, helper_loc);
    // std::cout << NN_result << std::endl;
    NN_result = F::softmax(NN_result, F::SoftmaxFuncOptions(1)); // Apply softmax across dim 1
    std::vector<double> v_NN_res(NN_result.data_ptr<float>(), NN_result.data_ptr<float>() + NN_result.numel());
    Vertices U = ins->G.U;

    //// semi-randomize the ordering of the actions
    std::vector<double> copy(5);
    std::vector<int> indices(5);
    std::vector<int> ordering(5);
    for(int k_n = 0; k_n<5; k_n++)
    {
      copy[k_n]= v_NN_res[k_n];
      indices[k_n] = k_n;
    }
    for (int i = 0; i<5; i++)
    {
      std::vector<float> prefix = prefix_sum_help(copy);
      float sum = prefix.back();
      float rand= get_random_float(MT, 0, sum);
      for(int j = 0; j < copy.size(); j++) {
        if (rand <= prefix[j]) {
          ordering[i] = indices[j];
          copy.erase(copy.begin()+j);
          indices.erase(indices.begin()+j);
          break;
        }
      }
    }

    ///// Populate predictions using nn results or randomize orderings
    int delta_row[5] = {0, 0, 1, -1,  0}; // wait, +col, +row, -row, -col
    int delta_col[5] = {0, 1, 0,  0, -1};
    int nn_index[5] = {0, 1, 2,  3,  4}; // If row col is identical to training
    for(int j = 0; j<5; j++) {
      int this_row = curr_row+delta_row[j];
      int this_col = curr_col+delta_col[j];
      if (this_row<0 || this_col<0 || this_col>=width || this_row >= height) 
        continue;
      auto location = U[width * (this_row) + (this_col)];
      if (location!=nullptr) {
        if (neural_random) {
          // Use randomized actions
          int index = find(ordering.begin(), ordering.end(), nn_index[j]) - ordering.begin();
          predictions[a_id][location->id] = 5-index;
        }
        else {
          // No randomizing actions, sort by probabilities directly
          predictions[a_id][location->id] = v_NN_res[nn_index[j]];
        }
      }
    }
    //// Force agent to wait if currently at goal location
    if (force_goal_wait && D.get(a_id, A[a_id]->v_now) == 0) {
      predictions[a_id][A[a_id]->v_now->id] = 100; // Force to wait by assigning large value
    }


    // numbers 0.2 0.3 0.4 0.05 0.05
    // random sample number from 0 to 1
    // ex: sample 0.6, 3rd bucket, remove bucket and normalize
    // new numbers = 0.2/0.6 0.3/0.6 0.05/0.6 0.05/0.6
    // sample 0.1, 1st bucket, remove bucket...
    // new numbers = 0.3 0.05 0.05 <-- normalize by dividing by 0.4
    // repeat total of 4 times
    // sample using MT randomness

    //sort using sampling here and just add weights of 5 4 3 2 1

    // working example with using D.get inputs (recreate LaCAM)
    // std::vector<Vertex*> c_next = C[a_id]->neighbor;
    // size_t next_size = c_next.size();
    // predictions[a_id][C[a_id]->id] = -D.get(a_id, C[a_id]->id);
    // for(size_t j = 0; j < next_size; j++) {
    //   predictions[a_id][c_next[j]->id] = -D.get(a_id, c_next[j]);
    // }
  }
  return predictions;
}


AllSolution Planner::solve()
{

  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\tstart search");

  // setup agents
  for (auto i = 0; i < N; ++i) A[i] = new Agent(i);

  if(neural_flag){
    grid = get_map();
    bd.resize(N);
    for(int i = 0; i < N; ++i)
    {
      bd[i] = get_bd(i);
    }
  }


  // setup search queues
  std::stack<Node*> OPEN;
  std::unordered_map<Config, Node*, ConfigHasher> CLOSED;
  std::vector<Constraint*> GC;  // garbage collection of constraints

  // insert initial node
  auto S = new Node(ins->starts, D, nullptr, this);
  OPEN.push(S);
  CLOSED[S->C] = S;

  // depth first search
  int loop_cnt = 0;
  std::vector<Config> solution;

  while (!OPEN.empty() && !is_expired(deadline)) {
    loop_cnt += 1;

    // do not pop here!
    S = OPEN.top();

    // check goal condition
    if (is_same_config(S->C, ins->goals)) {
      // backtrack
      while (S != nullptr) {
        solution.push_back(S->C);
        S = S->parent;
      }
      std::reverse(solution.begin(), solution.end());
      break;
    } //this says if the solution is found, backtrack and prepare the solution

    //we don't need search tree, out ranking are our search tree

    // low-level search end
    if (S->search_tree.empty()) {
      OPEN.pop();
      continue;
    }

    // create successors at the low-level search
    auto M = S->search_tree.front();
    if (!just_pibt) { // PIBT always keeps no constraints so doesn't have this step
      GC.push_back(M);
      S->search_tree.pop();
      if (M->depth < N) {
        auto i = S->order[M->depth];
        auto C = S->C[i]->neighbor;
        C.push_back(S->C[i]);
        if (MT != nullptr) std::shuffle(C.begin(), C.end(), *MT);  // randomize
        //insert based on order of C
        for (auto u : C) S->search_tree.push(new Constraint(M, i, u));
      }
    }

    // create successors at the high-level search
    if (!get_new_config(S, M)) continue; // get new config is the "for loop" part

    // create new configuration
    auto C = Config(N, nullptr);
    for (auto a : A) C[a->id] = a->v_next; //this should set to the current path indices a1 b1 stuff

    // check explored list
    if (!just_pibt) { // PIBT doesn't have a closed list
      auto iter = CLOSED.find(C);
      if (iter != CLOSED.end()) {
        OPEN.push(iter->second);
        continue;
      }
    }

    // insert new search node
    auto S_new = new Node(C, D, S, this);
    OPEN.push(S_new);
    CLOSED[S_new->C] = S_new;
  }
  std::cout << "cache_hit:"<< cache_hit << " total_nodes_opened: " << loop_cnt << std::endl;
  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\t",
       solution.empty() ? (OPEN.empty() ? "no solution" : "failed")
                        : "solution found",
       "\tloop_itr:", loop_cnt, "\texplored:", CLOSED.size());
  

  // memory management
  for (auto a : A) delete a;
  for (auto M : GC) delete M;
  for (auto p : CLOSED) delete p.second;

  return make_tuple(solution, cache_hit, loop_cnt); 
}

bool Planner::get_new_config(Node* S, Constraint* M) //Node contains the N by 5
{
  // setup cache
  for (auto a : A) {
    // clear previous cache
    if (a->v_now != nullptr && occupied_now[a->v_now->id] == a) {
      occupied_now[a->v_now->id] = nullptr;
    }
    if (a->v_next != nullptr) {
      occupied_next[a->v_next->id] = nullptr;
      a->v_next = nullptr;
    }

    // set occupied now
    a->v_now = S->C[a->id];
    occupied_now[a->v_now->id] = a;
  }

  // add constraints
  for (auto k = 0; k < M->depth; ++k) {
    const auto i = M->who[k];        // agent
    const auto l = M->where[k]->id;  // loc

    // check vertex collision
    if (occupied_next[l] != nullptr) return false;
    // check swap collision
    auto l_pre = S->C[i]->id;
    if (occupied_next[l_pre] != nullptr && occupied_now[l] != nullptr &&
        occupied_next[l_pre]->id == occupied_now[l]->id)
      return false;

    // set occupied_next
    A[i]->v_next = M->where[k];
    occupied_next[l] = A[i];
  }

  if(neural_flag)
  {
    //cache Nby5;
    if(S->predictions.empty())
    {
      //cache old predictions
      S->predictions = createNbyFive(S);
    } else {
      cache_hit+=1;
    }
  }



  // perform PIBT
  for (auto k : S->order) {
    auto a = A[k];
    if (a->v_next == nullptr && !funcPIBT(a, S->predictions)) return false;  // planning failure
  }
  return true;
}


bool Planner::funcPIBT(Agent* ai, std::vector<std::map<int,double>> &preds) //pass in proposals N by <=5 table
{
  const auto i = ai->id;
  const auto Ks = ai->v_now->neighbor.size();


  //get NN inputs should get passed from above
  //pass NN inputs through NN to get output predictions
  //output predictions is a tensor with probabilities
  //post process output predictions to ordered tentative location

  // get candidates for next locations <-- dont need this section
  for (size_t k = 0; k < Ks; ++k) {
    auto u = ai->v_now->neighbor[k];
    C_next[i][k] = u;
    if (MT != nullptr)
      tie_breakers[u->id] = get_random_float(MT);  // set tie-breaker
  }
  C_next[i][Ks] = ai->v_now;




  if(neural_flag)
  {
    //sort by NN
    std::sort(C_next[i].begin(), C_next[i].begin() + Ks + 1,
            [&](Vertex* const v, Vertex* const u) {
      if (tie_breaking) {
        if (D.get(i,v) == D.get(i,u)) {
          return preds[i][v->id] > preds[i][u->id];
        }
        return D.get(i,v) < D.get(i,u);
      }
      if (r_weight > 0) {
        // Note want lowest heuristic or highest NN, which is opposite directions
        // That's why we do 1-
        return D.get(i,v) + r_weight * (1-preds[i][v->id]) <
              D.get(i,u) + r_weight * (1-preds[i][u->id]);
      }
      return preds[i][v->id] >
              preds[i][u->id];
    });
  } else {
    //native lacam (sort by distance - backward dijkstras)
    std::sort(C_next[i].begin(), C_next[i].begin() + Ks + 1,
              [&](Vertex* const v, Vertex* const u) {
      return D.getDouble(i,v)  + tie_breakers[v->id] <
            D.getDouble(i,u) + tie_breakers[u->id];
      // if (D.get(i, v) == D.get(i, u)) {
      //   if (occupied_now[u->id] && !occupied_now[v->id])
      //     return true; // Prefer v here
      //   if (!occupied_now[u->id] && occupied_now[v->id])
      //     return false; // Prefer u here
      //   return tie_breakers[v->id] < tie_breakers[u->id];
      // }
      // return D.get(i, v) < D.get(i, u);
    });
  }

  bool naive_collision_checking = false;

  for (size_t k = 0; k < Ks + 1; ++k) {
    auto u = C_next[i][k];
    if (naive_collision_checking) {
      // Just keep best action and wait action
      if (!(k == 0 || u == ai->v_now)) {
        continue;
      }
    }

    // avoid vertex conflicts
    if (occupied_next[u->id] != nullptr) continue;

    auto& ak = occupied_now[u->id];

    // avoid swap conflicts with constraints
    if (ak != nullptr && ak->v_next == ai->v_now) continue;

    // reserve next location
    occupied_next[u->id] = ai;
    ai->v_next = u;

    // empty or stay
    if (ak == nullptr || u == ai->v_now) return true;

    // priority inheritance
    if (ak->v_next == nullptr && !funcPIBT(ak, preds)) continue;


    // success to plan next one step
    return true;
  }

  // failed to secure node
  occupied_next[ai->v_now->id] = ai;
  ai->v_next = ai->v_now;
  return false;
}

Solution solve(const Instance& ins, const int verbose, const Deadline* deadline,
               std::mt19937* MT, torch::jit::script::Module* module, int k, bool neural_flag,
               bool force_goal_wait, bool relative_last_action, bool target_indicator,
               bool neural_random, bool prioritized_helpers, bool just_pibt, bool tie_breaking,
               double r_weight, std::string h_type, double mult_noise, 
               std::string initial_ordering, bool adaptive_priorities)
{
  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\tpre-processing");
  auto planner = Planner(&ins, deadline, MT, module, k, verbose, neural_flag, force_goal_wait, 
                         relative_last_action, target_indicator, neural_random,
                        prioritized_helpers, just_pibt, tie_breaking, r_weight,
                        h_type, mult_noise, initial_ordering, adaptive_priorities);
  AllSolution all_solution = planner.solve();
  return std::get<0>(all_solution);
}


AllSolution solveAll(const Instance& ins, const int verbose, const Deadline* deadline,
               std::mt19937* MT, torch::jit::script::Module* module, int k, bool neural_flag,
               bool force_goal_wait, bool relative_last_action, bool target_indicator,
               bool neural_random, bool prioritized_helpers, bool just_pibt, bool tie_breaking,
               double r_weight, std::string h_type, double mult_noise, 
               std::string initial_ordering, bool adaptive_priorities)
{
  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\tpre-processing");
  auto planner = Planner(&ins, deadline, MT, module, k, verbose, neural_flag, force_goal_wait, 
                         relative_last_action, target_indicator, neural_random, 
                         prioritized_helpers, just_pibt, tie_breaking, r_weight,
                         h_type, mult_noise, initial_ordering, adaptive_priorities);
  return planner.solve();
}
