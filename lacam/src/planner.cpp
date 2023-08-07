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

Node::Node(Config _C, DistTable& D, Node* _parent)
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
    for (size_t i = 0; i < N; ++i) priorities[i] = (float)D.get(i, C[i]) / N;
  } else {
    // dynamic priorities, akin to PIBT
    for (size_t i = 0; i < N; ++i) {
      if (D.get(i, C[i]) != 0) {
        priorities[i] = parent->priorities[i] + 1;
      } else {
        priorities[i] = parent->priorities[i] - (int)parent->priorities[i];
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
                 int _k, int _verbose)
    : ins(_ins),
      deadline(_deadline),
      MT(_MT),
      module(_module),
      verbose(_verbose),
      K(_k),
      N(ins->N),
      V_size(ins->G.size()),
      D(DistTable(ins)),
      C_next(Candidates(N, std::array<Vertex*, 5>())),
      tie_breakers(std::vector<float>(V_size, 0)),
      A(Agents(N, nullptr)),
      occupied_now(Agents(V_size, nullptr)),
      occupied_next(Agents(V_size, nullptr))
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

at::Tensor Planner::inputs_to_torch(torch::Tensor& t_grid, torch::Tensor& t_bd,
            std::vector<torch::Tensor>& helper_bds, std::vector<std::vector<double>>& helper_loc)
{
  std::vector<torch::jit::IValue> inputs;
  torch::Tensor stacked = torch::stack({t_grid, t_bd, helper_bds[0],
                            helper_bds[1], helper_bds[2], helper_bds[3]});
  inputs.push_back(stacked.unsqueeze(0)); // (6,9,9) --> (1,6,9,9)
  inputs.push_back(torch::flatten(getTensorFrom2DVecs(helper_loc)).unsqueeze(0)); // (8) --> (1,8)
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
  grd.resize(N);
  for (int i = 0; i < N; ++i)
  {
    grd[i].resize(N);
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if(U[width * y + x] == nullptr) // put in 1 if obstacle, else 0
      {
        grd[x][y] = 1;
      } else {
        grd[x][y] = 0;
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
  bd.resize(N);
  for (int i = 0; i < N; ++i)
  {
    bd[i].resize(N);
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      auto u = U[width * y + x];
      if(U[width * y + x] == nullptr) // put in 1 if obstacle, else 0
      {
        bd[x][y] = 0;
      } else {
        bd[x][y] = D.get(a_id, u);
      }
    }
  }
  torch::Tensor t_bd = getTensorFrom2DVecs(bd);
  // pad with 1's in every direction
  t_bd = F::pad(t_bd, F::PadFuncOptions({K, K, K, K}).value(0));
  return t_bd;
}

torch::Tensor Planner::bd_helper(std::vector<std::pair<int, std::pair<int,int>>>& dist,
                        int nth_help, int curr_size)
{
  if(nth_help+1 >= curr_size)
  {

    std::vector<std::vector<double> > vec(2*K+1,
          std::vector<double>(2*K+1));
    return getTensorFrom2DVecs(vec);
  }
  return bd[dist[nth_help].first].index({Slice((dist[nth_help].second).first, (dist[nth_help].second).first+2*K+1),
	    Slice((dist[nth_help].second).second, (dist[nth_help].second).second + 2*K + 1)});
}

std::vector<double> help_loc_helper(std::vector<std::pair<int, std::pair<int,int>>>& dist,
                        int nth_help, int curr_size)
{
  std::vector<double> point;
  point.resize(2);
  std::fill(point.begin(), point.end(), 0);
  if(nth_help < curr_size)
  {
    point[0] = (dist[nth_help].second).first;
    point[1] = (dist[nth_help].second).second;
  }
  return point;
}

std::vector<std::map<int, double>> Planner::createNbyFive (const Vertices &C)
{
  std::vector<std::map<int, double>> predictions;
  predictions.resize(N);

  for(int a_id = 0; a_id<N; a_id++)
  {
    int width = ins->G.width;
    int height = ins->G.height;
    int curr_index = A[a_id]->v_now->index;
    int curr_x = curr_index % width;
    int curr_y = (curr_index - curr_x) / width;
    torch::Tensor loc_grid = grid.index({Slice(curr_x, curr_x+2*K+1),
											Slice(curr_y, curr_y + 2*K + 1)});
    torch::Tensor loc_bd =  bd[a_id].index({Slice(curr_x, curr_x+2*K+1),
											Slice(curr_y, curr_y + 2*K + 1)});
    // get 4 nearest agents
    std::vector<std::pair<int, std::pair<int,int>>> locs; //hold agt id, loc
    for (int j = 0; j<N; j++)
    {
      int help_index = A[j]->v_now->index;
      int help_x = help_index % width;
      int help_y = (help_index - help_x) / width;
      if(curr_x-K <= help_x && curr_y + K +1 >= help_y)
      {
        locs.push_back({j, {help_x, help_y}});
      }
    }
    std::sort(locs.begin(), locs.end(),
            [&](std::pair<int, std::pair<int,int>> W, std::pair<int,std::pair<int,int>> U) {
              return (abs((W.second).first - curr_x) + abs((W.second).second - curr_y)) <
                      (abs((U.second).first - curr_x) + abs((U.second).second - curr_y));
            });
    std::vector<torch::Tensor> help_bd;
    help_bd.resize(4);
    //sort distance and take indices [1][2][3][4] (0 is itself)
    std::vector<std::vector<double>> helper_loc;
    helper_loc.resize(4);
    for(int i = 0; i < 4; i++)
    {
      helper_loc[i] = help_loc_helper(locs, i, locs.size());
      help_bd[i] = bd_helper(locs, i, locs.size());
    }
    at::Tensor NN_result = inputs_to_torch(loc_grid, loc_bd, help_bd, helper_loc);

    std::vector<double> v_NN_res(NN_result.data_ptr<float>(), NN_result.data_ptr<float>() + NN_result.numel());
    //   if label[0] == 0 and label[1] == 0: index = 0
    //    elif label[0] == 0 and label[1] == 1: index = 1
        // elif label[0] == 1 and label[1] == 0: index = 2
        // elif label[0] == -1 and label[1] == 0: index = 3
        // else: index = 4
    Vertices U = ins->G.U;

    // agent location add for "no action"
    predictions[a_id][U[width * curr_y + curr_x]->id] = v_NN_res[0];
    int delta_y[4] = {1, -1, 0, 0}; //up down left right
    int delta_x[4] = {0, 0, -1, 1}; //up down left right
    int nn_index[4] = {1, 4, 3, 2};
    for(int j = 0; j<4; j++)
    {
      int this_y = curr_y+delta_y[j];
      int this_x = curr_x+delta_x[j];
      if(this_y<0 || this_x<0 || this_x>=width || this_y >= height) continue;
      auto location = U[width * (this_y) + (this_x)];
      if(location!=nullptr)
      {
        predictions[a_id][location->id] = v_NN_res[nn_index[j]];
      }
    }
    // working example with using D.get inputs (recreate LaCAM)
    // std::vector<Vertex*> c_next = C[i]->neighbor;
    // size_t next_size = c_next.size();
    // predictions[i][C[i]->id] = D.get(i, C[i]->id);
    // for(size_t j = 0; j < next_size; j++)
    // {
    //   predictions[i][c_next[j]->id] = D.get(i, c_next[j]);
    // }
  }
  return predictions;
}

Solution Planner::solve()
{

  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\tstart search");

  // setup agents
  for (auto i = 0; i < N; ++i) A[i] = new Agent(i);

  grid = get_map();
  bd.resize(N);
  for(int i = 0; i < N; ++i)
  {
    bd[i] = get_bd(i);
  }

  // setup search queues
  std::stack<Node*> OPEN;
  std::unordered_map<Config, Node*, ConfigHasher> CLOSED;
  std::vector<Constraint*> GC;  // garbage collection of constraints

  // insert initial node
  auto S = new Node(ins->starts, D);
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

    //setup caching system for node
    // only do if the high level node is new:
    // std::vector<std::vector<double>> preds = createNbyFive(N, D, S->C); //change this to only have effect on PIBT
    // each high level node should store the proposals
    // create successors at the high-level search
    if (!get_new_config(S, M)) continue; // get new config is the "for loop" part

    // create new configuration
    auto C = Config(N, nullptr);
    for (auto a : A) C[a->id] = a->v_next; //this should set to the current path indices a1 b1 stuff

    // check explored list
    auto iter = CLOSED.find(C);
    if (iter != CLOSED.end()) {
      OPEN.push(iter->second);
      continue;
    }

    // insert new search node
    auto S_new = new Node(C, D, S);
    OPEN.push(S_new);
    CLOSED[S_new->C] = S_new;
  }

  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\t",
       solution.empty() ? (OPEN.empty() ? "no solution" : "failed")
                        : "solution found",
       "\tloop_itr:", loop_cnt, "\texplored:", CLOSED.size());
  // memory management
  for (auto a : A) delete a;
  for (auto M : GC) delete M;
  for (auto p : CLOSED) delete p.second;

  return solution;
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

  //cache Nby5;
  std::vector<std::map<int,double>> preds;
  if(S->predictions.empty())
  {
    //cache old predictions
    preds = createNbyFive(S->C);
    S->predictions = preds;
  } else {
    preds = S->predictions;
  }


  // perform PIBT
  for (auto k : S->order) {
    auto a = A[k];
    if (a->v_next == nullptr && !funcPIBT(a, preds)) return false;  // planning failure
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


  //sort, note: K + 1 is sufficient <-- this is where the NN feeds in
  std::sort(C_next[i].begin(), C_next[i].begin() + Ks + 1,
            [&](Vertex* const v, Vertex* const u) {
              return preds[i][v->id] + tie_breakers[v->id] <
                      preds[i][u->id] + tie_breakers[u->id];
            });


  // std::sort(C_next[i].begin(), C_next[i].begin() + K + 1,
  //           [&](Vertex* const v, Vertex* const u) {
  //             return D.get(i,v) <// + tie_breakers[v->id] <
  //                    D.get(i,u);// + tie_breakers[u->id];
  //           });


  // D.get  should become -> proposal[i][v->id] or equivalent access or proposal weights

  // maybe sort instead of D by N by <=5

  // replace D with sort by  1 by <=5 (greedily choose best option for my current agent)

  for (size_t k = 0; k < Ks + 1; ++k) {
    auto u = C_next[i][k];

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
               std::mt19937* MT, torch::jit::script::Module* module, int k)
{
  info(1, verbose, "elapsed:", elapsed_ms(deadline), "ms\tpre-processing");
  auto planner = Planner(&ins, deadline, MT, module, k, verbose);
  return planner.solve();
}
