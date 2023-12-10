#include "../include/dist_table.hpp"

DistTable::DistTable(const Instance& ins)
    : K(ins.G.V.size()), table(ins.N, std::vector<int>(K, K)),
      ins_(&ins)
{
  setup(&ins);
}

DistTable::DistTable(const Instance* ins)
    : K(ins->G.V.size()), table(ins->N, std::vector<int>(K, K)),
    ins_(ins)
{
  setup(ins);
}

DistTable::DistTable(const Instance* ins, std::mt19937* MT, std::string h_type, double mult_noise)
    : K(ins->G.V.size()), table(ins->N, std::vector<int>(K, K)),
      ins_(ins), h_type_(h_type), mult_noise_(mult_noise), MT_(MT)
{
  setup(ins);
}

void DistTable::setup(const Instance* ins)
{
  noisyTable = std::vector<std::vector<double> >(ins->N, std::vector<double>(K, K));
  for (size_t i = 0; i < ins->N; ++i) {
    OPEN.push_back(std::queue<Vertex*>());
    auto n = ins->goals[i];
    OPEN[i].push(n);
    table[i][n->id] = 0;
  }
}

int DistTable::get(int i, int v_id)
{
  if (table[i][v_id] < K) return table[i][v_id];

  /*
   * BFS with lazy evaluation
   * c.f., Reverse Resumable A*
   * https://www.aaai.org/Papers/AIIDE/2005/AIIDE05-020.pdf
   * This is just backward djikstras
   */

  while (!OPEN[i].empty()) {
    auto n = OPEN[i].front();
    OPEN[i].pop();
    const int d_n = table[i][n->id];
    for (auto& m : n->neighbor) {
      const int d_m = table[i][m->id];
      if (d_n + 1 >= d_m) continue;
      table[i][m->id] = d_n + 1;
      OPEN[i].push(m);
    }
    if (n->id == v_id) return d_n;
  }
  return K;
}

int DistTable::get(int i, Vertex* v) { return get(i, v->id); }


double DistTable::getDouble(int i, Vertex* v) {
  if (noisyTable[i][v->id] < K) 
    return noisyTable[i][v->id];
  if (v->id == ins_->goals[i]->id) 
    return 0;

  if (h_type_ == "manhattan") {
    int curr_index = v->index;
    int width = ins_->G.width;
    int curr_col = curr_index % width;
    int curr_row = curr_index / width;
    int goal_index = ins_->goals[i]->index;
    int goal_col = goal_index % width;
    int goal_row = goal_index / width;
    noisyTable[i][v->id] = std::abs(curr_col - goal_col) + std::abs(curr_row - goal_row);
  }
  else if (h_type_ == "perfect") {
    noisyTable[i][v->id] = get(i, v);
  } 
  else if (h_type_ == "noisy") {
    double val = get(i, v);
    noisyTable[i][v->id] = get_noisy_value(MT_, val, mult_noise_);
  }
  else {
    throw std::runtime_error("Invalid heuristic type");
  }
  return noisyTable[i][v->id];
}