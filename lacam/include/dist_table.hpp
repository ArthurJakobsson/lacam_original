/*
 * distance table with lazy evaluation, using BFS
 */
#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

struct DistTable {
  const int K;  // number of vertices
  std::vector<std::vector<int> >
      table;  // distance table, index: agent-id & vertex-id
  std::vector<std::queue<Vertex*> > OPEN;  // search queue

  int get(int i, int v_id);   // agent, vertex-id
  int get(int i, Vertex* v);  // agent, vertex


  double getDouble(int i, Vertex* v);
  std::string h_type_;
  // double additive_noise_stddev;
  double mult_noise_;
  const Instance* ins_;
  std::mt19937* MT_;
  std::vector<std::vector<double> >
      noisyTable;  // distance table, index: agent-id & vertex-id
  DistTable(const Instance* ins, std::mt19937* MT, std::string h_type, double mult_noise);

  DistTable(const Instance& ins);
  DistTable(const Instance* ins);

  void setup(const Instance* ins);  // initialization
};
