#ifndef DIRICHLET_H
#define DIRICHLET_H

#include <random>
#include <vector>

namespace chess {

class DirichletDistribution {
 public:
  DirichletDistribution(float alpha) : gamma_(alpha), gen_(rd_()) {}

  std::vector<float> GetDistribution(int n) {
    std::vector<float> dist;
    dist.reserve(n);

    for (int i = 0; i < n; i++) {
      dist.push_back(gamma_(gen_));
    }

    // Normalize.
    float total = std::reduce(dist.begin(), dist.end());
    for (int i = 0; i < n; i++) {
      dist[i] = dist[i] / total;
    }

    return dist;
  }

 private:
  std::gamma_distribution<float> gamma_;

  std::random_device rd_;
  std::mt19937 gen_;
};

}  // namespace chess

#endif

