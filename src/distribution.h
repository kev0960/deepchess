#ifndef DISTRIBUTION_H 
#define DISTRIBUTION_H 

#include <random>
#include <vector>
#include <cassert>

namespace chess {

class Distribution {
 public:
  virtual std::vector<float> GetDistribution(int n) = 0;
};

class UniformDistribution : public Distribution {
  public:
    std::vector<float> GetDistribution(int n) {
      assert(n != 0);

      std::vector<float> dist(n, 1.0 / n);
      return dist;
    }
};

class DirichletDistribution : public Distribution {
 public:
  DirichletDistribution(float alpha) : gamma_(alpha), gen_(rd_()) {}

  std::vector<float> GetDistribution(int n) override {
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

