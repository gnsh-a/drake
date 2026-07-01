#pragma once

#include <vector>

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

// Struct used to store SAP solver statistics.
struct SapStatistics {
  // Initializes counters and time statistics to zero.
  void Reset() {
    num_iters = 0;
    num_line_search_iters = 0;
    optimality_criterion_reached = false;
    cost_criterion_reached = false;
    momentum_residual.clear();
    momentum_scale.clear();
    cost.clear();
    alpha.clear();
  }

  int num_iters{0};              // Number of Newton iterations.
  int num_line_search_iters{0};  // Total number of line search iterations.

  // Indicates if the optimality condition was reached.
  bool optimality_criterion_reached{false};

  // Indicates if the cost condition was reached.
  bool cost_criterion_reached{false};

  // Cost at each SAP Newton iteration. cost[0] stores cost at the initial
  // guess.
  std::vector<double> cost;

  // Line search step size at each SAP Newton iteration. alpha[0] stores alpha
  // = 1.
  std::vector<double> alpha;

  // Dimensionless momentum residual at each SAP Newton iteration. Of size
  // num_iters + 1.
  std::vector<double> momentum_residual;

  // Dimensionless momentum scale at each SAP Newton iteration. Of size
  // num_iters + 1.
  std::vector<double> momentum_scale;
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
