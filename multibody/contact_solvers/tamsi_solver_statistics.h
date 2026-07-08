#pragma once

namespace drake {
namespace multibody {

/// The result from TamsiSolver::SolveWithGuess() used to report the success or
/// failure of the solver.
enum class TamsiSolverResult {
  /// Successful computation.
  kSuccess = 0,

  /// The maximum number of iterations was reached.
  kMaxIterationsReached = 1,

  /// The linear solver used within the Newton-Raphson loop failed. This might
  /// be caused by a divergent iteration that led to an invalid Jacobian matrix.
  kLinearSolverFailed = 2,

  /// Could not solve for the alpha coefficient for per-iteration angle change.
  kAlphaSolverFailed = 3,
};

namespace contact_solvers {
namespace internal {

// Struct used to store TAMSI solver statistics.
struct TamsiStatistics {
  // The accepted number of substeps for the successful plant step.
  int accepted_num_substeps{0};

  // The number of substep counts attempted. For example, if TAMSI attempts
  // num_substeps = 1, 2, and 3 before succeeding, this value is 3.
  int num_substep_attempts{0};

  // The total number of calls to TamsiSolver::SolveWithGuess() across all
  // substep attempts.
  int num_solve_calls{0};

  // The total number of solver iterations across all SolveWithGuess() calls.
  int total_iterations{0};

  // The maximum number of solver iterations used by any SolveWithGuess() call.
  int max_iterations_per_solve{0};

  // The final tangential velocity residual from the last solve call that
  // reported residual data.
  double final_vt_residual{0.0};

  // Result from the final substep attempt.
  TamsiSolverResult result{TamsiSolverResult::kMaxIterationsReached};
};

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake
