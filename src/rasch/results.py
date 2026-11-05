from dataclasses import dataclass
import numpy as np

@dataclass
class RaschResults:
    """
    Container class storing parameter estimates
    and diagnostic information from Rasch model
    parameter estimation.
    """

    a_s: np.ndarray
    d_q: np.ndarray

    log_likelihoods: list[float]

    gamma_a_history: list[float]
    gamma_d_history: list[float]

    converged: bool
    iterations: int

    message: str

    def summary(self):
        """
        Print a summary of the model fit.
        """
        print()
        print("Rasch Model Fit")
        print()
        print(f"Converged: {self.converged}")
        print(f"Iterations: {self.iterations}")

        if self.log_likelihoods:
            print(f"Final log-likelihood value: {self.log_likelihoods[-1]:.3f}")
        
        print(self.message)