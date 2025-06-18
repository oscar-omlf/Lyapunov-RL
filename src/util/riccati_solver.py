import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are, expm

class RiccatiSolver:
    def solve_continuous_are(self, A, B, Q, R):
        """
        Solve the continuous-time algebraic Riccati equation (CARE):
        A^T P + P A - P B R^{-1} B^T P + Q = 0
        using an eigenvalue method.
        
        Parameters:
            A (ndarray): System matrix (n x n).
            B (ndarray): Input matrix (n x m).
            Q (ndarray): State cost matrix (n x n), should be symmetric positive semi-definite.
            R (ndarray): Control cost matrix (m x m), should be symmetric positive definite.
        
        Returns:
            P (ndarray): The unique symmetric positive semi-definite solution to the CARE.
        """
        n = A.shape[0]
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix R is singular and cannot be inverted.")
        
        # Form the Hamiltonian matrix
        H = np.block([
            [A, -B @ R_inv @ B.T],
            [-Q, -A.T]
        ])
        
        # Compute eigenvalues and eigenvectors of the Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eig(H)
        
        # Identify the indices of eigenvalues with negative real parts
        # Adding a small tolerance for floating point comparisons
        stable_indices = np.where(np.real(eigenvalues) < -1e-9)[0] 
        
        if len(stable_indices) != n:
            raise ValueError(f"The number of stable eigenvalues ({len(stable_indices)}) does not equal the system dimension n ({n}).")
        
        # Extract the eigenvectors corresponding to the stable eigenvalues
        stable_eigenvectors = eigenvectors[:, stable_indices]
        
        # Partition the eigenvectors into two n x n blocks: X (top half) and Y (bottom half)
        X = stable_eigenvectors[:n, :]
        Y = stable_eigenvectors[n:, :]
        
        # Ensure X is invertible; if not, numerical issues have occurred or system properties are not met
        if np.linalg.matrix_rank(X) < n:
            raise np.linalg.LinAlgError("The matrix X is not invertible.")
        
        # Compute the solution P = Y * inv(X)
        try:
            P = np.real(Y @ np.linalg.inv(X))
        except np.linalg.LinAlgError:
             raise np.linalg.LinAlgError("The matrix X from stable eigenvectors is numerically singular during inversion.")

        # Symmetrize P
        P = (P + P.T) / 2.0
        
        return P

    def solve_discrete_are(self, A, B, Q, R):
        """
        Solve the discrete-time algebraic Riccati equation (DARE):
        P = A^T P A - (A^T P B) (R + B^T P B)^{-1} (B^T P A) + Q
        using an eigenvalue method.
        
        This implementation requires the matrix A to be invertible.
        
        Parameters:
            A (ndarray): System matrix (n x n).
            B (ndarray): Input matrix (n x m).
            Q (ndarray): State cost matrix (n x n), should be symmetric positive semi-definite.
            R (ndarray): Control cost matrix (m x m), should be symmetric positive definite.
        
        Returns:
            P (ndarray): The unique symmetric positive semi-definite solution to the DARE.
        """
        n = A.shape[0]
        
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix R is singular and cannot be inverted.")
        
        try:
            A_inv_T = np.linalg.inv(A.T)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("Matrix A is singular, this method requires A to be invertible.")

        # Form the symplectic-like matrix M for the DARE
        # M = [ A + B*R_inv*B.T*A_inv_T*Q   -B*R_inv*B.T*A_inv_T ]
        #     [      -A_inv_T*Q                  A_inv_T         ]
        # This matrix is derived from transforming the generalized eigenvalue problem
        # ( [A  0], [I  B*R_inv*B.T] )
        # ( [-Q I]  [0     A.T    ] )
        # to a standard eigenvalue problem M_transformed*v = lambda*v, where M_transformed = S1_inv * S2.
        
        M11 = A + B @ R_inv @ B.T @ A_inv_T @ Q
        M12 = -B @ R_inv @ B.T @ A_inv_T
        M21 = -A_inv_T @ Q
        M22 = A_inv_T
        
        M = np.block([
            [M11, M12],
            [M21, M22]]
        )
        
        # Compute eigenvalues and eigenvectors of M
        eigenvalues, eigenvectors = np.linalg.eig(M)
        
        # Identify the indices of eigenvalues with magnitude inside the unit circle
        # Adding a small tolerance for floating point comparisons (e.g., 1.0 - 1e-9)
        stable_indices = np.where(np.abs(eigenvalues) < 1.0 - 1e-9)[0]
        
        if len(stable_indices) != n:
            raise ValueError(f"The number of stable eigenvalues ({len(stable_indices)}) (magnitude < 1) "
                             f"does not equal the system dimension n ({n}). ")
        
        # Extract the eigenvectors corresponding to the stable eigenvalues
        stable_eigenvectors = eigenvectors[:, stable_indices]
        
        # Partition the eigenvectors into two n x n blocks: U1 (top half) and U2 (bottom half)
        U1 = stable_eigenvectors[:n, :]
        U2 = stable_eigenvectors[n:, :]
        
        # Ensure U1 is invertible
        if np.linalg.matrix_rank(U1) < n:
            raise np.linalg.LinAlgError("The matrix U1 from stable eigenvectors is not invertible or ill-conditioned.")
            
        # Compute the solution P = U2 * inv(U1)
        try:
            P = np.real(U2 @ np.linalg.inv(U1))
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError("The matrix U1 from stable eigenvectors is numerically singular during inversion.")

        # Symmetrize P
        P = (P + P.T) / 2.0
            
        return P
    
    def solve_discounted_dare(self, A, B, Q_gamma, R_gamma, gamma):
        """
        Solves the gamma-discounted Discrete Algebraic Riccati Equation of the form:
        P = Q_gamma + gamma * A^T * P * A - gamma^2 * A^T * P * B * (R_gamma + gamma * B^T * P * B)^-1 * B^T * P * A
        This is equivalent to P = Q_gamma + A_eff^T P A_eff - (A_eff^T P B_eff) (R_gamma + B_eff^T P B_eff)^-1 (B_eff^T P A_eff)
        where A_eff = sqrt(gamma)*A and B_eff = sqrt(gamma)*B.
        
        Parameters:
            A (ndarray): Original system matrix (n x n).
            B (ndarray): Original input matrix (n x m).
            Q_gamma (ndarray): Discounted state cost matrix for the DARE (n x n).
            R_gamma (ndarray): Discounted control cost matrix for the DARE (m x m).
            gamma (float): Discount factor (0 < gamma <= 1).
        
        Returns:
            P_solution (ndarray): The unique symmetric positive semi-definite solution to the discounted DARE.
                                This is the P used for K* and H in the LAS paper.
        """
        if not (0 < gamma <= 1):
            raise ValueError("Discount factor gamma must be in (0, 1].")

        sqrt_gamma = np.sqrt(gamma)
        A_eff = sqrt_gamma * A
        B_eff = sqrt_gamma * B
        
        # Call the standard DARE solver with effective matrices A_eff, B_eff,
        P_solution = self.solve_discrete_are(A_eff, B_eff, Q_gamma, R_gamma)
        return P_solution

    def compute_H_matrix(self, A, B, Q_gamma, R_gamma, P_solution_discounted_dare, gamma):
        """
        Computes the matrix H for the Q-function Q*(x,u) = z^T H z from the LAS paper,
        where H = [ Q_gamma + gamma*A^T*P*A   gamma*A^T*P*B ]
                  [ gamma*B^T*P*A           R_gamma + gamma*B^T*P*B ]
        and P is the solution to the discounted DARE.

        Parameters:
            A (ndarray): Original system matrix (n x n).
            B (ndarray): Original input matrix (n x m).
            Q_gamma (ndarray): Discounted state cost matrix (n x n).
            R_gamma (ndarray): Discounted control cost matrix (m x m).
            P_solution_discounted_dare (ndarray): Solution P to the discounted DARE (n x n).
            gamma (float): Discount factor (0 < gamma <= 1).
        
        Returns:
            H (ndarray): The computed H matrix ((n+m) x (n+m)).
        """
        if not (0 < gamma <= 1):
            raise ValueError("Discount factor gamma must be in (0, 1].")

        # P_gamma in the H matrix formula corresponds to P_solution_discounted_dare
        A_T_P = A.T @ P_solution_discounted_dare
        B_T_P = B.T @ P_solution_discounted_dare

        H11 = Q_gamma + gamma * A_T_P @ A
        H12 = gamma * A_T_P @ B
        H21 = gamma * B_T_P @ A
        H22 = R_gamma + gamma * B_T_P @ B
        
        H = np.block([
            [H11, H12],
            [H21, H22]]
        )
        return H
    