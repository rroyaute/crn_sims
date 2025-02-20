data {
  int <lower=1> K; // Number of shared observations between y_n
  int <lower=1> J; // Number of x predictors
  int <lower=0> N; // Number of responses
  
  array[N] vector[J] x; // J-row vector of predictors
  array[N] vector[K] y; // K-row vector of responses
}

parameters { 
  matrix[K, J] beta; 
  cholesky_factor_corr[K] L_Omega;
  vector<lower=0>[K] L_sigma;
} 
model {
  array[N] vector[K] mu;
  for (n in 1:N) 
    mu[n] = beta * x[n]; // Multivariate regression
  
  to_vector(beta) ~ normal(0, 10);
  L_Omega ~ lkj_corr_cholesky(4); // Cholesky decomposition of covariance matrix: correlations
  L_sigma ~ exponential(1); // Cholesky decomposition of covariance matrix: standard deviations
  
  y ~ multi_normal_cholesky(mu, diag_pre_multiply(L_sigma, L_Omega)); // Sample from Cholesky decomposition
}
generated quantities {
  matrix[K, K] Omega; // Response residual correlation matrix
  matrix[K, K] Sigma; // Variance-Covariance matrix
  Omega = multiply_lower_tri_self_transpose(L_Omega);
  Sigma = quad_form_diag(Omega, L_sigma); 
}
