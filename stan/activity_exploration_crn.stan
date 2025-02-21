functions {

//functions are used fom prior work by
//Dan Schrage (https://gitlab.com/dschrage/rcovreg)
  
  real sum_square_x(matrix x, int i, int j) {
    int j_prime;
    real sum_x = 0;
    if(j==1) return(sum_x);

    j_prime = 1;
    while(j_prime < j) {
      sum_x += x[i,j_prime]^2;
      j_prime += 1;
    }
    return(sum_x);
  }
  
  matrix lkj_to_chol_corr(row_vector constrained_reals, int ntrait) {
    int z_counter;
    matrix[ntrait,ntrait] x;

    z_counter = 1;
    x[1,1] = 1;
    for(j in 2:ntrait) {
      x[1,j] = 0;
    }
    for(i in 2:ntrait) {
      for(j in 1:ntrait) {
        if(i==j) {
          x[i,j] = sqrt(1 - sum_square_x(x, i, j));
        } else if(i > j) {
          x[i,j] = constrained_reals[z_counter]*sqrt(1 - sum_square_x(x, i, j));
          z_counter += 1;
        } else {
          x[i,j] = 0;
        }
      }
    }
    return(x);
  }
}

data {
  int<lower=1> N; // number of observations
  int<lower=1> C; // number of concentrations
  int<lower=1> I; // number of individuals
  int<lower=1> D; // number of traits dimensions
  int<lower=0> P_y; // number of predictors on correlations (1)
  int<lower=1> P_a; // number of predictors on activity (1)
  int<lower=0> P_e; // number of predictors on exploration (1)
  
  array[N] int<lower=0> id; // index linking observations to individuals
  array[N] int<lower=0> c_id; // index linking observations to contexts
  array[N] int<lower=0> idc; // index linking individuals to positions in cmat
  array[N] int<lower=0> id_lm; // index of observations
  
  matrix[C,P_y] X; //environmental predictor matrix (+ intercept) on correlation
  matrix[N,P_a] X_a; //environmental predictor matrix (+ intercept) on activity
  matrix[N,P_e] X_e; //environmental predictor matrix (+ intercept) on exploration
  matrix[I,I] A; //relatedness matrix
  
  int<lower=1> cm; //max number of individuals observed in a context
  array[C, cm] int cmat; //matrix with all individuals observed in each context (row)
  array [C] int<lower=0> cn; //count of individuals observed per context
  int<lower=1> cnt; //total number of individuals across contexts
  
  array[N] real Act; // activity (distance traveled)
  array[N] real Exp; // exploration (% surface covered)
}

transformed data{
  matrix[I, I] LA = cholesky_decompose(A);
  int ncor = (D*(D-1))/2; //unique cov/cor parameters
  // Compute, thin, and then scale QR decomposition
  matrix[C, P_y] Q = qr_thin_Q(X) * sqrt(C-1);
  matrix[P_y, P_y] R = qr_thin_R(X) / sqrt(C-1);
  matrix[P_y, P_y] R_inv = inverse(R);
  
  matrix[N, P_a] Q_a = qr_thin_Q(X_a) * sqrt(N-1);
  matrix[P_a, P_a] R_a = qr_thin_R(X_a) / sqrt(N-1);
  matrix[P_a, P_a] R_inv_a = inverse(R_a);
  
  matrix[N, P_e] Q_e = qr_thin_Q(X_e) * sqrt(N-1);
  matrix[P_e, P_e] R_e = qr_thin_R(X_e) / sqrt(N-1);
  matrix[P_e, P_e] R_inv_e = inverse(R_e);
}

parameters { 
  //fixed effects
  vector[P_a] B_mq_a; //RN of means for activity
  vector[P_e] B_mq_e; //RN of means for exploration
  matrix[P_y, ncor] B_cpcq; //RN of canonical partial correlations

  //random effects
  matrix[cnt, D] Z_G; //all context-specific additive genetic values
  array[C] vector<lower=0>[D] sd_G; //sd of ind effects
  
}

model {
  //predicted values from reaction norms
  //growth
  vector[N] mu_act =  Q_a * B_mq_a;
  
  //fecundity
  vector[N] mu_explo =  Q_e * B_mq_e;
                       
  //correlations (expressed as canonical partial correlations)
  matrix[C, ncor] cpc_G = tanh(Q * B_cpcq);
  
  //initialize mean linear predictors
  vector[N] mu_a = mu_act[id_lm];
  vector[N] mu_e = mu_explo[id_lm];

  //scale context-specific multivariate additive genetic effects
  matrix[cnt, D] mat_G;
  int pos = 1; //keep track of position 1:cnt
  for(c in 1:C){
      mat_G[pos:(pos+cn[c]-1)] = 
      LA[cmat[c,1:cn[c]],cmat[c,1:cn[c]]] * Z_G[pos:(pos+cn[c]-1)] * diag_pre_multiply(sd_G[c],lkj_to_chol_corr(cpc_G[c], D))';
      pos = pos + cn[c];   
  }
        
//add context-specific genetic effects to linear predictors
  for(n in 1:N){
  mu_a[n]  += col(mat_G,1)[idc[n]];
  mu_e[n]  += col(mat_G,2)[idc[n]];
  }
  
                  
//likelihood 
  Act ~ normal(mu_a, 0.05);
  Exp ~ normal(mu_e, 0.05);


//priors
  to_vector(B_mq_a) ~ normal(0,1);
  to_vector(B_mq_e) ~ normal(0,1);
  to_vector(B_cpcq) ~ normal(0,1);
  to_vector(Z_G) ~ std_normal();

  
  for(c in 1:C){
  sd_G[c] ~ exponential(2);
  }
}

generated quantities{
  vector[P_a] B_m_a; //mean RN parameters for X
  vector[P_e] B_m_e; //mean RN parameters for X
  matrix[P_y,ncor] B_cpc; //partial correlation RN parameters for X

  B_m_a= R_inv_a * B_mq_a;
  B_m_e= R_inv_e * B_mq_e;

  for(d in 1:ncor){
    B_cpc[,d]= R_inv * B_cpcq[,d];
    }
}
