# 1. "Standard" multivariate linear model ----

set.seed(42)
N = 400
x = runif(N, -1, 1)

Omega = rbind( # correlation matrix
  c(1, 0.9),
  c(0.9, 1)
)
sigma = c(0.6, 0.4) # residual SDs
Sigma = diag(sigma) %*% Omega %*% diag(sigma) # covariance matrix
Sigma
errors = mvtnorm::rmvnorm(N, c(0,0), Sigma)
plot(errors)
cor(errors) # realized correlation

y1 = -0.5 + x * 1.1 + errors[,1]
y2 = 0.8 + x * 0.4 + errors[,2]
plot(x, y1)
plot(x, y2)
plot(y1, y2)

mod = cmdstan_model(here("stan/multivariate_reg.stan"),
                     pedantic=TRUE,
                     force_recompile=TRUE)
mod$print()

dlist = list(J = 2, K = 2, N = length(y1),
             x = matrix(c(rep(1, N), x), ncol = 2),
             y = matrix(c(y1, y2), ncol = 2))

fit.multi = mod$sample(
  data = dlist,
  seed = 123, iter_warmup = 300,
  iter_sampling = 500,
  chains = 4,
  parallel_chains = 4)
fit.multi$diagnostic_summary()

print(fit.multi)
# beta[1,1] is the first intercept
# beta[1,2] is the first slope
# beta[2,1] is the second intercept
# beta[2,2] is the second slope
# Omega are the elements of the correlation matrix
# Sigma are the elements of the covariance matrix

mcmc_hist(fit.multi$draws(c("beta[1,1]", "beta[1,2]",
                            "beta[2,1]", "beta[2,2]")))
mcmc_hist(fit.multi$draws(c("Omega[1,2]")))
mcmc_hist(fit.multi$draws(c("Sigma[1,1]", "Sigma[1,2]",
                            "Sigma[2,1]", "Sigma[2,2]")))

mcmc_trace(fit.multi$draws(c("beta[1,1]", "beta[1,2]",
                                "beta[2,1]", "beta[2,2]")))
mcmc_trace(fit.multi$draws(c("Omega[1,2]")))
mcmc_trace(fit.multi$draws(c("Sigma[1,1]", "Sigma[1,2]",
                            "Sigma[2,1]", "Sigma[2,2]")))

mcmc_pairs(fit.multi$draws(c("Sigma[1,1]", "Sigma[1,2]",
                             "Sigma[2,1]", "Sigma[2,2]")))


