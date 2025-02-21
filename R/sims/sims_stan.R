# simulation code from https://gist.github.com/seananderson/32906dda9af81482221166449087b357
library(tidyverse); library(here); library(cmdstanr); 
library(bayesplot); library(plotly); library(mvtnorm); 
library(shinystan); library(tidybayes); library(posterior)

source("R/funs/generate_mvn.R")

# # 1. "Standard" multivariate linear model ----
# 
# set.seed(42)
# N = 400
# x = runif(N, -1, 1)
# 
# Omega = rbind( # correlation matrix
#   c(1, 0.9),
#   c(0.9, 1)
# )
# sigma = c(0.6, 0.4) # residual SDs
# Sigma = diag(sigma) %*% Omega %*% diag(sigma) # covariance matrix
# Sigma
# errors = mvtnorm::rmvnorm(N, c(0,0), Sigma)
# plot(errors)
# cor(errors) # realized correlation
# 
# y1 = -0.5 + x * 1.1 + errors[,1]
# y2 = 0.8 + x * 0.4 + errors[,2]
# plot(x, y1)
# plot(x, y2)
# plot(y1, y2)
# 
# mod = cmdstan_model(here("stan/multivariate_reg.stan"),
#                      pedantic=TRUE, 
#                      force_recompile=TRUE)
# mod$print()
# 
# dlist = list(J = 2, K = 2, N = length(y1), 
#              x = matrix(c(rep(1, N), x), ncol = 2), 
#              y = matrix(c(y1, y2), ncol = 2))
# 
# fit.multi = mod$sample(
#   data = dlist,
#   seed = 123, iter_warmup = 300,
#   iter_sampling = 500,
#   chains = 4,
#   parallel_chains = 4)
# fit.multi$diagnostic_summary()
# 
# print(fit.multi)
# # beta[1,1] is the first intercept
# # beta[1,2] is the first slope
# # beta[2,1] is the second intercept
# # beta[2,2] is the second slope
# # Omega are the elements of the correlation matrix
# # Sigma are the elements of the covariance matrix
# 
# mcmc_hist(fit.multi$draws(c("beta[1,1]", "beta[1,2]",
#                             "beta[2,1]", "beta[2,2]")))
# mcmc_hist(fit.multi$draws(c("Omega[1,2]")))
# mcmc_hist(fit.multi$draws(c("Sigma[1,1]", "Sigma[1,2]",
#                             "Sigma[2,1]", "Sigma[2,2]")))
# 
# mcmc_trace(fit.multi$draws(c("beta[1,1]", "beta[1,2]",
#                                 "beta[2,1]", "beta[2,2]")))
# mcmc_trace(fit.multi$draws(c("Omega[1,2]")))
# mcmc_trace(fit.multi$draws(c("Sigma[1,1]", "Sigma[1,2]",
#                             "Sigma[2,1]", "Sigma[2,2]")))
# 
# mcmc_pairs(fit.multi$draws(c("Sigma[1,1]", "Sigma[1,2]",
#                              "Sigma[2,1]", "Sigma[2,2]")))
# 

# 2. CRN model ----
## Simulate data ----
# Data simulation parameter
set.seed(42)

n_obs = 60 # number of observation per discret point in the gradient
x = rep(c(-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2,0.4,0.6,0.8, 1), each = n_obs) # environmental gradent

# Mean parameters
beta0 = c(1, -1) # vector of intercepts
beta1 = c(0.6, 0.4) # vector of slopes 

# Dispersion parameters
beta0_disp = c(0, 0, 0) # vector of intercepts
beta1_disp = c(0.1, 0.1, 0.8) # vector of slopes 

df_obs = data.frame(x = x) %>%
  mutate(
    id = 1:(n_obs*length(unique(x))),
    mu_y1 = beta0[1] + beta1[1] * x, # change in y1 mean with x
    mu_y2 = beta0[2] + beta1[2] * x, # change in y2 mean with x
    sigma_y1 = exp(beta0_disp[1] + beta1_disp[1] * x), # y1 sd change with x
    sigma_y2 = exp(beta0_disp[2] + beta1_disp[2] * x), # y2 sd change with x
    re_12 = tanh(beta0_disp[3] + beta1_disp[3] * x)) %>%  # y1y2 correlation change with x
  mutate(cov_12 = re_12 * sigma_y1^2 * sigma_y2^2)


df_sim = df_obs %>%
  purrr::pmap_dfr(~generate_mvn(c(...)), .id = "row_id") %>% 
  rename(Act = y1, Exp = y2)

df_sim = cbind(df_obs, df_sim)

#write.csv(df_sim, "data/data-sims/df_sim.csv", row.names = F)

plot(Act ~ Exp, df_sim)
plot(Act ~ x, df_sim)
plot(Exp ~ x, df_sim)
plot(re_12 ~ x, df_sim)
plot_ly(df_sim, z = ~x, x = ~Act, y = ~Exp, type = "scatter3d")

## Prepare data for Stan ----
#df_sim = read.csv("data/data-sims/df_sim.csv")

X = data.frame(unique(model.matrix(Act ~ x, data = df_sim)))
X_a = data.frame(model.matrix(Act ~ x, data = df_sim))
X_e = data.frame(model.matrix(Act ~ x, data = df_sim))

cn = as.numeric(table(as.numeric(as.factor(df_sim$x))))

#create empty cmat
cmat = matrix(NA, 
               nrow = length(unique(df_sim$x)), 
               ncol =  max(as.numeric(table(as.numeric(as.factor(df_sim$x))))))

#fill cmat
temporary = as.data.frame(cbind(as.numeric(as.factor(df_sim$x)),
                                 as.numeric(as.factor(df_sim$id))))
for (i in 1:length(unique(df_sim$x))) {
  cmat[i, 1:cn[i]] = temporary$V2[temporary$V1 == i]
}
cmat_n = apply(cmat, 1, FUN = function(x) sum(!is.na(x)) )
cmat[is.na(cmat)] = 0 #remove NAs

temp = t(cmat)
corder = data.frame(id = temp[temp>0], c = rep(seq(1:nrow(cmat)), times = cmat_n))

idc = match(paste0(as.numeric(as.factor(df_sim$id)),
                   as.numeric(as.factor(df_sim$x)),
                   sep="."), 
            paste0(corder$id,corder$c,sep="."))

rownames(df_sim) = NULL

stan.df =
  list(N = nrow(df_sim), # number of observations
       C = length(unique(df_sim$x)), # number of years
       I = length(unique(df_sim$id)), # number of observations / individuale
       D = 2, # number of traits
       P_y = 2, # number of predictors on correlations (including intercept)
       P_a = 2, # number of predictors on activity (including intercept)
       P_e = 2, # number of predictors on exploration (including intercept)
       
       id = as.numeric(as.factor(df_sim$id)),           
       c_id = as.numeric(as.factor(df_sim$x)),
       idc = idc,
       id_lm = as.numeric(rownames(df_sim)),
       
       X = X,
       X_a = X_a,
       X_e = X_e,
       A = diag(length(unique(df_sim$id))),
       
       cm = max(as.numeric(table(as.numeric(as.factor(df_sim$x))))),
       cmat = cmat,
       cn = cn,
       cnt = length(unique(paste(df_sim$id, df_sim$x))), # number of events
       Act = df_sim$Act,             # Activity (response variable)
       Exp = df_sim$Exp    # Exploration (response variable)
  )


## Fit to stan ----
mod_crn = cmdstan_model(here("stan/activity_exploration_crn.stan"),
                     stanc_options = list("O1"),
                     pedantic=TRUE, 
                     force_recompile=TRUE)

# mod_crn = cmdstan_model(here("~/Downloads/activity_exploration_crn.stan"),
#                         stanc_options = list("O1"),
#                         pedantic=TRUE,
#                         force_recompile=TRUE)

mod_crn$print()

fit_crn <- mod_crn$sample(
  data = stan.df,
  #output_dir = "/Downloads", 
  seed = 1234, 
  chains = 4, 
  parallel_chains = 4, 
  iter_warmup = 1000, 
  iter_sampling = 1000,
  adapt_delta = 0.99, 
  max_treedepth = 15,
  refresh = 200 # print update every 200 iters
)

fit_crn$save_object(file = "outputs/mods/sims/fit_crn.RDS")

# show summary of model output
fit_crn$summary(variables = c("B_m_a", "B_m_e", "B_cpc", "sd_G")) 



# fit_crn$save_object(file = "outputs/mods/sims/fit_crn.RDS")
# 
# fit_crn$diagnostic_summary()
# 
# print(fit_crn)
# 
## Inspect the model -----

#### The following 2 lines of code work fine for me, it doesn't throw any error.
# maybe add the "output_dir" command in the code above in case the issue was 
# because it saved the model output in a temporary folder

fit <- read_cmdstan_csv(fit_crn$output_files(),
                        variables = c("B_m_a", "B_m_e", "B_cpc", "sd_G")) 
post_draws <- posterior::as_draws_df(fit$post_warmup_draws)  

post_draws <- as_draws_df(fit_crn$draws())
post_draws <- subset_draws(post_draws,
                           variable =
                             c("B_mq_a", "B_mq_e", "B_cpcq",
                               "sd_G", "B_m_a", "B_m_e", "B_cpc"))


# post_draws contain the posterior distributions for all your parameters of 
# interest (intercept and slope for each of the 2 traits and for the correlation
# between both traits). This is not the right format to directly use Jordan's 
# post processing functions, but in the end i don't think his functions are needed,
# what they do is reshape the output to a specific format to make plots.
# I think it is easier to use the output format that you already have in 
# "postdraws", because with these you already have everything you need to make
# predictions and plots.


#### Plotting effect of your environmental gradient on the correlation between both traits

x2.sim = seq(min(stan.df$X[,2]),
             max(stan.df$X[,2]),
             by =  0.1) 

# predict the correlation across the range of your environmental value
int.sim <- matrix(rep(NA, nrow(post_draws)*length(x2.sim)), nrow = nrow(post_draws))
for(i in 1:length(x2.sim)){
  int.sim[, i] <- tanh(post_draws$`B_cpc[1,1]` + post_draws$`B_cpc[2,1]` * (x2.sim[i])) 
}

# calculate quantiles of predictions
bayes.c.eff.median <- tanh(median(post_draws$`B_cpc[1,1]`) + median(post_draws$`B_cpc[2,1]`) * (x2.sim)) 
bayes.c.eff.lower <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.05)))
bayes.c.eff.upper <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.95)))
bayes.c.eff.lower.bis <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.15)))
bayes.c.eff.upper.bis <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.85)))
bayes.c.eff.lower.bis2 <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.25)))
bayes.c.eff.upper.bis2 <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.75)))
bayes.c.eff.lower.bis3 <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.35)))
bayes.c.eff.upper.bis3 <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.65)))
bayes.c.eff.lower.bis4 <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.45)))
bayes.c.eff.upper.bis4 <- apply(int.sim, 2, function(x) quantile(x, probs = c(0.55)))
plot.dat <- data.frame(x2.sim, bayes.c.eff.median,
                       bayes.c.eff.lower, bayes.c.eff.upper,
                       bayes.c.eff.lower.bis, bayes.c.eff.upper.bis,
                       bayes.c.eff.lower.bis2, bayes.c.eff.upper.bis2,
                       bayes.c.eff.lower.bis3, bayes.c.eff.upper.bis3,
                       bayes.c.eff.lower.bis4, bayes.c.eff.upper.bis4)


p <- ggplot(plot.dat, aes(x = x2.sim, y = bayes.c.eff.median)) +
  geom_ribbon(aes(ymin = bayes.c.eff.lower.bis4, ymax = bayes.c.eff.upper.bis4), fill = "darkseagreen4", alpha = 0.1)+
  geom_ribbon(aes(ymin = bayes.c.eff.lower.bis3, ymax = bayes.c.eff.upper.bis3), fill = "darkseagreen4", alpha = 0.1)+
  geom_ribbon(aes(ymin = bayes.c.eff.lower.bis2, ymax = bayes.c.eff.upper.bis2), fill = "darkseagreen4", alpha = 0.1)+
  geom_ribbon(aes(ymin = bayes.c.eff.lower.bis, ymax = bayes.c.eff.upper.bis), fill = "darkseagreen4", alpha = 0.1)+
  geom_ribbon(aes(ymin = bayes.c.eff.lower, ymax = bayes.c.eff.upper), fill = "darkseagreen4", alpha = 0.1)+
  geom_line(color = "darkseagreen4", size = 1.8, alpha=0.6)+
  ylim(-1,1)+
  xlab("Environmetal gradient")+
  ylab("Observation-level correlation")+
  theme_bw() +
  theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank())+
  theme(plot.title = element_text(size=10, hjust = 0.5),
        axis.text.y = element_text(size=14),
        axis.text.x = element_text(size=14))
p




# fit_crn$metadata()$model_params
# # # launch_shinystan(fit_crn)
# # csv_paths = fit_crn$output_files()
# # CRN_fit = rstan::read_stan_csv(fit_crn$output_files())
# # saveRDS(CRN_fit, "CRN_fit.RDS")
# # NOT WORKING!
# # Trying to find an equivalent postprocessing procedure using {posterior} or {tidybayes}
# # What would replace the extract(CRN_fit) line so that I could follow the rest of Jordan's tutorial?
# 
# 
# ## Plots ----
# mcmc_hist(fit_crn$draws("B_m_a[1]"))
# draws <- posterior::as_draws_rvars(fit_crn$draws())
# draws <- posterior::as_draws_array(fit_crn$draws())
# mcmc_hist(draws$B_m_a)
# summarise_draws(tidy_draws(fit_crn$draws()))
# get_variables(fit_crn$draws())

## Plots ----
# summarise_draws(tidybayes::tidy_draws(fit_crn$draws()))
# get_variables(fit_crn$draws())
# mcmc_hist(fit_crn$draws("B_m_a[1]"))
# 
# draws <- as_draws_rvars(fit_crn$draws())
# draws <- as_draws_array(fit_crn$draws())
# draws <- as_draws_df(fit_crn$draws())
# df_draws <- subset_draws(draws, 
#                          variable = 
#                            c("B_mq_a", "B_mq_e", "B_cpcq", 
#                              "sd_G", "B_m_a", "B_m_e", "B_cpc"))
# 
# 
# mcmc_trace(df_draws)
# # mcmc_rank_hist(df_draws)
# mcmc_rank_overlay(df_draws)
# mcmc_rank_ecdf(df_draws)
# summarise_draws(df_draws)
