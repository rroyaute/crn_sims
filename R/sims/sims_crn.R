# simulation code from https://gist.github.com/seananderson/32906dda9af81482221166449087b357
library(plyr); library(tidyverse); library(here); library(cmdstanr)
library(bayesplot); library(plotly); library(mvtnorm)
library(shinystan); library(tidybayes); library(posterior)
library(easystats); library(patchwork)

source("R/funs/generate_mvn.R")
source("R/funs/CRN functions.R")

# Simulate data ----
# Data simulation parameter
set.seed(42)

n_obs = 30 # number of observation per discret point in the gradient
x = rep(c(-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2,0.4,0.6,0.8, 1), each = n_obs) # environmental gradient

# Mean parameters
beta0 = c(1, -1) # vector of intercepts
beta1 = c(0.6, -0.4) # vector of slopes 

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

write.csv(df_sim, "data/data-sims/df_sim.csv", row.names = F)

plot(Act ~ Exp, df_sim)
plot(Act ~ x, df_sim)
plot(Exp ~ x, df_sim)
plot(re_12 ~ x, df_sim)
plot_ly(df_sim, z = ~x, x = ~Act, y = ~Exp, type = "scatter3d")

# Prepare data for Stan ----
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
       Act = df_sim$Act,   # Activity (response variable)
       Exp = df_sim$Exp    # Exploration (response variable)
  )


# Fit to stan ----
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
# Inspect the model -----
fit_crn <- readRDS("outputs/mods/sims/fit_crn.RDS")

post_draws <- as_draws_df(fit_crn$draws())
post_draws <- subset_draws(post_draws,
                           variable =
                             c("B_mq_a", "B_mq_e", "B_cpcq",
                               "sd_G", "B_m_a", "B_m_e", "B_cpc"))
post_draws.2 = extract_samples(fit_crn)
get_variables(post_draws)
summarise_draws(post_draws)
mcmc_trace(post_draws)
mcmc_rank_overlay(post_draws)
mcmc_rank_ecdf(post_draws)


# Plots ----
# Store dose gradient based on min and max value
Dose.pred = seq(min(stan.df$X[,2]),
                max(stan.df$X[,2]),
                by =  0.1)
# Create empty dataframe of Doses x nb of posterior draws
post_pred = data.frame(Dose = rep(unique(Dose.pred), 
                                  nrow(post_draws)))

# x2.sim = seq(min(stan.df$X[,2]),
#              max(stan.df$X[,2]),
#              by =  0.1)
# int.sim <- matrix(rep(NA, nrow(post_draws)*length(x2.sim)),
#                   nrow = nrow(post_draws))

## Coefficient plot (TODO) ----

## Plotting the change in mean with the gradient ----

### Activity ----
# predict the mean across the range of your environmental value
post_pred = post_pred %>% 
  mutate(pred = post_draws$`B_m_a[1]` + post_draws$`B_m_a[2]` * Dose)

# calculate quantiles of predictions
df.post.summary = post_pred %>% 
  group_by(Dose) %>% 
  median_qi(.width = c(.9))
df.post.summary

fig_act_mu = post_pred %>% 
  ggplot(aes(x = Dose, y = pred)) +
  stat_lineribbon() +
  scale_fill_brewer() +
  # ylim(-1,1)+
  xlab("Environmetal gradient")+
  ylab("Average Activity")+
  theme_bw() +
  theme(legend.position = "none")
fig_act_mu

### Exploration ----
# predict the mean across the range of your environmental value
post_pred = post_pred %>% 
  mutate(pred = post_draws$`B_m_e[1]` + post_draws$`B_m_e[2]` * Dose)

# for(i in 1:length(x2.sim)){
#   int.sim[, i] <- post_draws$`B_m_e[1]` + post_draws$`B_m_e[2]` * (x2.sim[i])
# }

# calculate quantiles of predictions
df.post.summary = post_pred %>% 
  group_by(Dose) %>% 
  median_qi(.width = c(.9))
df.post.summary

fig_explo_mu = post_pred %>% 
  ggplot(aes(x = Dose, y = pred)) +
  stat_lineribbon() +
  scale_fill_brewer() +
  # ylim(-1,1)+
  xlab("Environmetal gradient")+
  ylab("Average Exploration")+
  theme_bw() +
  theme(legend.position = "none")
fig_explo_mu

## Plotting the change in variance with the gradient ----
### Activity ----
# calculate quantiles of predictions
df.post.summary = post_draws[,7:17] %>% 
  data.frame() %>% 
  pivot_longer(names_to = "Dose", cols = 1:11) %>% 
  mutate(Dose = rep(c(-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2,0.4,0.6,0.8, 1), 
                    nrow(post_draws))) %>% 
  group_by(Dose) %>% 
  median_qi(.width = c(.9))
df.post.summary

fig_act_sd = post_draws[,7:17] %>% 
  data.frame() %>% 
  pivot_longer(names_to = "Dose", cols = 1:11) %>% 
  mutate(Dose = rep(c(-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2,0.4,0.6,0.8, 1), 
                    nrow(post_draws))) %>%  
  ggplot(aes(x = Dose, y = value)) +
  stat_halfeye() +
  scale_fill_brewer() +
  xlab("Environmetal gradient")+
  ylab("Activity sd")+
  theme_bw() +
  theme(legend.position = "none")
fig_act_sd

### Exploration ----
# calculate quantiles of predictions
df.post.summary = post_draws[,18:28] %>% 
  data.frame() %>% 
  pivot_longer(names_to = "Dose", cols = 1:11) %>% 
  mutate(Dose = rep(c(-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2,0.4,0.6,0.8, 1), 
                    nrow(post_draws))) %>% 
  group_by(Dose) %>% 
  median_qi(.width = c(.5, .8, .95))


fig_explo_sd = post_draws[,18:28] %>% 
  data.frame() %>% 
  pivot_longer(names_to = "Dose", cols = 1:11) %>% 
  mutate(Dose = rep(c(-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2,0.4,0.6,0.8, 1), 
                    nrow(post_draws))) %>%  
  ggplot(aes(x = Dose, y = value)) +
  stat_halfeye() +
  scale_fill_brewer() +
  xlab("Environmetal gradient")+
  ylab("Exploration sd")+
  theme_bw() +
  theme(legend.position = "none")
fig_explo_sd

## Plotting the change in correlation with the gradient ----
post_pred = post_pred %>% 
  mutate(pred = tanh(post_draws$`B_cpc[1,1]` + 
                       post_draws$`B_cpc[2,1]` * Dose))

# calculate quantiles of predictions
df.post.summary = post_pred %>% 
  group_by(Dose) %>% 
  median_qi(.width = c(.9))
df.post.summary

fig_act_explo_corr =  post_pred %>% 
  ggplot(aes(x = Dose, y = pred)) +
  stat_lineribbon() +
  scale_fill_brewer() +
  ylim(-1,1)+
  xlab("Environmetal gradient")+
  ylab("Observation-level correlation")+
  theme_bw() +
  theme(legend.position = "none")
fig_act_explo_corr


## Export figures ----
fig_mu = fig_act_mu + fig_explo_mu
fig_sd = fig_act_sd + fig_explo_sd

fig_all_effects = 
  (fig_act_mu + fig_explo_mu) /
  (fig_act_sd + fig_explo_sd) /
  fig_act_explo_corr

ggsave("outputs/figs/fig_mu.png", fig_mu)
ggsave("outputs/figs/fig_sd.png", fig_sd)
ggsave("outputs/figs/fig_all_effects.png", fig_all_effects)



##


# Plots with extract_samples function (TODO) ----
traits = seq(1:stan.df$D)

v_b = v_beta_f(x = stan.df$X, 
               traits = 1:4, 
               v_b = post_draws.2)

