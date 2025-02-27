# simulation code from https://gist.github.com/seananderson/32906dda9af81482221166449087b357
library(plyr); library(tidyverse); library(here); library(cmdstanr)
library(bayesplot); library(plotly); library(mvtnorm)
library(shinystan); library(tidybayes); library(posterior)
library(easystats); library(patchwork)

source("R/funs/generate_mvn.R")
source("R/funs/CRN functions.R")

# Simulate data ----
# Data simulation parameter
set.seed(24)

n_obs = 30 # number of observation per discret point in the gradient
Dose_range = c(0, 10, 15, 20, 30)
Dose_rep = rep(Dose_range, each = n_obs) # environmental gradient

# Mean parameters
beta0 = c(150, .5) # vector of intercepts
beta1 = c(-0.06, -0.004) # vector of slopes 

# Dispersion parameters
beta0_disp = c(log(1), log(.1), atanh(.7)) # vector of intercepts
beta1_disp = c(0.001, -0.01, -0.005) # vector of slopes 

df_obs = data.frame(Dose = Dose_rep) %>%
  mutate(
    id = 1:(n_obs*length(unique(Dose_rep))),
    mu_y1 = log(beta0[1] + beta1[1] * Dose_rep), # change in y1 mean with x on log scale
    mu_y2 = logit(beta0[2] + beta1[2] * Dose_rep), # change in y2 mean with x
    sigma_y1 = exp(beta0_disp[1] + beta1_disp[1] * Dose_rep), # y1 sd change with x
    sigma_y2 = exp(beta0_disp[2] + beta1_disp[2] * Dose_rep), # y2 sd change with x
    re_12 = tanh(beta0_disp[3] + beta1_disp[3] * Dose_rep)) %>%  # y1y2 correlation change with x
  mutate(cov_12 = re_12 * sigma_y1 * sigma_y2)


df_sim = df_obs %>%
  pmap_dfr(~generate_mvn(c(...)), .id = "row_id") %>% 
  rename(Act_log = y1, Exp_logit = y2) %>% 
  mutate(Act = exp(Act_log),
         Exp = inv_logit(Exp_logit),
         Act_log_sc = as.numeric(scale(Act_log)))

df_sim = cbind(df_obs, df_sim)
df_sim = df_sim %>% 
  mutate(Dose_sc = as.numeric(scale(Dose)))

write.csv(df_sim, "data/data-sims/df_sim_dose.csv", row.names = F)

plot(Act_log_sc ~ Exp, df_sim)
plot(Act_log_sc ~ Dose, df_sim)
plot(Exp ~ Dose, df_sim)
plot(re_12 ~ Dose, df_sim)
plot_ly(df_sim, z = ~Dose, x = ~Act_log_sc, y = ~Exp, type = "scatter3d")

df_sim %>% 
  ggplot(aes(x = log(Act), y = Exp)) +
  geom_point() +
  geom_smooth(method = "lm", se = F) +
  facet_wrap(~Dose) +
  theme_bw()

df_sim %>%
  group_by(Dose) %>% 
  summarise(cor = cor(Act_log_sc, Exp))


# Prepare data for Stan ----
#df_sim = read.csv("data/data-sims/df_sim.csv")
Dose_sc_range = unique(df_sim$Dose_sc)

X = data.frame(unique(model.matrix(Act_log_sc ~ Dose_sc, data = df_sim)))
X_a = data.frame(model.matrix(Act_log_sc ~ Dose_sc, data = df_sim))
X_e = data.frame(model.matrix(Exp ~ Dose_sc, data = df_sim))

cn = as.numeric(table(as.numeric(as.factor(df_sim$Dose_sc))))

#create empty cmat
cmat = matrix(NA, 
              nrow = length(unique(df_sim$Dose_sc)), 
              ncol =  max(as.numeric(table(as.numeric(as.factor(df_sim$Dose_sc))))))

#fill cmat
temporary = as.data.frame(cbind(as.numeric(as.factor(df_sim$Dose_sc)),
                                as.numeric(as.factor(df_sim$id))))
for (i in 1:length(unique(df_sim$Dose_sc))) {
  cmat[i, 1:cn[i]] = temporary$V2[temporary$V1 == i]
}
cmat_n = apply(cmat, 1, FUN = function(x) sum(!is.na(Dose_sc_range)) )
cmat[is.na(cmat)] = 0 #remove NAs

temp = t(cmat)
corder = data.frame(id = temp[temp>0], c = rep(seq(1:nrow(cmat)), times = cmat_n))

idc = match(paste0(as.numeric(as.factor(df_sim$id)),
                   as.numeric(as.factor(df_sim$Dose_sc)),
                   sep="."), 
            paste0(corder$id,corder$c,sep="."))

rownames(df_sim) = NULL

# TODO: # investigate why idc is not created properly

stan.df =
  list(N = nrow(df_sim), # number of observations
       C = length(unique(df_sim$Dose_sc)), # number of doses
       I = length(unique(df_sim$id)), # number of observations / individuale
       D = 2, # number of traits
       P_y = 2, # number of predictors on correlations (including intercept)
       P_a = 2, # number of predictors on activity (including intercept)
       P_e = 2, # number of predictors on exploration (including intercept)
       
       id = as.numeric(as.factor(df_sim$id)),           
       c_id = as.numeric(as.factor(df_sim$Dose_sc)),
       idc = as.numeric(as.factor(df_sim$id)), 
       id_lm = as.numeric(rownames(df_sim)),
       
       X = X,
       X_a = X_a,
       X_e = X_e,
       A = diag(length(unique(df_sim$id))),
       
       cm = max(as.numeric(table(as.numeric(as.factor(df_sim$Dose_sc))))),
       cmat = cmat,
       cn = cn,
       cnt = length(unique(paste(df_sim$id, df_sim$Dose_sc))), # number of events
       Act = df_sim$Act_log_sc,   # Activity (response variable)
       Exp = df_sim$Exp    # Exploration (response variable)
  )


# Fit to stan ----
mod_crn_dose = cmdstan_model(here("stan/activity_exploration_crn.stan"),
                        stanc_options = list("O1"),
                        pedantic=TRUE, 
                        force_recompile=TRUE)

mod_crn_dose$print()

fit_crn_dose <- mod_crn_dose$sample(
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

fit_crn_dose$save_object(file = "outputs/mods/sims/fit_crn_dose.RDS")

# show summary of model output
fit_crn_dose$summary(variables = c("B_m_a", "B_m_e", "B_cpc", "sd_G")) 
fit_crn_dose$diagnostic_summary()
print(fit_crn_dose)
 
# Inspect the model -----
fit_crn_dose <- readRDS("outputs/mods/sims/fit_crn_dose.RDS")

post_draws <- as_draws_df(fit_crn_dose$draws())
post_draws <- subset_draws(post_draws,
                           variable =
                             c("B_mq_a", "B_mq_e", "B_cpcq",
                               "sd_G", "B_m_a", "B_m_e", "B_cpc"))
post_draws.2 = extract_samples(fit_crn_dose)
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
  mutate(pred_data_scale = exp((pred * sd(log(df_sim$Act)) + 
                                  mean(log(df_sim$Act)))),
  Dose_data_scale = Dose * sd(df_sim$Dose) +
    mean(df_sim$Dose)) %>% 
  ggplot(aes(x = Dose_data_scale , y = pred_data_scale)) +
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

# calculate quantiles of predictions
df.post.summary = post_pred %>% 
  group_by(Dose) %>% 
  median_qi(.width = c(.9))
df.post.summary

fig_explo_mu = post_pred %>% 
  mutate(Dose_data_scale = Dose * sd(df_sim$Dose) +
           mean(df_sim$Dose)) %>% 
  ggplot(aes(x = Dose_data_scale, y = pred)) +
  stat_lineribbon() +
  scale_fill_brewer() +
  ylim(0,1)+
  xlab("Environmetal gradient")+
  ylab("Average Exploration")+
  theme_bw() +
  theme(legend.position = "none")
fig_explo_mu

## Plotting the change in variance with the gradient ----
### Activity ----
# calculate quantiles of predictions
df.post.summary = post_draws[,7:11] %>% 
  data.frame() %>% 
  pivot_longer(names_to = "Dose", cols = 1:5) %>% 
  mutate(Dose = rep(unique(df_sim$Dose), nrow(post_draws))) %>% 
  group_by(Dose) %>% 
  median_qi(.width = c(.9))
df.post.summary

fig_act_sd = post_draws[,7:11] %>% 
  data.frame() %>% 
  pivot_longer(names_to = "Dose", cols = 1:5) %>% 
  mutate(Dose = rep(unique(df_sim$Dose), nrow(post_draws))) %>% 
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
df.post.summary = post_draws[,12:16] %>% 
  data.frame() %>% 
  pivot_longer(names_to = "Dose", cols = 1:5) %>% 
  mutate(Dose = rep(unique(df_sim$Dose), nrow(post_draws))) %>% 
  group_by(Dose) %>% 
  median_qi(.width = c(.9))
df.post.summary

fig_explo_sd = post_draws[,12:16] %>% 
  data.frame() %>% 
  pivot_longer(names_to = "Dose", cols = 1:5) %>% 
  mutate(Dose = rep(unique(df_sim$Dose), nrow(post_draws))) %>% 
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
  mutate(Dose_data_scale = Dose * sd(df_sim$Dose) +
           mean(df_sim$Dose)) %>% 
  ggplot(aes(x = Dose_data_scale, y = pred)) +
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

ggsave("outputs/figs/fig_mu_dose.png", fig_mu)
ggsave("outputs/figs/fig_sd_dose.png", fig_sd)
ggsave("outputs/figs/fig_act_explo_corr_dose.png", fig_act_explo_corr)
ggsave("outputs/figs/fig_all_effects_dose.png", fig_all_effects)
