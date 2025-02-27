df_obs[1,] # Dose = 0
df_obs[120,] # Dose max  = 30

cov2cor(matrix(c(1^2, 0.07, 0.07, 0.1^2), nrow = 2))
cov2cor(matrix(c(1.020201^2, 0.05390475, 0.05390475, 0.08187308^2), 
               nrow = 2))
# Simulate dose = 0 ----
## From rmvnorm() ----
dat = 
  rmvnorm(30, c(5.010635, 0),
        matrix(c(1^2, 0.07, 0.07, 0.1^2), nrow = 2))

dat = dat %>% 
  data.frame() %>% 
  mutate(Act = exp(X1),
         Exp = inv_logit(X2))
dat %>%
  ggplot(aes(X1, X2)) +
  geom_point()

dat %>%
  ggplot(aes(log(Act), Exp)) +
  geom_point()

dat %>%
  ggplot(aes(Act, Exp)) +
  geom_point()

cor(dat$X1, dat$X2)
cor(log(dat$Act), dat$Exp)

## From generate_mvn() ----
dat = df_obs[1:30,] %>% 
  pmap_dfr(~generate_mvn(c(...)), .id = "row_id") 

dat = dat %>% 
  data.frame() %>% 
  mutate(Act = exp(y1),
         Exp = inv_logit(y2))
dat %>%
  ggplot(aes(y1, y2)) +
  geom_point()

dat %>%
  ggplot(aes(log(Act), Exp)) +
  geom_point()

dat %>%
  ggplot(aes(Act, Exp)) +
  geom_point()

cor(dat$y1, dat$y2)
cor(log(dat$Act), dat$Exp)


# Simulate dose = 30 ----
## From rmvnorm() ----
dat = 
  rmvnorm(30, c(5.002603, -0.3227734),
          matrix(c(1^2, 0.07, 0.07, 0.1^2), nrow = 2))


dat = dat %>% 
  data.frame() %>% 
  mutate(Act = exp(X1),
         Exp = inv_logit(X2))
dat %>%
  ggplot(aes(X1, X2)) +
  geom_point()


dat %>%
  ggplot(aes(log(Act), Exp)) +
  geom_point()

dat %>%
  ggplot(aes(Act, Exp)) +
  geom_point()


cor(dat$X1, dat$X2)
cor(log(dat$Act), dat$Exp)

## From generate_mvn() ----
dat = df_obs[120:150,] %>% 
  pmap_dfr(~generate_mvn(c(...)), .id = "row_id") 

dat = dat %>% 
  data.frame() %>% 
  mutate(Act = exp(y1),
         Exp = inv_logit(y2))
dat %>%
  ggplot(aes(y1, y2)) +
  geom_point()

dat %>%
  ggplot(aes(log(Act), Exp)) +
  geom_point()

dat %>%
  ggplot(aes(Act, Exp)) +
  geom_point()

cor(dat$y1, dat$y2)
cor(log(dat$Act), dat$Exp)


