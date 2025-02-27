cov2cor(matrix(c(1^2, 0.07, 0.07, 0.1^2), nrow = 2))
        

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
