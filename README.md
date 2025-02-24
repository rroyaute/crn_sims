# Simulations for Covariance Reaction Norms for Dose-Response data


## Ongoing issues

- Difference between B_mq and B_m, B_cpc and B_cpcq matrices / vectors
- Why no LKJ prior in the model block ?
- Making a coefficient plot for mean, sd and correlations, where do we take the coefficients from?
	- $\beta{_\mu}$ and $\beta{_\rho}$ easy enough but no clue for where to fetch $\beta{_\sigma}$ from
	- Why do we not see $log(\beta{_\sigma})$ formula in the efficient specification
- Making summary table of coefficients +/- CIs
- Do I need to add betas for variances in the stan code or is it already accounted for?
- How to expand to a Gaussian-Beta multivariate model
- Why do we need a relationship matrix A if we do not have a pedigree?
- How is A assumed to be distributed?