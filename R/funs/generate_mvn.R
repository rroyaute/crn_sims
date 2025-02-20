# Function for generating multivariate normal values for each row of a dataframe

generate_mvn <- function(row) {
  # Extract means vector
  means <- c(row["mu_y1"], row["mu_y2"])
  
  # Construct covariance matrix
  sigma <- matrix(
    c(row["sigma_y1"], row["cov_12"],
      row["cov_12"], row["sigma_y2"]),
    nrow = 2
  )
  
  # Generate one sample
  result <- rmvnorm(n = 1, mean = means, sigma = sigma)
  
  # Return as dataframe row
  data.frame(
    y1 = result[1],
    y2 = result[2]
  )
}
