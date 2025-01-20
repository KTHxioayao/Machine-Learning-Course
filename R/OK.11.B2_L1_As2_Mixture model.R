# Set parameters for the EM algorithm
set.seed(1234567890) # Ensure reproducibility
max_it <- 100 # Maximum number of EM iterations
min_change <- 0.1 # Minimum change in log-likelihood to stop iterations
n <- 1000 # Number of training data points
D <- 10 # Number of dimensions

# Initialize the training data and true model parameters
x <- matrix(nrow = n, ncol = D) # Matrix to store training data
true_pi <- vector(length = 3) # True mixing coefficients
true_mu <- matrix(nrow = 3, ncol = D) # True conditional distributions

# Set true mixing coefficients (equal probabilities for each cluster)
true_pi <- c(1 / 3, 1 / 3, 1 / 3)

# Define the true conditional probabilities for each cluster
true_mu[1, ] <- c(0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9, 0.1, 1) # Cluster 1
true_mu[2, ] <- c(0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0) # Cluster 2
true_mu[3, ] <- rep(0.5, D) # Cluster 3

# Plot the true conditional distributions of each cluster
plot(
  true_mu[1, ],
  type = "o",
  col = "blue",
  ylim = c(0, 1),
  xlab = "Dimension Index",
  ylab = "Probability",
  main = "True Conditional Distributions"
)
points(true_mu[2, ], type = "o", col = "red")
points(true_mu[3, ], type = "o", col = "green")
legend(
  "topright",
  legend = c("Cluster 1", "Cluster 2", "Cluster 3"),
  col = c("blue", "red", "green"),
  lty = 1
)

# Generate the training data
for (i in 1:n) {
  m <- sample(1:3, 1, prob = true_pi) # Randomly assign each point to a cluster
  for (d in 1:D) {
    x[i, d] <- rbinom(1, 1, true_mu[m, d]) # Generate binary data based on the cluster's mean
  }
}

# Initialize EM algorithm parameters
M <- 3 # Number of clusters
w <- matrix(nrow = n, ncol = M) # Responsibilities (weights) for each cluster
pi <- vector(length = M) # Mixing coefficients
mu <- matrix(nrow = M, ncol = D) # Conditional distributions (means)

# Random initialization of parameters
pi <- runif(M, 0.49, 0.51) # Random initialization of mixing coefficients
pi <- pi / sum(pi) # Normalize mixing coefficients
for (m in 1:M) {
  mu[m, ] <- runif(D, 0.49, 0.51) # Random initialization of conditional means
}

# Log-likelihood to track progress
llik <- vector(length = max_it)

# EM algorithm iterations
for (it in 1:max_it) {
  plot(mu[1, ],
       type = "o",
       col = "blue",
       ylim = c(0, 1))
  points(mu[2, ], type = "o", col = "red")
  points(mu[3, ], type = "o", col = "green")#This line should be ignored if K=2
  #points(mu[4,], type="o", col="yellow")#This line should be ignored if K=2 or k=3
  Sys.sleep(0.5)

  # E-step: Computation of the weights
  for (i in 1:n) {
    numerators <- sapply(1:M, function(m) {
      pi[m] * prod(mu[m, ] ^ x[i, ] * (1 - mu[m, ]) ^ (1 - x[i, ]))
    })
    # 分母是分子的总和
    denominator <- sum(numerators)
    # 计算每个簇的责任值
    w[i, ] <- numerators / denominator
  }

# OR for (i in 1:n) {
#  for (m in 1:M) {
#    numerator <- pi[m] * prod(mu[m, ]^x[i, ] * (1 - mu[m, ])^(1 - x[i, ]))
#   denominator <- sum(sapply(1:M, function(k) {
#      pi[k] * prod(mu[k, ]^x[i, ] * (1 - mu[k, ])^(1 - x[i, ]))
#    }))
#   w[i, m] <- numerator / denominator
#  }
#}

  #Log likelihood computation.
  llik[it] <- sum(sapply(1:n, function(i) {
    log(sum(sapply(1:M, function(m) {
      pi[m] * prod(mu[m, ] ^ x[i, ] * (1 - mu[m, ]) ^ (1 - x[i, ]))
    })))
  }))

  cat("iteration: ", it, "log likelihood: ", llik[it], "\n")
  flush.console()

  # Stop if the log likelihood has not changed significantly
  if (it > 1 && abs(llik[it] - llik[it - 1]) < min_change) {
    cat("Converged at iteration", it, "\n")
    break
  }

  #M-step: ML parameter estimation from the data and weights
  for (m in 1:M) {
    pi[m] <- sum(w[, m]) / n
    for (d in 1:D) {
      mu[m, d] <- sum(w[, m] * x[, d]) / sum(w[, m])
    }
  }
}
pi
mu
plot(llik[1:it], type = "o")

#We can get results for different K values.When M is equal to 2, the coeﬀicient (pi) approaches 0.5, meaning
#that the two clusters contribute nearly equally to the data. Although there are significant differences in the
#conditional distribution (mu) on each dimension,there is a big deviation from the true value, resulting in
#underfitting.When M is equal to 3, the convergence rate is slightly slower than when M is equal to 2, but the
#mu is more dispersed and closer to the true value, which can accurately capture the true structure of the data
#without being overly complicated.When M is equal to 4, the log-likelihood converges at the 44th iteration,
#and the convergence speed is the slowest, which indicates that the complexity of the model is significantly
#increased, and the weight of some clusters (such as pi1 and pi2) is very small, which indicates that the data
#may be unnecessarily split and overfitting may occur.
