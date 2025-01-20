#assignment 2
#task1

#install.packages("caret")
library(caret)

data <- read.csv("parkinsons.csv")
n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.6))
train_data = data[id, ]
test_data = data[-id, ]

scaler = preProcess(train_data)
train_data_scaled = predict(scaler, train_data)
test_data_scaled = predict(scaler, test_data)



#task2
model <- lm(motor_UPDRS ~ . - subject. - age - sex - test_time - total_UPDRS ,
            train_data_scaled)
train_prediction <- predict(model, train_data_scaled)
train_mse <- mean((train_prediction - train_data_scaled$motor_UPDRS) ^ 2)

test_prediction <- predict(model, test_data_scaled)
test_mse <- mean((test_prediction - test_data_scaled$motor_UPDRS) ^ 2)
#A smaller p-value means a larger contribution to the model.
#Specifically, Jitter.Abs., Shimmer.APQ5,
#Shimmer.APQ11, NHR, HNR, DFA and PPE contribute significantly to the model.

#task3
#a
logLikelihood <- function(theta, sigma, x, y) {
  n <- length(y)
  predictions <- x %*% theta
  residuals <- y - predictions
  log_likelihood <- -n / 2 * log(2 * pi * sigma ^ 2) - 1 / (2 * sigma ^
                                                              2) * sum(residuals ^ 2)
  return(as.numeric(log_likelihood))
}
#b
ridge <- function(theta, sigma, lambda, x, y) {
  log_likelihood <- logLikelihood(theta, sigma, x, y)
  ridge_penalty <- lambda * sum(theta ^ 2)# Ridge penalty: λ‖θ‖²
  return(-log_likelihood + ridge_penalty)
}


#c
#在 Ridge 回归优化中，我们希望同时找到最佳的：
#θ：模型的系数向量（与特征 𝑋X 的维度一致）。
#σ：模型的标准差（标量）。由于 optim() 函数只能接受单个向量作为参数，所以我们需要将 θ 和
#𝜎合并到一个向量 params 中进行优化。优化完成后，再通过分割这个向量来提取 θ 和 σ。
#参数分割逻辑params[1:p]: 提取前 p 个元素，表示 θ，这里 p 是特征的个数。
#params[p + 1]: 提取最后一个元素，表示σ。

ridgeopt <- function(lambda, x, y) {
  n <- ncol(x)# 特征的数量
  init_params <- c(rep(0, n), 1)# 初始化θ为0向量，σ为1,不同初始值会给一样的结果，但是需要保证n+1个参数

  # Objective function for optimization (negative penalized log-likelihood)
  ridge_obj <- function(params) {
    theta <- params[1:n]
    sigma <- params[n + 1]
    return(ridge(theta, sigma, lambda, x, y))
  }
  # Optimization using optim() with method = "BFGS"
  #init_params is usde as the initial values for the optimization (fn)
  opt <- optim(par = init_params, fn = ridge_obj, method = "BFGS")

  # Extract optimized theta and sigma
  theta_opt <- opt$par[1:n]
  sigma_opt <- opt$par[n + 1]
  return(list(theta = theta_opt, sigma = sigma_opt))
}

#d 计算Ridge模型的自由度
freedom_degree <- function(lambda, x) {
  #n <- nrow(x)

  xT <- t(x) %*% x
  p <- ncol(x)# 特征数量
  I <- diag(p)# 单位矩阵
  xtx <- t(x) %*% x# XᵀX
  ridge_matrix <- xtx + lambda * I  # Ridge matrix: XᵀX + λI
  hat_matrix <- x %*% solve(ridge_matrix) %*% t(x)  # Hat matrix
  df <- sum(diag(hat_matrix))  # Trace of the hat matrix # Hat矩阵的迹即为自由度
  return(as.numeric(df))
}

# Generate example data
set.seed(123)
x <- matrix(rnorm(100), nrow = 20, ncol = 5)  # 20 samples, 5 predictors
y <- rnorm(20)  # Response vector
lambda <- 1  # Ridge penalty parameter

# a. Compute log-likelihood
theta <- runif(5)
sigma <- 1.5
ll <- logLikelihood(theta, sigma, x, y)
print(ll)

# b. Compute Ridge penalized log-likelihood
ridge_ll <- ridge(theta, sigma, lambda, x, y)
print(ridge_ll)

# c. Optimize Ridge regression
opt_result <- ridgeopt(lambda, x, y)
print(opt_result)

# d. Compute degrees of freedom
df <- freedom_degree(lambda, x)
print(df)




#task4
train_data2 <- as.matrix(train_data_scaled[7:length(train_data_scaled)])
test_data2 <- as.matrix(test_data_scaled[7:length(test_data_scaled)])
train_value<- train_data_scaled$motor_UPDRS
test_value<- test_data_scaled$motor_UPDRS

lambda_values <- c(1, 100, 1000)

train_mse2 <- c()
test_mse2 <- c()
df <- c()
theta_value <- list()
# 遍历不同的λ值，训练模型并计算指标

for (i in 1:length(lambda_values)) {
  lambda <- lambda_values[i]

  ridgemodel <- ridgeopt(lambda, train_data2, train_value) # 模型
  theta_value[[i]]  <- ridgemodel$theta# 存储θ

  # 计算训练集的预测值和MSE
  train_predictions <- train_data2 %*%  theta_value[[i]]
  train_mse2[i] <- mean((train_value - train_predictions) ^ 2)

  # 计算测试集的预测值和MSE
  test_predictions <- test_data2 %*%  theta_value[[i]]
  test_mse2[i] <- mean((test_value - test_predictions) ^ 2)

  df[i] <- freedom_degree(lambda, train_data2)

  result <- list(
    train_mse2 = train_mse2,
    test_mse2 = test_mse2,
    df = df,
    theta_value = theta_value
  )

}
print(result)
