#task1
library(ggplot2)
#install.packages("caret")
library(caret)

data <- read.csv("communities.csv")
response <- data$ViolentCrimesPerPop
#data$ViolentCrimesPerPop<- c()

############# PART 1 ###################
#scale(): Standardizes the data to have a mean of 0 and a standard deviation of 1.
#Standardization is essential for PCA because it ensures that all variables are
#on the same scale, so no single variable dominates the results.
scaled <- scale(data[, -which(colnames(data) == 'ViolentCrimesPerPop')])

#cov(): Computes the covariance matrix of the standardized data.
cov_matrix <- cov(scaled)
#eigen(): Performs an eigen decomposition of the covariance matrix
#Eigenvalues: Indicate the amount of variance explained by each principal component.
#Eigenvectors: Indicate the direction of the principal components.
pca_eigen <- eigen(cov_matrix)

#eigenvalues: The variances explained by each principal component.
eigenvalues <- pca_eigen$values
#prop_variance:
#The proportion of the total variance explained by each principal component. Calculated as:
prop_variance <- eigenvalues / sum(eigenvalues)

cum_variance <- cumsum(prop_variance)
num_components <- which(cum_variance >= 0.95)[1]
cat("Number of components explaining at least 95% variance:",
    num_components,
    "\n")
prop_variance_12 <- prop_variance[1:2]
cat("Proportion of variance explained by first two components:",
    prop_variance_12,
    "\n")

# task2
## Observations
# 35 components are needed to obtain atleast 95% of variation in the data. The proportion
# of variance explained by the first two components are 0.25016 and 0.16935 respectively.

############## PART 2 ###################

pca_princomp <- princomp(scaled, cor = TRUE)
#pca_princomp <- princomp(scaled)  will give the same results in this case

summary(pca_princomp)
# Tracing plot for first principle component

# Extract the loadings (principal component coefficients)
loadings <- pca_princomp$loadings[, 1]
# Sort features by their absolute contribution to PC1
top_features <- sort(abs(loadings), decreasing = TRUE)[1:5]

cat("Top 5 features contributing to PC1 (absolute value):\n")
print(top_features)


#loadings_df <- data.frame(Feature = colnames(scaled),
#Loading = abs(pca_princomp$loadings[, 1]))
#top_features <- head(arrange(loadings_df, desc(Loading)), 5)
#cat("Top 5 contributing features to PC1:\n")
#print(top_features)

#ggplot(loadings_df, aes(x = reorder(Feature, Loading), y = Loading)) +
#geom_bar(stat = "identity") +
#coord_flip() +
#ggtitle("Contributions to PC1") +
#xlab("Feature") +
#ylab("Absolute Loading")

trace_plot <- barplot(
  abs(loadings),
  main = "Trace Plot: Contributions to First Principal Component",
  xlab = "Features",
  ylab = "Absolute Contribution",
  names.arg = colnames(scaled),
  las = 2,
  col = "blue"
)

# Plot of PC Scores
pc_scores <- as.data.frame(pca_princomp$scores)
pc_scores$Response <- response
#pc_scores <- cbind(pc_scores,response)

#Here, PC1 is Comp.1 and PC2 is Comp.2
ggplot(pc_scores, aes(x = Comp.1, y = Comp.2, color = Response)) +
  geom_point() +
  scale_color_gradient(low = "blue", high = "red") +
  ggtitle("Scatter Plot of PC Scores (PC1 vs PC2)") +
  xlab("PC1") +
  ylab("PC2") +
  theme_minimal()

## Observations
# While performing the PCA using princomp() function, not many features have much contribution
# to first principle component and the top 5 features that contribute most are-
# Here, distinct clusters does not exist which shows PC1 and PC2 does not capture pattern
# effectively in the data.There is no smooth gradient implies that PCs does not effectively
# captures the spectrum of variation in crime level.Also, the overlapping low and high crime
# areas indicate that more components are needed to capture crime related variability.


###
#There are many features have contribution to PC1. Ranked by contribution values,
#the top 5 contributing features in PC1 are: "medFamInc""medIncome""PctKids2Par"
#"pctWInvInc" and "PctPopUnderPov". These characteristics have a common theme,
#namely economic stress. Most of the features are related to household income
#plus the percentage of children living with both parents. These factors are
#logically related to crime levels, as elevated financial pressures force
#individuals to engage in illegal activities. Areas with limited economic
#opportunities and higher population densities tend to experience higher
#rates of criminal activity.

###
#PC1, the color change is more obvious, and the larger the PC1 (moving to the right),
#the higher the corresponding crime rate (red) seems to be, indicating that the key features of the
#data points are mainly reflected in PC1.

######### PART 3 ############

# Scaling feature and response variables of dataset
features <- data[, !(colnames(data) %in% "ViolentCrimesPerPop")]
response1 <- data$ViolentCrimesPerPop
#scaling
features_scaled <- scale(features)
response_scaled <- scale(response)

#combine both features and response
#should be converted to data frame first and add the response variable
data_scaled <- as.data.frame(features_scaled)
data_scaled$ViolentCrimesPerPop <- as.vector(response_scaled)

n = dim(data)[1]
set.seed(12345)
#train_indices <- sample(1:nrow(features_scaled), size = nrow(features_scaled) / 2)
train_indices <- sample(1:n, floor(n * 0.5))
train_data <- data_scaled[train_indices, ]
test_data <- data_scaled[-train_indices, ]

lm_model <- lm(ViolentCrimesPerPop ~ ., data = train_data)
train_predictions <- predict(lm_model, newdata = train_data)
test_predictions <- predict(lm_model, newdata = test_data)
train_mse <- mean((train_data$ViolentCrimesPerPop - train_predictions) ^
                    2)
cat("Training MSE:", train_mse, "\n")
test_mse <- mean((test_data$ViolentCrimesPerPop - test_predictions) ^ 2)
cat("Test MSE:", test_mse, "\n")

## Observations
# The training MSE is 0.25917 and testing MSE is 0.40005.
# A relatively low training error suggest that model captures the relationships in the training
# data reasonably well. The gap between training and test MSE indicates slight overfitting.
# This model does not generalize well to test data.

######### PART 4 ###########

# Extracting X(features) and Y(target) for training and test dataset
X_train <- as.matrix(train_data[, !(colnames(train_data) %in% "ViolentCrimesPerPop")])
Y_train <- train_data$ViolentCrimesPerPop
X_test <- as.matrix(test_data[, !(colnames(test_data) %in% "ViolentCrimesPerPop")])
Y_test <- test_data$ViolentCrimesPerPop

#Defining the cost function for linear regression without intercept
cost_function <- function(theta, X, Y) {
  predictions <- X %*% theta # Linear model predictions
  MSE <- mean((Y - predictions) ^ 2) # Residuals
  return(MSE) # Mean Squared Error
}

theta <- rep(0, ncol(X_train)) # Initialize theta
# To store errors during optimization
#errors <- data.frame(
# Iteration = integer(),
#TrainError = numeric(),
#TestError = numeric()
#)
train_errors <- c()
test_errors <- c()
# Defining a wrapper to track errors at each iteration
cost_with_tracking <- function(theta) {
  train_error <- cost_function(theta, X_train, Y_train)
  test_error <- cost_function(theta, X_test, Y_test)
  # errors <<- rbind(
  #  errors,
  #  data.frame(
  #    Iteration = nrow(errors) + 1,
  #    TrainError = train_error,
  #    TestError = test_error
  #  )
  # )
  # will store the training error at each iteration
  train_errors <<- c(train_errors, train_error)
  # will store the training error at each iteration
  test_errors <<- c(test_errors, test_error)

  return(train_error) # Return the training error for optimization
}

optim_result <- optim(par = theta, fn = cost_with_tracking, method = "BFGS")

# Plotting the error vs iterations
df<-data.frame(Iteration=1:length(train_errors),train_errors=train_errors,test_errors=test_errors)
sub_df<-df[-(1:500),] #removing first 500 iterations
ggplot(sub_df, aes(x = Iteration)) +
  geom_line(aes(y = train_errors, color = "Train Error")) +
  geom_line(aes(y = test_errors, color = "Test Error")) +
  labs(title = "Error vs Iterations", x = "Iteration", y = "Test Error") +
  scale_color_manual(values = c("Train Error" = "blue", "Test Error" = "red")) +
  theme_minimal()

optimal_iteration <- which.min(test_errors)
cat("Optimal Iteration:", optimal_iteration, "\n")
cat("Training Error at Optimal Iteration:",
    min(train_errors),
    "\n")
cat("Test Error at Optimal Iteration:", min(test_errors), "\n")

## Observation
# The test error is minimum at iteration 2183, which makes it optimal.
# Here, the errors are significantly lower compared to Part 3, indicating overall
# improvement in model performance. Also, the iterative optimization approach used in
# Part 4 with early stopping captures the underlying data pattern effectively.
