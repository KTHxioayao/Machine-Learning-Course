library(glmnet)
library(caret)

data <- read.csv("../Data/tecator.csv")

set.seed(12345)
trainIndex <- createDataPartition(data$Fat, p = 0.5, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]
# task 1: Fit a linear regression model to predict fat content from
#absorbance characteristics
#optional method to split data
#X <- as.matrix(data[, 1:100])
#y <- data$Fat
#set.seed(12345)
#train_index <- createDataPartition(y, p = 0.5, list = FALSE)
#X_train <- X[train_index, ]
#X_test <- X[-train_index, ]

#y_train <- y[train_index]
#y_test <- y[-train_index]
#lm_model <- lm(y_train ~ X_train)
#y_train_pred <- predict(lm_model, newdata = as.data.frame(X_train))
#y_test_pred <- predict(lm_model, newdata = as.data.frame(X_test))

# Fit the linear regression model excluding moisture, sample, and protein
lm_model <- lm(Fat ~ . - Moisture - Sample - Protein, data = train)
summary(lm_model)

# Predict on the training set, adding predict to train data set
train$pred <- predict(lm_model, train)

# Predict on the test set, adding predict to test data set
test$pred <- predict(lm_model, test)

# Calculate training and test errors MSE
train_mse <- mean((train$Fat - train$pred)^2)
test_mse <- mean((test$Fat - test$pred)^2)
cat("Training Error:", train_mse, "\n")
cat("Test Error:", test_mse, "\n")
# > cat("Training Error:", train_error, "\n")
# Training Error: 0.005283986
# > cat("Test Error:", test_error, "\n")
# Test Error: 591.1322
# (1)Training Error: This value indicates how well the model fits the training
# data. A lower training error suggests a better fit to the training data.
# (2)Test Error: This value indicates how well the model generalizes to unseen
# data. A lower test error suggests better generalization.
# This approach provides a basic understanding of how to use linear regression
# to predict fat content from absorbance characteristics. Regularization
# techniques like LASSO or Ridge regression can further improve model performance
# by addressing potential overfitting or multicollinearity issues.


cat("Linear Regression - Training MSE:", train_mse, "\n")
cat("Linear Regression - Testing MSE:", test_mse, "\n")
#task 2 report the cost function

#task 3: Fit a LASSO regression model to predict fat content from absorbance
# LASSO
#here for glmnet, we need to convert the data to matrix, only matrix can be used

X_train <- as.matrix(train[ , !colnames(train) %in% c("Fat", "Moisture", "Sample",
                                 "Protein","pred")])
y_train <- train$Fat

lasso_model <- glmnet(X_train,   y_train, alpha = 1)

#  log()
plot(lasso_model, xvar = "lambda", label = TRUE)
title("LASSO Coefficients vs log()")

#  lambda
# methods to find the best lambda value, but the feature number is not limited to 3
#cv_lasso_model <- cv.glmnet(X_train, y_train, alpha = 1)

#optimal_lambda <- cv_lasso_model$lambda.min
#cat("best lambda value", optimal_lambda, "\n")


coef_matrix <- coef(lasso_model)
lambda_values <- lasso_model$lambda

#The -1 removes the first row, which corresponds to the intercept
#Here, 2 specifies that we want to apply the function to columns of the
#coef_matrix[-1, ], meaning for each value of lambda.
#For each column (x), this function counts the number of non-zero coefficients.

coef_values <- apply(coef_matrix[-1, ], 2, function(x) sum(x != 0))

# Find lambda values corresponding to 3 non-zero coefficients
lambdas <- lambda_values[which(coef_values == 3)]
#The smallest lambda (least regularization).
#The largest lambda (most regularization).
#The lambda closest to the cross-validated lambda.min

print(lambdas)

#task 4

ridge_model <- glmnet(X_train, y_train, alpha = 0) # alpha = 0  Ridge
# Ridge  log(lambda)
plot(ridge_model, xvar = "lambda", label = TRUE)
title("RelationShip between Ridge and log(Lambda)")


# Conclusions:
# (1)LASSO: Coefficients shrink to zero as lambda increases, leading to sparse
# solutions (feature selection). Suitable when we believe only a subset of
# features are truly relevant. It simplifies the model by selecting a subset of
# predictors.
# (2)Ridge: Coefficients shrink but do not go to zero, which means all features
# are retained with reduced magnitudes. Suitable when we suspect all features
# contribute to the response but need regularization to avoid overfitting.

# task 5 Plot the cross-validation score vs. log(lambda)

#Use cross-validation with default number of folds to compute the optimal
#LASSO model. Present a plot showing the dependence of the CV score on log 𝜆𝜆
#and comment how the CV score changes with log 𝜆

# 1) compute the optimal LASSO model and present a plot showing the dependence
#of the CV score on log 𝜆
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)
plot(cv_lasso)

# 2) report the optimal 𝜆 and how many variables were chosen in this model
optimal_lambda <- cv_lasso$lambda.min # Optimal lambda based on minimum CV error
#Report the optimal 𝜆𝜆 andhow many variables were chosen in this model.
coefficients_optimal <- coef(cv_lasso, s = "lambda.min")
num_selected_variables <- sum(coefficients_optimal != 0) - 1 # Subtract 1 to
#exclude intercept
num_selected_variables


# 3) Compare the optimal lambda with log(lambda) = -4 (statistical significance)
# To check if log(lambda) = -4 leads to a statistically significantly worse prediction:

#sort out the index of the lambda value closest to log(lambda) = -4
#lambda_at_minus_4 <- cv_lasso$lambda[which.min(abs(log(cv_lasso$lambda) - (-4)))]
#lambda_at_minus_4
#extract the cross validation error
#cv_error_at_minus_4 <- cv_lasso$cvm[which.min(abs(log(cv_lasso$lambda) - (-4)))]
#cv_error_at_minus_4


X_test <- as.matrix(test[ , !colnames(test) %in% c("Fat", "Moisture",
                                                   "Sample", "Protein","pred")])
y_test <- test$Fat

# Predict on the test set with the optimal lambda and log(lambda) = -4
# can be also done with s = "lambda.min" or s = exp(-4)
# or use the glmnet do define the optimal model an d the model with log(lambda) = -4
test_pred_optimal_lambda <- predict(cv_lasso, X_test, s = "lambda.min")
test_pred_optimal_loglambda4 <- predict(cv_lasso, X_test, s = exp(-4))

# 4)Create a scatter plot
# create a scatter plot of the original test versus predicted test values for
#the model corresponding to optimal lambda and comment whether the model
#predictions are good

plot(y_test, test_pred_optimal_lambda, main = "Test vs Predicted Fat Content (Optimal Lambda)",
     xlab = "Actual Fat Content", ylab = "Predicted Fat Content", col = "blue", pch = 19)

# Add a reference line (y = x) for perfect prediction
# line with 45% grade as the x is actual fat content and y is the predicted
abline(0, 1, col = "red") #(intercept = 0, and slope = 1)

#
#We can see from the above picture, when the λ value is small, the model fits
#the data very closely, resulting in a lower bias but higher variance,
#after a certain point, further decreasing logλ results in only minimal changes
#in the MSE indicating that the model is in a stable state, with little
#change in the prediction error despite further penalization,
#As logλ continues to increase, the penalty on the coefficients increases,
#forcing more of them to zero and the model begins to underfit, leading to an
#increase in MSE. The model becomes too simple to capture the underlying patterns
#in the data effectively. From the code we could get that
#the optimal λ is 0.068，the MSE at log(λ)=−4 is higher.
