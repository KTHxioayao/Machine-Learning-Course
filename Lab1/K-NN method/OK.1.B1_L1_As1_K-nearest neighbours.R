# Load the dataset
df <- read.csv("../Data/optdigits.csv", header = FALSE)

# Split the data into training, validation, and testing sets
n <- dim(df)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.5)) # 50% of the data for training
training <- df[id,]

id1 <- setdiff(1:n, id) # Remaining 50% of the data
set.seed(12345)
id2 <- sample(id1, floor(n * 0.25)) # 25% of the total data for validation
validation <- df[id2,]

id3 <- setdiff(id1, id2) # Remaining 25% of the data for testing
testing <- df[id3,]

# Load kknn library
library(kknn)

# Perform k-NN on the testing set
testing_result <- kknn(as.factor(V65) ~ ., train = training, test = testing,
                       k = 30, kernel = "rectangular")
testing_predicted <- fitted(testing_result)
testing_actual <- testing$V65


# Generate the confusion matrix for the testing set
confusion_testing <- table( Actual = testing_actual,
  Predicted = testing_predicted)

test_error_rate <- 1 - sum(diag(confusion_testing)) / sum(confusion_testing)

# Perform k-NN on the training set
training_result <- kknn(as.factor(V65) ~ ., train = training, test = training,
                        k = 30, kernel = "rectangular")
training_predicted <- fitted(training_result)
training_actual <- training$V65

# Generate the confusion matrix for the training set
confusion_training <- table(Actual = training_actual,
  Predicted = training_predicted )
train_error_rate <- 1 - sum(diag(confusion_training)) / sum(confusion_training)


# Perform k-NN on the validation set
validation_result <- kknn(as.factor(V65) ~ ., train = training, test = validation,
                          k = 30, kernel = "rectangular")
validation_predicted <- fitted(validation_result)
validation_actual <- validation$V65

# Generate the confusion matrix for the validation set
confusion_validation <- table( Actual = validation_actual,
  Predicted = validation_predicted)


# Function to calculate misclassification error
validation_error_rate <- 1 - sum(diag(confusion_validation)) / sum(confusion_validation)


#task3
# Task 3: Find rows corresponding to the desired class ("8")
probs <- training_result$prob  # Probabilities matrix from training result
desired_class <- "8"  # Target class
desired_rows <- which(apply(probs, 1, function(row) which.max(row) == 9))  # Rows predicted as class "8"

# Extract probabilities for the desired class
prob_eight <- probs[desired_rows, 9]  # Probabilities corresponding to class "8"

# Find indices for the top 2 highest and bottom 3 lowest probabilities
two_highest_indices <- order(prob_eight, decreasing = TRUE)[1:2]
three_lowest_indices <- order(prob_eight)[1:3]

# Plot heatmaps for the bottom 3 probabilities
for (i in three_lowest_indices) {
  row_in_training <- desired_rows[i]
  train_row <- training[row_in_training, ]
  heat_matrix <- matrix(as.numeric(train_row[1:64]), 8, 8, byrow = TRUE)
  heatmap(heat_matrix, Rowv = NA, Colv = NA, scale = "none")
}

# Plot heatmaps for the top 2 probabilities
for (i in two_highest_indices) {
  row_in_training <- desired_rows[i]
  train_row <- training[row_in_training, ]
  heat_matrix <- matrix(as.numeric(train_row[1:64]), 8, 8, byrow = TRUE)
  heatmap(heat_matrix, Rowv = NA, Colv = NA, scale = "none")
}

# Task 4: Perform k-NN with varying k and calculate misclassification error
train_miss_rates <- numeric(30)
valid_miss_rates <- numeric(30)

for (k in 1:30) {
  # Train and validation results for each k
  train_result <- kknn(as.factor(V65) ~ ., train = training, test = training, k = k, kernel = "rectangular")
  valid_result <- kknn(as.factor(V65) ~ ., train = training, test = validation, k = k, kernel = "rectangular")

  # Calculate misclassification error
  train_miss_rates[k] <- 1 - sum(diag(table(fitted(train_result), training$V65))) / nrow(training)
  valid_miss_rates[k] <- 1 - sum(diag(table(fitted(valid_result), validation$V65))) / nrow(validation)
}

# Plot misclassification rates
plot(1:30, train_miss_rates, type = "l", col = "red", ylim = range(c(train_miss_rates, valid_miss_rates)),
     ylab = "Misclassification Rates", xlab = "k")
lines(1:30, valid_miss_rates, type = "l", col = "blue")
legend("bottomright", legend = c("Train", "Validation"), col = c("red", "blue"), lty = 1:1, cex = 0.8)

# Find the optimal k
optimal_k <- which.min(valid_miss_rates)

# Test error for the optimal k
test_result <- kknn(as.factor(V65) ~ ., train = training, test = testing, k = optimal_k, kernel = "rectangular")
test_miss_rate <- 1 - sum(diag(table(fitted(test_result), testing$V65))) / nrow(testing)

# Print misclassification rates
print(test_miss_rate)
print(valid_miss_rates[optimal_k])
print(train_miss_rates[optimal_k])

# Task 5: Cross-entropy calculations
true_probs <- model.matrix(~ as.factor(validation$V65) - 1)  # One-hot encoding of true labels

log_spec <- function(x) log(x + 1e-15)  # Prevent log(0)

cross_entropy <- function(true_probs, predicted_probs) {
  return(-sum(true_probs * log_spec(predicted_probs)) / nrow(true_probs))
}

# Calculate cross-entropy for varying k
cross_entropy_list <- numeric(30)
for (k in 1:30) {
  pred_probs <- kknn(as.factor(V65) ~ ., train = training, test = validation, k = k, kernel = "rectangular")$prob
  cross_entropy_list[k] <- cross_entropy(true_probs, pred_probs)
}

# Print and plot cross-entropy values
print(cross_entropy_list)
plot(cross_entropy_list, col = "blue", type = "b", pch = 5, ylab = "Cross-Entropy", xlab = "k")

# Find the optimal k based on cross-entropy
opt_k_cross_entropy <- which.min(cross_entropy_list)
