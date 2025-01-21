#install.packages("randomForest")
library(randomForest)
set.seed(1234)
x1 <- runif(1000)
x2 <- runif(1000)
testdata <- cbind(x1, x2)
colnames(testdata) <- c("x1", "x2")
y1 <- as.numeric(x1 < x2)
testlabels <- as.factor(y1)

y2 <- as.numeric(x1 < 0.5)
testlabels2 <- as.factor(y2)

y3 <- as.numeric((x1<0.5 & x2<0.5) | (x1>0.5 & x2>0.5))
testlabels3 <- as.factor(y3)

set.seed(123)

train_data_list <- lapply(1:1000, function(i) {
  x3 <- runif(100)
  x4 <- runif(100)
  trdata <- cbind(x3, x4)
  colnames(trdata) <- c("x1", "x2")
  list(trdata = trdata)
})

#Conditon1 Compute the misclassification error in the same test dataset
# of size 1000. Report results for when the random forest has 1, 10 and
#100 trees.

error_rate_1<- list(
  number1 = rep(0,1000),
  number2 = rep(0,1000),
  number3 = rep(0,1000)
)
mean_error_1<- c()
var_error_1<- c()


for (i in 1:1000) {
  trdata <- train_data_list[[i]]$trdata
  y <- as.numeric(trdata[, 1] < trdata[, 2])
  trlabels <- as.factor(y)

  #build the models
  rf_model1_1 <- randomForest(trdata, trlabels, ntree = 1, nodesize = 25, keep.forest = TRUE)
  rf_model1_2 <- randomForest(trdata, trlabels, ntree = 10, nodesize = 25, keep.forest = TRUE)
  rf_model1_3 <- randomForest(trdata, trlabels, ntree = 100, nodesize = 25, keep.forest = TRUE)

  #predictions and error rates
  predictions1<- predict(rf_model1_1,testdata)
  error_rate_1$number1[i] <- mean(predictions1 != testlabels)
  predictions2<- predict(rf_model1_2,testdata)
  error_rate_1$number2[i] <- mean(predictions2 != testlabels)
  predictions3<- predict(rf_model1_3,testdata)
  error_rate_1$number3[i] <- mean(predictions3 != testlabels)

}

#compute the mean and variance of error rates

mean_error_1[1]<- mean(error_rate_1$number1)
mean_error_1[2]<- mean(error_rate_1$number2)
mean_error_1[3]<- mean(error_rate_1$number3)

var_error_1[1] <- var(error_rate_1$number1)
var_error_1[2] <- var(error_rate_1$number2)
var_error_1[3] <- var(error_rate_1$number3)

#Then, we can change the condition,and the calculation is the same as
#before.

#Repeat for the conditions (x1\<0.5) for 1,10,100 trees, the results are
#summarized in mean_error_2 and var_error_2.

error_rate_2<- list(
  number1 = rep(0,1000),
  number2 = rep(0,1000),
  number3 = rep(0,1000)
)
mean_error_2<- c()
var_error_2<- c()
for (i in 1:1000) {
  trdata <- train_data_list[[i]]$trdata
  y <- as.numeric(trdata[, 1] < 0.5)
  trlabels <- as.factor(y)

  #build the models


  rf_model2_1 <- randomForest(trdata, trlabels, ntree = 1, nodesize = 25, keep.forest = TRUE)
  rf_model2_2 <- randomForest(trdata, trlabels, ntree = 10, nodesize = 25, keep.forest = TRUE)
  rf_model2_3 <- randomForest(trdata, trlabels, ntree = 100, nodesize = 25, keep.forest = TRUE)

  #predictions and error rates
  predictions1<- predict(rf_model2_1,testdata)
  error_rate_2$number1[i] <- mean(predictions1 != testlabels2)
  predictions2<- predict(rf_model2_2,testdata)
  error_rate_2$number2[i] <- mean(predictions2 != testlabels2)
  predictions3<- predict(rf_model2_3,testdata)
  error_rate_2$number3[i] <- mean(predictions3 != testlabels2)

}

mean_error_2[1]<- mean(error_rate_2$number1)
mean_error_2[2]<- mean(error_rate_2$number2)
mean_error_2[3]<- mean(error_rate_2$number3)

var_error_2[1] <- var(error_rate_2$number1)
var_error_2[2] <- var(error_rate_2$number2)
var_error_2[3] <- var(error_rate_2$number3)

#Repeat for the conditions ((x1\<0.5 & x2\<0.5) \| (x1\>0.5 & x2\>0.5))
#and node size 12 for 1,10,100 trees, the results are summarized in
#mean_error_3 and var_error_3.

error_rate_3<- list(
  number1 = rep(0,1000),
  number2 = rep(0,1000),
  number3 = rep(0,1000)
)
mean_error_3<- c()
var_error_3<- c()
for (i in 1:1000) {
  trdata <- train_data_list[[i]]$trdata
  y <- as.numeric((trdata[, 1] < 0.5 & trdata[, 2] < 0.5) | (trdata[, 1] > 0.5 & trdata[, 2] > 0.5))
  trlabels <- as.factor(y)

  #build the models
  rf_model3_1 <- randomForest(trdata, trlabels, ntree = 1, nodesize = 12, keep.forest = TRUE)
  rf_model3_2 <- randomForest(trdata, trlabels, ntree = 10, nodesize = 12, keep.forest = TRUE)
  rf_model3_3 <- randomForest(trdata, trlabels, ntree = 100, nodesize = 12, keep.forest = TRUE)

  #predictions and error rates
  predictions1<- predict(rf_model3_1,testdata)
  error_rate_3$number1[i] <- mean(predictions1 != testlabels3)
  predictions2<- predict(rf_model3_2,testdata)
  error_rate_3$number2[i] <- mean(predictions2 != testlabels3)
  predictions3<- predict(rf_model3_3,testdata)
  error_rate_3$number3[i] <- mean(predictions3 != testlabels3)

}

mean_error_3[1]<- mean(error_rate_3$number1)
mean_error_3[2]<- mean(error_rate_3$number2)
mean_error_3[3]<- mean(error_rate_3$number3)

var_error_3[1] <- var(error_rate_3$number1)
var_error_3[2] <- var(error_rate_3$number2)
var_error_3[3] <- var(error_rate_3$number3)

#summary the results

result<- list(
  mean_error_1 = mean_error_1,
  mean_error_2 = mean_error_2,
  mean_error_3 = mean_error_3,

  var_error_1 = var_error_1,
  var_error_2 = var_error_2,
  var_error_3 = var_error_3

)
print(result)


library(ggplot2)

# Collect predictions and true labels for visualization
predictions_df <- data.frame(
  testlabels = testlabels,
  pred_ntree1 = predict(rf_model1_1, testdata),
  pred_ntree10 = predict(rf_model1_2, testdata),
  pred_ntree100 = predict(rf_model1_3, testdata)
)



library(ggplot2)


# Create data frames for true labels and predictions
test_df_1 <- as.data.frame(testdata)
test_df_1$true_label <- testlabels
test_df_1$pred_label <- predict(rf_model1_3, as.data.frame(testdata))

test_df_2 <- as.data.frame(testdata)
test_df_2$true_label <- testlabels2
test_df_2$pred_label <- predict(rf_model2_3, as.data.frame(testdata))

test_df_3 <- as.data.frame(testdata)
test_df_3$true_label <- testlabels3
test_df_3$pred_label <- predict(rf_model3_3, as.data.frame(testdata))

# Define a function to plot true vs predicted labels
plot_decision_boundary <- function(test_df, title_true, title_pred) {
  p_true <- ggplot() +
    geom_point(data = test_df, aes(x = x1, y = x2, color = as.factor(true_label)), size = 1.5) +
    labs(
      title = title_true,
      x = "x1",
      y = "x2",
      color = "True Label"
    ) +
    scale_color_manual(values = c("#FF5733", "#33FF57")) +
    theme_minimal()

  p_pred <- ggplot() +
    geom_point(data = test_df, aes(x = x1, y = x2, color = as.factor(pred_label)), size = 1.5) +
    labs(
      title = title_pred,
      x = "x1",
      y = "x2",
      color = "Predicted Label"
    ) +
    scale_color_manual(values = c("#E0BBE4", "#957DAD")) +
    theme_minimal()

  return(list(p_true = p_true, p_pred = p_pred))
}

# Plot for Condition 1
plots_1 <- plot_decision_boundary(test_df_1, "True Labels (Condition 1)", "Predicted Labels (Condition 1)")

# Plot for Condition 2
plots_2 <- plot_decision_boundary(test_df_2, "True Labels (Condition 2)", "Predicted Labels (Condition 2)")

# Plot for Condition 3
plots_3 <- plot_decision_boundary(test_df_3, "True Labels (Condition 3)", "Predicted Labels (Condition 3)")


#The correct answer is that the decision boundary in the third dataset is piece-wisely
#aligned with the axes, while it is not in the first dataset. This means that the third
#dataset can be theoretically split using just two splits, while the first dataset
#would need an infinite amount. You can see this visually by plotting the model predictions

