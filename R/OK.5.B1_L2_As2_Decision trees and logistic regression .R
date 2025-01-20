#Assignment 2

data <- read.csv(
  '../Data/bank-full.csv',
  sep = ";",
  header = TRUE,
  stringsAsFactors = TRUE
)
data$duration <- c()  #remove column 'duration'

#1
#divide into train/test /validation data
n = dim(data)[1]
set.seed(12345)
id = sample(1:n, floor(n * 0.4))
train = data[id, ]

id1 = setdiff(1:n, id)
set.seed(12345)
id2 = sample(id1, floor(n * 0.3))
validation = data[id2, ]

id3 = setdiff(id1, id2)
test = data[id3, ]

#2
#install.packages("tree")
library(tree)
n_train <- dim(train)
n_validation <- dim(validation)
#a) decision tree with default settings
fit_default <- tree(as.factor(y) ~ ., train)
#b) decision tree with smallest allowed node size equal to 7000
fit_node <- tree(as.factor(y) ~ ., train, control = tree.control(nrow(train), minsize = 7000))
#c) decision tree with minimum deviance as 0.0005
fit_dev <- tree(as.factor(y) ~ ., train, control = tree.control(nrow(train), mindev = 0.0005))

##MCR for train data
#use the trees to predict the training data
pred_default_tr <- predict(fit_default, train, type = 'class')
pred_node_tr <- predict(fit_node, train, type = 'class')
pred_dev_tr <- predict(fit_dev, train, type = 'class')

#calculate the misclassification rate
mcr_default_tr <- 1 - sum(diag(table(pred_default_tr, train$y))) / n_train[1]
mcr_node_tr <- 1 - sum(diag(table(pred_node_tr, train$y))) / n_train[1]
mcr_dev_tr <- 1 - sum(diag(table(pred_dev_tr, train$y))) / n_train[1]

##MCR for validation data
#use the trees to predict the training data
pred_default_va <- predict(fit_default, validation, type = 'class')
pred_node_va <- predict(fit_node, validation, type = 'class')
pred_dev_va <- predict(fit_dev, validation, type = 'class')

#calculate the misclassification rate
mcr_default_va <- 1 - sum(diag(table(pred_default_va, validation$y))) / n_validation[1]
mcr_node_va <- 1 - sum(diag(table(pred_node_va, validation$y))) / n_validation[1]
mcr_dev_va <- 1 - sum(diag(table(pred_dev_va, validation$y))) / n_validation[1]


results <- data.frame(
  Model = c("Default", "NodeSize = 7000", "MinDeviance = 0.0005"),
  TrainingMisclass = c(mcr_default_tr, mcr_node_tr, mcr_dev_tr),
  ValidationMisclass = c(mcr_default_va, mcr_node_va, mcr_dev_va)
)
print(results)
#visualize the trees
# Default Tree
plot(fit_default)
text(fit_default, pretty = 0)

# Tree with minimum node size
plot(fit_node)
text(fit_node, pretty = 0)

# Tree with minimum deviance
plot(fit_dev)
text(fit_dev, pretty = 0)


#3 using training and validataion sets to choose the optimal tree depth in model:
# study the trees up to 50 leaves

trainScore = rep(0, 50)
testScore = rep(0, 50)
for (i in 2:50) {
  prunedTree = prune.tree(fit_dev, best = i)
  pred = predict(prunedTree, newdata = validation, type = "tree")

  #divided by the number of observations in the training set
  trainScore[i]=1/nrow(train)*deviance(prunedTree)
  #divided by the number of observations in the validation set
  testScore[i]=1/nrow(validation)*deviance(pred)
  #trainScore[i] = deviance(prunedTree)
  #testScore[i] = deviance(pred)
}
optimal_num <- which.min(testScore[2:50])

# plot the trees with 2-50 leaves
plot(
  2:50,
  trainScore[2:50],
  type = "b",
  col = "red",
  #ylim = c(0, 1)
)
points(2:50, testScore[2:50], type = "b", col = "blue")

# plot the trees with 2-21 leaves
plot(
  2:21,
  trainScore[2:21],
  type = "b",
  col = "red",
  #ylim = c(0, 1)
)
points(2:21, testScore[2:21], type = "b", col = "blue")

#4
#estimate the confusion matrix, accuracy and F1 score for the test data
#by using the optimal model from step 3
finalTree = prune.tree(fit_dev, best = optimal_num)
Yfit = predict(finalTree, newdata = test, type = "class")
#confusion matrix
conf_mat <- table(True=test$y,Predicted=Yfit)
#accuracy
accuracy = sum(diag(conf_mat))/ dim(test)[1]
#F1 score

true_positives <- conf_mat["yes", "yes"]
false_positives <- conf_mat["no", "yes"]
false_negatives <- conf_mat["yes", "no"]

precision <- true_positives / (true_positives + false_positives)
recall <- true_positives / (true_positives + false_negatives)
f1_score <- 2 * (precision * recall) / (precision + recall)



#5
# Define loss matrix for no/yes classes
loss_matrix <- matrix(c(0, 1, 5, 0), nrow = 2, byrow = TRUE)
rownames(loss_matrix) <- c("True_0", "True_1")     # Row names for true labels
colnames(loss_matrix) <- c("Pred_0", "Pred_1")     # Column names for predicted labels
#Returns the predicted probabilities for each class (no and yes) for the test dataset.
#For example, if there are 2 classes, each row in Yfit_mat contains probabilities like [P(no), P(yes)].
Yfit = predict(finalTree, newdata = test, type = "vector")
#Multiplies the predicted probabilities (Yfit_mat) by the loss matrix (loss_matrix)
#to compute the expected loss for each class
expected_loss <- Yfit %*% loss_matrix
new_pred_class_index <- apply(expected_loss, 1, which.min)
new_prediction = c('no','yes')[new_pred_class_index]


#confusion matrix
conf_mat_mat <- table(test$y,new_prediction)
#accuracy
accuracy_mat = sum(diag(conf_mat_mat)) / dim(test)[1]
#F1 score

true_positives_mat <- conf_mat_mat["yes", "yes"]
false_positives_mat <- conf_mat_mat["no", "yes"]
false_negatives_mat <- conf_mat_mat["yes", "no"]

precision_mat <- true_positives_mat / (true_positives_mat + false_positives_mat)
recall_mat <- true_positives_mat / (true_positives_mat + false_negatives_mat)
f1_score_mat <- 2 * (precision_mat * recall_mat) / (precision_mat + recall_mat)
#The loss matrix sets a penalty on false negative (FN) to encourage the model to
#focus on reducing the errors of the decision uncertain predictions. As a result, the model
#predicts fewer false negative and an increase in true positive(TP). This leads to an increase
#in the recall of the model and the F1 score. However, consequently, the precision decreases
#as there are now also more false positive (FP) predicted. The accuracy drops ever so slightly
# as well as due to the rise in false positive
#to predict a positive to a negative class is now risky

#6 use the optional tree and a logistic regression model to classify the test data
# by using the following principle
tree_tpr <- c()
tree_fpr <- c()
log_tpr <- c()
log_fpr <- c()

#gives the predicted probabilities for each class (no and yes) for the test dataset.
test_pred <- predict(finalTree, test, type= "vector")
logistic_model <- glm(y ~ ., data = train, family = binomial)
#gives the predicted probabilities for each class (no and yes) for the test dataset.
logistic_pred <- predict(logistic_model, test, type = 'response')

for (i in seq(1, 19)) {
  j <- 0.05 * i

  # Tree predictions based on threshold
  tree_pred <- ifelse(test_pred[, "yes"] > j, "yes", "no")
  #matrix(0,37,3,0)
  #Pred=ifelse(ynew/(1-ynew)>37/3, "Osmancik", "Cammeo")

  conf_matrix_tree <- table(tree_pred, test$y)
  tree_tpr[i] <- conf_matrix_tree[4]/(conf_matrix_tree[3] + conf_matrix_tree[4])
  tree_fpr[i] <- conf_matrix_tree[2]/(conf_matrix_tree[1] + conf_matrix_tree[2])

  # Logistic predictions based on threshold
  log_pred <- ifelse(logistic_pred > j, "yes", "no")
  conf_matrix_log <- table(log_pred, test$y)
  log_tpr[i] <- conf_matrix_log[4]/(conf_matrix_log[3] + conf_matrix_log[4])
  log_fpr[i] <- conf_matrix_log[2]/(conf_matrix_log[1] + conf_matrix_log[2])

}

# Plot ROC for Decision Tree
#Receiver operating characteristic (ROC) curve

plot(log_fpr, log_tpr, type = "l", col = "blue", xlab = "False Positive Rate",
     ylab = "True Positive Rate", lwd = 2)
lines(tree_fpr, tree_tpr, col = "red", lwd = 2)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"),
       col = c("blue", "red"), lwd = 2)

# However, due to the class seem to have similar performance as imbalance in the
#data, with more 'no' than 'yes', the large number of true negatives,
#in the denominator of the false positive rate(FPR) will affect the FPR value,
#giving us a small FPR,which in-turn will not be a good indicator for the
#number of false positives in the model.This makes the precision-recall curve a
#better alternative than the ROC curve, especially for imbalanced classes as
#in our case since it shows the precision and recall tradeoff.


#The area under the curve (AUC) is likely similar for both models,
#indicating comparable performance.
#If the dataset has imbalanced classes (e.g., many more negatives than positives),
#the ROC curve can be misleading. A high TPR might come at the cost of very high
#FPR, which the ROC curve may not highlight adequately.
#The precision-recall curve, on the other hand, focuses on the positive class
#and highlights performance in terms of precision (how many predicted positives
#are true positives).
#For tasks where identifying the positive class is more critical
#(e.g., disease detection, fraud identification), precision-recall
#curves provide clearer insights into the trade-off between precision and recall.
