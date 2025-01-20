#assignment 3

diabetes <- read.csv('pima-indians-diabetes.csv', header = FALSE)
colnames(diabetes) <- c(
  'Pregnancies',
  'Plasma_glucose',
  'blood_pressure',
  'TricepsSkinFoldThickness',
  'SerumInsulin',
  'BMI',
  'DiabetesPedigreeFunction',
  'Age',
  'Diabetes'
)
library(ggplot2)
library(caret)
# task 1
ggplot(diabetes,
       aes(
         x = diabetes$Age,
         y = diabetes$Plasma_glucose,
         color = diabetes$Diabetes
       )) +
  geom_point() + labs(x = "Age", y = "Plasma Glucose Concentration", color = "Diabetes") +
  ggtitle("Scatterplot of Plasma Glucose vs Age by Diabetes Status")


#type = "prob" predict probability
#type = "raw" predict the raw value/ class
#diabetes_pred <- predict(gml_model, type = "prob")
#Since diabetes only has two outcomes, 1 for you have it and 0 for you don’t have
#it, and the correlation between your age and your glucose levels, it should be
#considered easy to classify diabetes by a standard logistic regression model.

#task 2
formula <- Diabetes ~ Age + Plasma_glucose

# Convert Diabetes to a factor for classification
diabetes$Diabetes <- as.factor(diabetes$Diabetes)
#gml_model <- caret::train(formula,
                          #data = diabetes,
                          #method = "glm",
                          #family = "binomial")

#gml_model<- glm(formula, data = diabetes, family = "binomial")
# Fit a logistic regression model using Age and Plasma Glucose as predictors

gml_model<- glm(Diabetes ~ Age + Plasma_glucose, data = diabetes, family = "binomial")
diabetes_pred <- predict(gml_model,diabetes, type = "response")
diabetes_pred_r1 <- as.factor(ifelse(diabetes_pred > 0.5, 1, 0))


diabetes_confusion <- table(diabetes$Diabetes, diabetes_pred_r1)
error_rate <- 1 - (sum(diag(diabetes_confusion)) / sum(diabetes_confusion))

ggplot(
  diabetes,
  aes(
    x = diabetes$Age,
    y = diabetes$Plasma_glucose,
    color = diabetes_pred_r1
  )
) + geom_point() +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes") +
  ggtitle("Scatterplot of Plasma Glucose vs Age by Diabetes Status")


#task 3
coeffs <- gml_model$coefficients
 decision_boundary <- function(x) {
   -(coeffs[1] + coeffs[2] * x) / coeffs[3]
 }

ggplot(diabetes,
       aes(
         x = diabetes$Age,
         y = diabetes$Plasma_glucose,
         color  = diabetes_pred_r1
       )) +
  geom_point() +
  geom_abline(
    slope = -coeffs[2] / coeffs[3],
    intercept = -coeffs[1] / coeffs[3],
    color = "blue",
    linetype = "dashed"
  ) +
  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Diabetes") +
  ggtitle("Scatterplot with Decision Boundary")

#task 4

pred_res_0.2 <- as.factor(ifelse(diabetes_pred > 0.2, 1, 0))
pred_res_0.8 <- as.factor(ifelse(diabetes_pred > 0.8, 1, 0))


ggplot(
  diabetes,
  aes(
    x = diabetes$Age,
    y = diabetes$Plasma_glucose,
    color  = pred_res_0.2
  )
) + geom_point() +

  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes") +
  ggtitle("Scatterplot of Plasma Glucose vs Age by predicted diabetes Status")


ggplot(
  diabetes,
  aes(
    x = diabetes$Age,
    y = diabetes$Plasma_glucose,
    color  = pred_res_0.8
  )
) + geom_point() +

  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes") +
  ggtitle("Scatterplot of Plasma Glucose vs Age by raw diabetes Status")

#task 5

# Create polynomial features (basis functions) for the model
diabetes$z1 <- diabetes$Plasma_glucose ^ 4
diabetes$z2 <- diabetes$Plasma_glucose ^ 3 * diabetes$Age
diabetes$z3 <- diabetes$Plasma_glucose ^ 2 * diabetes$Age ^ 2
diabetes$z4 <- diabetes$Plasma_glucose * diabetes$Age ^ 3
diabetes$z5 <- diabetes$Age ^ 4

#formula <- Diabetes ~ Age + Plasma_glucose + z1 + z2 + z3 + z4 + z5
new_gml_model<- glm(Diabetes ~ Age + Plasma_glucose+ z1 + z2 + z3 + z4 + z5,
                    data = diabetes, family = "binomial")
diabetes_pred <- predict(new_gml_model,diabetes, type = "response")

new_diabetes_pred <- predict(new_gml_model,diabetes, type = "response")
new_diabetes_pred_r1 <- as.factor(ifelse(new_diabetes_pred > 0.5, 1, 0))


new_diabetes_confusion <- table(diabetes$Diabetes, new_diabetes_pred_r1)
error_rate <- 1 - (sum(diag(new_diabetes_confusion)) / sum(new_diabetes_confusion))

cat(" training misclassification error:", error_rate)

ggplot(
  diabetes,
  aes(
    x = diabetes$Age,
    y = diabetes$Plasma_glucose,
    color  = new_diabetes_pred_r1
  )
) + geom_point() +

  labs(x = "Age", y = "Plasma Glucose Concentration", color = "Predicted Diabetes") +
  ggtitle("Scatterplot of Plasma Glucose vs Age by raw diabetes Status")

#We can see that after add the basis function expansion, the misclassification
#error is lower than the previous model, it means that the basis function
#expansion can improve the performance of the model, look at the
#coeﬀicients furtherly, the new added variables slightly affect the prediction,
#it means that the new added variables affect the prediction positively and
#the decision boundary become from a line to a multidimensional graphics.

