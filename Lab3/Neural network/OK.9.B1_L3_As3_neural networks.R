#assignment4
#task1
#install.packages("neuralnet")
library(neuralnet)
set.seed(1234567890)


Var <- runif(500, 0, 10)
mydata <- data.frame(Var, Sin=sin(Var))
tr <- mydata[1:25,] # Training
te <- mydata[26:500,] # Test

# Random initialization of the weights in the interval [-1, 1]
n_input <- 1 # Number of input neurons 输入层神经元数
n_hidden <- 10 # Number of hidden neurons 隐藏层神经元数
n_output <- 1 # Number of output neurons 输出层神经元数

# Use one hidden layer with 10 hidden units.
n_weights_input_to_hidden <- n_input * n_hidden
n_bias_hidden <- n_hidden
n_weights_hidden_to_output <- n_hidden * n_output
n_bias_output <- n_output
total_weights <- n_weights_input_to_hidden +
  n_bias_hidden + n_weights_hidden_to_output + n_bias_output

set.seed(1234567890)
winit <- runif(total_weights, -1, 1)
# Sin ~ Var meaning Sin is the target and Var is the input
nn_logistic <- neuralnet(Sin ~ Var, data=tr, hidden = 10, act.fct="logistic",
                startweights = winit, linear.output = TRUE)

# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr, cex=2, main ='Sine Function with NN logistic', xlab='x', ylab= 'sin(x)')
# cex is the size of the points #

points(te, col = "blue", cex=1) # plot the test data in blue

#predict (model, te) #test data includes(Var, sin(Var)
points(te[,1],predict(nn_logistic,te), col="red", cex=1) # plot the predictions in red

#The model tends to follow the sine function very well as the non-linear sigmoid activation function is used. However, there is some variations in the prediction when there is an interval with no training data. It can be improved if the training data is spread evenly over the complete interval of [0,10]. This can be visualised in the figure below.

#task2

#In question (1), you used the default logistic (a.k.a. sigmoid) activation function, i.e.act.fct = "logistic". Repeat question (1) with the following custom activation functions: h1(x) = x, h2(x) = max{0, x} and h3(x) = ln(1 +exp x) (a.k.a. linear, ReLU and softplus). See the help ﬁle of the neuralnetpackage to learn how to use custom activation functions. Plot and comment your results.
act_linear <- function(x) {x}
act_relu <- function(x) { ifelse(x > 0, x, 0) }

act_softplus <- function(x) {log(1 + exp(x))}

nn_linear <- neuralnet(Sin ~ Var, data=tr, hidden = 10, act.fct=act_linear, startweights = winit, linear.output=TRUE)
nn_relu <- neuralnet(Sin ~ Var, data=tr, hidden=10, act.fct=act_relu, startweights = winit, linear.output=TRUE)

nn_softplus <- neuralnet(Sin ~ Var, data=tr, hidden=10, act.fct=act_softplus, startweights = winit, linear.output=TRUE)


plot(tr, cex=2)
points(te, col = "blue", cex=1)
points(te[,1], predict(nn_linear, te), col="red", cex=1)
points(te[,1], predict(nn_relu, te), col="green", cex=1)
points(te[,1], predict(nn_softplus, te), col="yellow", cex=1)

#As seen from the figures below, it can be said that the linear activation function doesn’t capture the non-linear sine function effectively. The non-linear ReLu tends to capture the data at the start. But the activation of zero for negative values makes the model loose a lot of information and it cannot perform well. The softplus activation function performs the best among these three and captures the trend of sine function well.
#Linear: Fails to capture the sine function's non-linearity.ReLU: Works well for positive inputs but struggles with periodic sine values due to its zero output for negative inputs.Softplus: Performs better than linear and ReLU, capturing the sine function more smoothly, though slightly less accurately than sigmoid.

#Task3
#  Sample 500 points uniformly at random in the interval [0,50], and apply the sine function to each point. Use the NN learned in question (1) to predict the sine function value for these new 500 points. You should get mixed results.
set.seed(1234567890)
Var <- runif(500, 0, 50)
new_mydata <- data.frame(Var=Var, Sin = sin(Var))

# here new_mydata$Var,new_mydata$Sin  replace with new_mydata will be the same
plot(new_mydata$Var,new_mydata$Sin, xlab = "Var", ylab = "Sin", col = "blue", cex = 2,
     ylim = c(-10, 1))
#points(te, col = "blue", cex = 1)
points(new_mydata$Var, predict(nn_logistic, new_mydata), col = "red", cex = 1)

#The neural network was trained on the data in the range of [0, 10] and it performed well in that range. When it was trying to predict the data outside this range, the predictions converged to a certain value. This can be visualised in the graph below. The periodic function like sine cannot be predicted outside the range of the training data. The model fails to generalise the periodic property of sine.
#The model performs well in the interval [0,10] (training range) but fails to generalize for points outside this range. Predictions converge to a constant value for points beyond the training interval due to the nature of the sigmoid activation and the learned weights.

#task4
#In question (3), the predictions seem to converge to some value.
#Explain why this happens. To answer this question, you may need to get access
#to the weights of the NN learned. You can do it by running nn or nn$weights where nn is the NN learned.
input_hidden_weight<- nn_logistic$weights[[1]][[1]]
print(input_hidden_weight) # 2 columns, bias and Input
hidden_output_weight<- nn_logistic$weights[[1]][[2]]
print(hidden_output_weight)   #eleven elmements and the last one is bias
#The sigmoid activation function outputs values in the range (0,1).
#Beyond the training interval [0,10], the model has no training data and relies on extrapolation.
#The weights and biases learned during training lead to the saturation of the sigmoid activation function for inputs outside the training range.
#As a result, the network outputs values near the upper or lower bounds of the sigmoid function, causing predictions to converge.

#Task5
#Sample 500 points uniformly at random in the interval [0,10], and apply the sine function to each point. Use all these points as training points for learning a NN that tries to predict x from sin(x) xx from sin(x) from sin(x), i.e. unlike before when the goal was to predict sin(x) from x. Use the learned NN to predict the training data. You should get bad results. Plot and comment your results. Help: Some people get a convergence error in this ques- tion. It can be solved by stopping the training before reaching convergence by setting threshold = 0.1.
#Sample 500 points uniformly at random in the interval [0,10]
new_points2 <- data.frame(Var = seq(0, 10, length.out = 500))
new_points2$Sin <- sin(new_points2$Var)

nn_inverse <- neuralnet(Var ~ Sin, data=new_points2, hidden=10, act.fct="logistic", linear.output=TRUE,threshold = 0.1)

plot(new_points2$Sin, new_points2$Var, col="blue", pch=20)
#predictions_inverse <- as.data.frame(predict(nn_inverse, data.frame(new_points2$Sin)))

predictions_inverse<-predict(nn_inverse, new_points2)
#predictions_inverse <- predictions_inverse[, 1]
points(new_points2$Sin, predictions_inverse, col="red", pch=20)

#he model struggles because:sin(x) is not injective (𝑥1≠𝑥2 can have sin(𝑥1)=sin(𝑥2)
#For many values of sin(𝑥), there are multiple possible , making it hard for the network to learn a clear mapping.
#The neural network produces inaccurate results, particularly in overlapping regions of the sine function.



