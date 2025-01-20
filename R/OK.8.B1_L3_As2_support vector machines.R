# Lab 3 block 1 of 732A99/TDDE01/732A68 Machine Learning
# Author: jose.m.pena@liu.se
# Made for teaching purposes
#install.packages("kernlab")
library(kernlab)
set.seed(1234567890)

data(spam)
foo <- sample(nrow(spam))
spam <- spam[foo,]
tr <- spam[1:3000, ] # Training
va <- spam[3001:3800, ] # Validation
trva <- spam[1:3800, ] # Training + Validation
te <- spam[3801:4601, ] # Test

by <- 0.3
err_va <- NULL
for(i in seq(by,5,by)){
  filter <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),
                 C=i,scaled=FALSE) # model
  mailtype <- predict(filter,va[,-58])
  t <- table(mailtype,va[,58])
  err_va <-c(err_va,(t[1,2]+t[2,1])/sum(t))  # error rate for each iteration
}

filter0 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter0,va[,-58])
t <- table(mailtype,va[,58])
err0 <- (t[1,2]+t[2,1])/sum(t)
err0

filter1 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter1,te[,-58])
t <- table(mailtype,te[,58])
err1 <- (t[1,2]+t[2,1])/sum(t)
err1

filter2 <- ksvm(type~.,data=trva,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter2,te[,-58])
t <- table(mailtype,te[,58])
err2 <- (t[1,2]+t[2,1])/sum(t)
err2

filter3 <- ksvm(type~.,data=spam,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter3,te[,-58])
t <- table(mailtype,te[,58])
err3 <- (t[1,2]+t[2,1])/sum(t)
err3

# Implementation of SVM predictions.
# Questions

# 1. Which filter do we return to the user ? filter0, filter1, filter2 or filter3? Why?
#Among filter0, filter1, and filter2, the best choice is filter2, which is trained on both the training set (tr) and validation set (va) combined (trva).
#It leverages more data for training (3000 rows from tr + 800 rows from va = 3800 rows), leading to better generalization compared to filter0 (trained only on tr). While filter1 is tested on te, it was trained only on tr, making it less robust than filter2, which uses more data for training.

# 2. What is the estimate of the generalization error of the filter returned to the user? err0, err1, err2 or err3? Why?
#err2
# 3. Implementation of SVM predictions.
#Once a SVM has been ﬁtted to the training data, a new point is essentially classiﬁed according to the sign of a linear combination of the kernel function values between the support vectors and the new point. You are asked to implement this linear combination for filter3. You should make use of the functions alphaindex, coef and b that return the indexes of the support vectors, the linear coefﬁcients for the support vectors, and the negative intercept of the linear combination. See the help ﬁle of the kernlab package for more information. You can check if your results are correct by comparing them with the output of the function predict where you set type = "decision". Do so for the ﬁrst 10 points in the spamdataset. Feel free to use the template provided in the Lab3Block1 2021 SVMs St.Rﬁle.

sv<-alphaindex(filter3)[[1]] #number of support vectors
co<-coef(filter3)[[1]] #coefficients for the support vectors
inte<- - b(filter3) #intercept
k <- NULL
for (i in 1:10) {# We produce predictions for just the first 10 points in the dataset.
  k2 <- NULL
  for (j in 1:length(sv)) {
    #k(xi,xj) = exp(-0.05*sum((xi-xj)^2))  sigma=0.05 Basic Gaussian kernel
    # Linear combination:
    k2 <- c(k2, exp(-0.05 * sum((as.numeric(spam[i, -58]) - as.numeric(spam[sv[j], -58]))^2)))
    # Your code here
  }
  k <- c(k, sum(co * k2) + inte) # Your code here)
}
k
predict(filter3,spam[1:10,-58], type = "decision")
