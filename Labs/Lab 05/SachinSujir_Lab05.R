# Sachin Mohan Sujir

## Part I  

# 1. Create a training set containing a random sample of 800 observations,
# and a test set containing the remaining observations. 

library(tree)
library(ISLR)
set.seed(2)
trainingSet=sample(1:nrow(OJ), 800)
oj.trainingSet = OJ[trainingSet,]
oj.testSet = OJ[-trainingSet,]

# 2. Fit a tree to the training data, with Purchase as the response and 
# the other variables except for Buy as predictors. Use the summary() 
# function to produce summary statistics about the tree, and describe the results obtained. 
# What is the training error rate? How many terminal nodes does the tree have? 
str(OJ)
tree = tree(Purchase ~., oj.trainingSet)
summary(tree)


# As we can see in the summary the training error rate is 0.1588
# and there are 9 terminal nodes.

# 3. Type in the name of the tree object in order to get a detailed text
# output. Pick one of the terminal nodes, and interpret the information displayed. 

tree
# Let us consider node No 2 .There are total of 359 branch with the deviance of 422.80. The split 
# criteria is LoyalCH < 0.5036 More than 72% are part of MM while only upto 2% are the value of CH.

# 4. Create a plot of the tree, and interpret the results. 
plot(tree)
text(tree, pretty = 0)


# Here it can be seen that LoyalCH is the the important indicator.First 3 top nodes have LoyalCH. Rest of the nodes are split into different category.

# 5. Predict the response on the test data, and produce a confusion matrix comparing the test labels
# to the predicted test labels. What is the test error rate? 
tree.prediction = predict(tree, oj.testSet, type = "class")
table(tree.prediction, oj.testSet$Purchase)
mean(tree.prediction != oj.testSet$Purchase)

# Test Error rate is around 19.26%.

# 6. Apply the cv.tree()  function to the training set in order to determine the optimal tree size
cv = cv.tree(tree, FUN = prune.misclass)
cv

# 7. Produce a plot with tree size on the x -axis and cross-validated classification error rate on the y -axis. 
plot(cv$size, cv$dev, type = "b", xlab = "Tree size", ylab = "Error")

# 8. Which tree size corresponds to the lowest cross-validated classification error rate? 
# Tree zize 7 has lowest cross-validated classificcation error rate.

# 9. Produce a pruned tree corresponding to the optimal tree size obtained using crossvalidation. 
# If cross-validation does not lead to selection of a pruned tree, then create a pruned tree with 
# five terminal nodes

prune = prune.misclass(tree, best =2)
plot(prune)
text(prune, pretty = 0)


# 10. Compare the training error rates between the pruned and un-pruned trees. Which is higher? 
summary(tree)
summary(prune)


# Error rate for tree : 0.1588 VS Error rate for pruned tree: 0.1862. The error rate is higher in pruned tree.

# 11. Compare the test error rates between the pruned and un-pruned trees. Which is higher
prune.prediction = predict(prune, oj.testSet, type = "class")
table(prune.prediction, oj.testSet$Purchase)
mean(prune.prediction !=oj.testSet$Purchase)

 
# The error rate was around 19% before pruning.
# After pruning the error rate has increased to around 24%(0.2407407). 
# Despite the error rate, the pruned tree is easy to interpret as it is not
# complicated like the normal tree.

# Test error rate for pruned tree is greater than normal tree.


# Part II  

# 1. Split the data set into a training set and a test set. 

set.seed(3)
trainSet = sample(1:nrow(Carseats), nrow(Carseats) / 2)
Carseats.trainSet = Carseats[trainSet, ]
Carseats.testSet = Carseats[-trainSet, ]

# 2. Fit a regression tree to the training set. Plot the tree, and interpret the results. 
# What test MSE do you obtain? 
tree.carseats = tree(Sales ~ ., data = Carseats.trainSet)
summary(tree.carseats)
plot(tree.carseats)
text(tree.carseats, pretty = 0)
pred.car = predict(tree.carseats, newdata = Carseats.testSet)
mean((pred.car - Carseats.testSet$Sales)^2)

# There are 16 terminal nodes in this regression tree

# The mean deviance is 2.134 which means the result can be deviated by 2.13.
# Test MSE : 4.78

# 3. Use cross-validation in order to determine the optimal level of tree complexity.
# Does pruning the tree improve the test MSE? 

crossValidate.carseats = cv.tree(tree.carseats)
plot(crossValidate.carseats$size, crossValidate.carseats$dev, type = "b", xlab = "Tree size", ylab = "Error")
tree.min = which.min(crossValidate.carseats$dev)
points(tree.min, crossValidate.carseats$dev[tree.min], col = "Blue", cex = 2, pch = 20)
prune.carseats = prune.tree(tree.carseats, best = 5)
plot(prune.carseats)
text(prune.carseats, pretty = 0)
pred.car = predict(prune.carseats, newdata = Carseats.testSet)
mean((pred.car - Carseats.testSet$Sales)^2)


# Tree with size 5 is pruned. After pruning MSE is 5.39.

# No, pruning does not improve test MSE. It increases.

# 4. Use the bagging approach in order to analyze this data. What test MSE do you obtain? 
# Use the importance() function to determine which variables are most important. 
require(randomForest)
bag.carseats = randomForest(Sales ~ ., data = Carseats.trainSet, mtry = 10, ntree = 500, importance = TRUE)
pred.car.bag = predict(bag.carseats, newdata = Carseats.testSet)
mean((pred.car.bag - Carseats.testSet$Sales)^2)
importance(bag.carseats)

# With bagging Test MSE is 2.76 which is lesser.
# By using Importance we find Price and ShelveLoc are the two most important variables
# since their node purity is highest.

# 5. Use random forests to analyze this data. What test MSE do you obtain? 
# Use the importance() function to determine which variables are most important.
# Describe the effect of m, the number of variables considered at each split, on the error rate obtained. 

randomFor.carseats = randomForest(Sales ~ ., data = Carseats.trainSet, mtry = 3, ntree = 500, importance = TRUE)
pred.car.rf = predict(randomFor.carseats, newdata = Carseats.testSet)
mean((pred.car.rf  - Carseats.testSet$Sales)^2)
importance(randomFor.carseats)

# The test MSE increases to 3.37.This is the result since m = p^1/2 i.e (root of p).
# Price and ShelveLoc are the two most important variables