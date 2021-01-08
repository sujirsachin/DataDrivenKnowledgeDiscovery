#Sachin Mohan Sujir

#Lab 04

library(glmnet)
college=read.csv("College.csv", header=T, na.strings="?")
fix(college)
rownames(college)=college[,1]
fix(college)
college = college[,-1]
fix(college)

dim(college)

# 1. Split the data set into a training set and a test set.
# 1. Split the data set into a training set and a test set.
set.seed(1)
college[, -1] = apply(college[, -1], 2, scale)
train.size =  dim(college)[1] / 2
train =sample(1:dim(college)[1], train.size)
test = -train
college.train =  college[train, ]
college.test =  college[test, ]

# 2. Fit a linear model using least squares on the training set, and report the test error obtained. 
lmFit= lm(Apps ~ ., data = college.train)
lmPrediction = predict(lmFit, college.test)
mean((lmPrediction - college.test$Apps)^2)


# Test MSE for linear model is 0.07582611


#Fit a ridge regression model on the training set, with ?? chosen by cross-validation.
#Report the test error obtained.

train.matrix = model.matrix(Apps ~ ., data = college.train)
test.matrix =model.matrix(Apps ~ ., data = college.test)
grid =10 ^ seq(4, -2, length = 100)
ridge = glmnet(train.matrix, college.train$Apps, alpha = 0, lambda = grid, thresh = 1e-12)
ridgeCV = cv.glmnet(train.matrix, college.train$Apps, alpha = 0, lambda = grid, thresh = 1e-12)
bestlam.ridge = ridgeCV$lambda.min
bestlam.ridge
ridgePrediction = predict(ridge, s = bestlam.ridge, newx = test.matrix)
mean((ridgePrediction - college.test$Apps)^2)


# Test MSE for ridge regression model is 0.06994741

# 4. Fit a lasso model on the training set, with ?? chosen by cross-validation. 
# Report the test error obtained, along with the number of non-zero coefficient estimates.

lasso =glmnet(train.matrix, college.train$Apps, alpha = 1, lambda = grid, thresh = 1e-12)
lassoCV = cv.glmnet(train.matrix, college.train$Apps, alpha = 1, lambda = grid, thresh = 1e-12)
bestlam.lasso = lassoCV$lambda.min
bestlam.lasso
lassoPrediction <- predict(lasso, s = bestlam.lasso, newx = test.matrix)

mean((lassoPrediction - college.test$Apps)^2)

# Test MSE for lasso model is 0.06942227

# 5. Comment on the results obtained. How accurately can we predict the number of college applications 
# received? 
# Is there much difference among the test errors resulting from these five approaches?
test.avg = mean(college.test$Apps)
lmAcc = 1 - mean((lmPrediction - college.test$Apps)^2) / mean((test.avg - college.test$Apps)^2)
lmAcc
ridgeAcc = 1 - mean((ridgePrediction - college.test$Apps)^2) / mean((test.avg - college.test$Apps)^2)
ridgeAcc
lassoAcc = 1 - mean((lassoPrediction - college.test$Apps)^2) / mean((test.avg - college.test$Apps)^2)
lassoAcc

#All models show high accuracy since test R^2 of all models is above or near to 0.9

#In ridge regression, all p predictors are considered in the final model. 
#The penalty will reduce all the coefficients towards the zero but not exactly equals to zero. 
#In lasso regression, the coefficients for the predictors Outstate,PhD,S.F.Ratio are set to zero.
mean((lassoPrediction - college.test$Apps)^2)
mean((ridgePrediction - college.test$Apps)^2)
mean((lmPrediction - college.test$Apps)^2)

# The errors comapritevly are lesser in lasso and more in linear. 
#Ridge gives error rate in between.
#There isn't much differnce in error in the three.

## Part2  
#We will now try to predict per capita crime rate in the Boston data set, which is part of ISLR package.  
library(MASS)
set.seed (3)
data("Boston")
attach(Boston)

# 1. Try out some of the regression methods explored in this week, such as best subset selection, 
# the lasso, and ridge regression. Present and discuss results for the approaches that you consider. 

train = sample(506,0.65*506)
traindata = Boston[train,]
testdata = Boston[-train,]
trainresponse = crim[train]
testresponse = crim[-train]

training=model.matrix (crim~.,traindata)
testing=model.matrix (crim~.,testdata)
cv.out=cv.glmnet (training,trainresponse,alpha =0)

cvlamda.ridge = cv.out$lambda.min
cvlamda.ridge

ridge.fit =glmnet(testing,testresponse,alpha=0,lambda=cvlamda.ridge)


ridge.pred = predict(ridge.fit,s= cvlamda.ridge,newx = testing)
MeanSqError=mean((testresponse-ridge.pred)^2)
MeanSqError


error.ridge = sqrt(MeanSqError)
error.ridge


#Lasso

cvlamda.lasso = cv.out$lambda.min
cvlamda.lasso
lasso =glmnet(testing,testresponse,alpha=1,lambda=cvlamda.lasso)


lassoPredict = predict(lasso,s= cvlamda.lasso,newx = testing)

MeanSqError1=mean((testresponse-lassoPredict)^2)
error.lasso = sqrt(MeanSqError1)
error.lasso


#Best Subset

test.fit = lm(crim~.,data= Boston)
summary(test.fit)

#rad is highly significant

bestfit= lm(crim~rad,traindata) 
summary(bestfit)
MSE_bestfit = mean((testresponse-predict(bestfit,testdata))^2)
MSE_bestfit

error.lm = sqrt(MSE_bestfit)
error.lm

#choosing two best 

bs.fit = lm(crim~rad+medv,traindata) 
summary(bs.fit)

MSE.bs = mean((testresponse-predict(bs.fit,testdata))^2)
MSE.bs


error.lm = sqrt(MSE.bs)
error.lm


bs.fit = lm(crim~dis+rad+medv+black,traindata) 
summary(bs.fit)

MSE.bs = mean((testresponse-predict(bs.fit,testdata))^2)
MSE.bs


error.lm = sqrt(MSE.bs)
error.lm

#Test error rate is 7.63 which means the crim per capita per town varies from + 8.080 and -8.080 for the predicted response.


#Lasso was 7.9912 and ridge was 7.806 whcih has not much differnece

#The test error rates obtained by performing best subset selection on the predictors as well as forward stepwise selection yield a lowest error rate when predictors rad,medv,black are considered.

#Ridge is our best model since the error rate was 7.806 which was the lowest.

#Yes the model contains all fatures since we use ridge model. 