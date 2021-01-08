#Lab3 Part1
#Sachin Mohan Sujir

require(ISLR)
require(MASS)
require(class)

# (a) Produce some numerical and graphical summaries of the Weekly data. 
# Do there appear to be any patterns? 

data(Weekly)
summary(Weekly)
pairs(Weekly)
dim(Weekly)

#The dataset contains 1089 records and 9 columns.
#The dataset contains 605 rows which have the direction up and 484 down.

#(a)Produce some numerical and graphical summaries of the Weekly data. Do there appear to be any patterns? 

#It seen from the scattert plotthat Volume and Year has logarithmic pattern


# (b) Use the full data set to perform a logistic regression with Direction as the response and the five lag variables plus Volume as predictors.
# Use the summary function to print the results. 
#Do any of the predictors appear to be statistically significant? If so, which ones? 

attach(Weekly)
logModel = glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, family="binomial")
summary(logModel)

#With the summary on, it seems like the value for lag2 is statistically significant.Its estimate is 0.05844.
#All other predictors are insignificant.

#The coefficients of these predictors suggests that as lag2 increases we have better probability of estimating the direction as Up. 
#Negative coefficients predict the response as Down.


# (c) Compute the confusion matrix and overall fraction of correct predictions. 
# Explain what the confusion matrix is telling you about the types of mistakes 
# made by logistic regression. 


logModel.probs = predict(logModel, type="response")
logModel.preds = ifelse(logModel.probs>.5, "Up", "Down")
confusionMatrix = table(Weekly$Direction,logModel.preds)
confusionMatrix

# We can observe from the confusion matrix that the model will predict
# Up prediction better than the Down predictions.The number of True Positives 
# suggests taht there is a trade off with respect to False Positives.
#We have obtained training accuracy of 56.1% by using this model.

#From the confusion matrix we can see that for the prediction "Down" the model had an accuracy of 
#**(54/(54+430))** i,e 11.1% acuurate but for the Up prediction the model has an accuracy of **557/(557+48)** i,e 92.06% accurate.


# (d) Now fit the logistic regression model using a training data period from 
# 1990 to 2008, with Lag2 as the only predictor. Compute the confusion matrix 
# and the overall fraction of correct predictions for the held out data 
# (that is, the data from 2009 and 2010).

trainingSet = Weekly[Year<2009,]
testSet = Weekly[Year>2008,]
logModel1 =glm(Direction ~ Lag2, data=trainingSet, family="binomial")
summary(logModel)



testProbablity = predict(logModel, type="response", newdata = testSet)
testsDir = Direction[Year>2008]
plot(testProbablity, col= ifelse(Direction[Year>2008]=="Down", "Blue","Green"), pch=16)
abline(h = 0.55, lwd= 5)

testPredictors = rep("Down", 104)
testPredictors[testProbablity>0.5] = "Up"
mean(testProbablity)
confMatrixLogReg = table(testsDir, testPredictors)
confMatrixLogReg

# correct predictions
correctPrediction= (confMatrixLogReg["Down", "Down"] + confMatrixLogReg["Up", "Up"])/sum(confMatrixLogReg)
correctPrediction

# Test error rate

testerrorLogReg = ((1-correctPrediction)*100)
testerrorLogReg

# The Test error rate for logistic regression is 37.5% or 0.375

# (e) Repeat (d) using LDA. 

#LDA MODEL
LDAmodel = lda(Direction~Lag2, data= trainingSet)
LDAmodel

LDApredictions = predict(LDAmodel, newdata=testSet, type="response")
LDAclass = LDApredictions$class
confMatrixLDA = table(testSet$Direction,LDAclass)
confMatrixLDA

# correct predictions
correctPrediction2= (confMatrixLDA["Down", "Down"] + confMatrixLDA["Up", "Up"])/sum(confMatrixLDA)
correctPrediction2

# Test error rate

testerrorLDA = ((1-correctPrediction2)*100)
testerrorLDA

# Test error rate for LDA is same as logistic regression which is 37.5%

# (f) Repeat (d) using QDA. 

#QDA MODEL
QDAmodel = qda(Direction~Lag2, data= trainingSet)
QDAmodel

QDApredictions = predict(QDAmodel, newdata=testSet, type="response")
QDAclass = QDApredictions$class
confMatrixQDA = table(testSet$Direction, QDAclass)
confMatrixQDA

# correct predictions
correctPrediction3= (confMatrixQDA["Down", "Down"] + confMatrixQDA["Up", "Up"])/sum(confMatrixQDA)
correctPrediction3

# Test error rate

testerrorQDA <- ((1-correctPrediction3)*100)
testerrorQDA

# Test error rate for QDA is 41.34%

# (g) Repeat (d) using KNN with K  = 1.

set.seed(1)
training.X <- cbind(trainingSet$Lag2)
testing.X <- cbind(testSet$Lag2)
training.Y <- cbind(trainingSet$Direction)
KNNPredictions <- knn(training.X, testing.X, training.Y, k=1)
confMatrixKNN<-table(testSet$Direction, KNNPredictions)
confMatrixKNN

# correct predictions
correctPrediction4= (confMatrixKNN["Down", "1"] + confMatrixKNN["Up", "2"])/sum(confMatrixKNN)
correctPrediction4

# Test error rate

testerrorKNN <- ((1-correctPrediction4)*100)
testerrorKNN

# Test error rate for KNN where K=1 is 50%

# (h) Which of these methods appears to provide the best results on this data? 
# -> LDA and Logistic regression are the best model developed.


# (h(i)) Experiment with different combinations of predictors, including possible transformations 
# and interactions, for each of the methods. Report the variables, method, and associated 
# confusion matrix that appears to provide the best results on the held out data. Note that you
# should also experiment with values for K in the KNN classifier. 

#-> Changing values of K in KNN.
# Let K=5.

KNNPredictions5 <- knn(training.X, testing.X, training.Y, k=5)
confMatrixKNN5<-table(testSet$Direction, KNNPredictions5)
confMatrixKNN5

# correct predictions
correctPrediction5<- (confMatrixKNN5["Down", "1"] + confMatrixKNN5["Up", "2"])/sum(confMatrixKNN5)
correctPrediction5

# Test error rate

testerrorKNN5 <- ((1-correctPrediction5)*100)
testerrorKNN5

#The test error rate has decreases from 50% to 47.11%
# Lets try for K=50

KNNPred50 <- knn(training.X, testing.X, training.Y, k=50)
confMatrixKNN50<-table(testSet$Direction, KNNPred50)
confMatrixKNN50

# correct predictions
correctPred50= (confMatrixKNN50["Down", "1"] + confMatrixKNN50["Up", "2"])/sum(confMatrixKNN50)
correctPred50

# Test error rate

testerrorKNN50 <- ((1-correctPred50)*100)
testerrorKNN50

#There is a change from 47% to 44% but not a significant change

#So change in K does not significantly affect the model.

# Now checking LDA with all Lag predictor
LDAmodel2 <- lda(Direction~Lag1 + Lag2 + Lag3 + Lag4 + Lag5, data= trainingSet)
LDAmodel2

LDApredictions2 <- predict(LDAmodel2, newdata=testSet, type="response")
LDAclass2 <- LDApredictions2$class
confMatrixLDA2 <- table(testSet$Direction,LDAclass2)
confMatrixLDA2

# correct predictions
correctPredictionLDA2<- (confMatrixLDA2["Down", "Down"] + confMatrixLDA2["Up", "Up"])/sum(confMatrixLDA2)
correctPredictionLDA2

# Test error rate

testerrorLDA2 <- ((1-correctPredictionLDA2)*100)
testerrorLDA2

# We can see that including all lag as predictor made the model worst than previous


# FOR QDA

QDAmodel2 <- qda(Direction~Lag1 + Lag2 + Lag3 + Lag4 + Lag5, data= trainingSet)
QDAmodel2

QDApredictions2 <- predict(QDAmodel2, newdata=testSet, type="response")
QDAclass2 = QDApredictions$class
confMatrixQDA2 <- table(testSet$Direction, QDAclass2)
confMatrixQDA2

# correct predictions
correctPredictionQDA2<- (confMatrixQDA2["Down", "Down"] + confMatrixQDA2["Up", "Up"])/sum(confMatrixQDA2)
correctPredictionQDA2

# Test error rate

testerrorQDA2 <- ((1-correctPrediction3)*100)
testerrorQDA2

#QDA-including all Lag predictor makes model worse.
