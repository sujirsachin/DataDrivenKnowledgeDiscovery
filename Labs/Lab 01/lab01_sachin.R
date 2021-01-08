rm(list=ls())
College=read.csv("College.csv",na.strings="?",header = T)

fix(College)
rownames(College) = College[,1]
fix(College)


College = College[,-1]
fix(College)

summary(College)

pairs(College[,1:10])

College$Private = as.factor(College$Private)
plot(College$Private,College$Outstate,xlab="Private",ylab= "Outstate",main="Boxplot: Outstate vs Private")

Elite = rep("No",nrow(College))
Elite[College$Top10perc > 50] = "Yes"
Elite = as.factor(Elite)
College = data.frame(College,Elite)
summary(College)

College$Elite = as.factor(College$Elite)
plot(College$Elite,College$Outstate,xlab="Elite",ylab= "Outstate",main="Boxplot: Outstate vs Elite")


par(mfrow=c(2,2))
hist(College$Enroll,col="Red")
hist(College$Accept,col="Green")
hist(College$Apps,col="Yellow")
hist(College$PhD,col="Blue")

sapply(College,class)
dim(College)

rangeEnroll=range(College$Enroll)
meanEnroll=mean(College$Enroll)
sdEnroll=sd(College$Enroll)
"Range of Enroll: " 
rangeEnroll

"Mean of Enroll :"
meanEnroll

"Standard Deviation of Enroll: "
sdEnroll


newenroll = College$Enroll[c(-100,-200)]
rangeEnroll_1=range(newenroll)
meanEnroll_1=mean(newenroll)
sdEnroll_1=sd(newenroll)
"Range of Enroll: " 
rangeEnroll_1

"Mean of Enroll :"
meanEnroll_1

"Standard Deviation of Enroll: "
sdEnroll_1

