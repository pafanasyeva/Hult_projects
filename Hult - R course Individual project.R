##### LIBRARIES AND FIRST PROCEDURES #####
library(dplyr)
library(ggplot2)
library(DataExplorer)
library(powerjoin)
library(corrplot)
library(vtreat)
library(MLmetrics)
library(pROC)
#install.packages('ROSE')
library(ROSE)
library(ranger)
library(caret)
library(ggthemes)
#install.packages("neuralnet")
library(neuralnet)
#install.packages("VIM")
library(VIM)


setwd("C:/Users/polin/Desktop/HULT/R/Hult_R_Classes/BAN1_Case_Info/A2_Hospital_Readmission/caseData")

#Cleaning the environment
#rm(list=ls())

#Creating function for correlation
corr_simple <- function(data=df,sig=0.5){
  #Convert data to numeric in order to run correlations
  #Convert to factor first to keep the integrity of the data - each value will become a number rather than turn into NA
  df_cor <- data %>% mutate_if(is.character, as.factor)
  df_cor <- df_cor %>% mutate_if(is.factor, as.numeric)
  #Run a correlation and drop the insignificant ones
  corr <- cor(df_cor)
  #Prepare to drop duplicates and correlations of 1     
  corr[lower.tri(corr,diag=TRUE)] <- NA 
  #Drop perfect correlations
  corr[corr == 1] <- NA 
  #Turn into a 3-column table
  corr <- as.data.frame(as.table(corr))
  #Remove the NA values from above 
  corr <- na.omit(corr) 
  #Select significant values  
  corr <- subset(corr, abs(Freq) > sig) 
  #Sort by highest correlation
  corr <- corr[order(-abs(corr$Freq)),] 
  #Print table
  print(corr)
  #Turn corr back into matrix in order to plot with corrplot
  mtx_corr <- reshape2::acast(corr, Var1~Var2, value.var="Freq")
  
  #Plot correlations visually
  corrplot(mtx_corr, is.corr=FALSE, tl.col="black", na.label=" ")
}


##### Initial work and EDA with data #####
patient_train <- read.csv('diabetesPatientTrain.csv')
med_train <- read.csv('diabetesMedsTrain.csv')
hosp_train <- read.csv('diabetesHospitalInfoTrain.csv')

#Upload the same data and apply the same rules for it
patient_test <- read.csv('diabetesPatientTest.csv')
med_test <- read.csv('diabetesMedsTest.csv')
hosp_test <- read.csv('diabetesHospitalInfoTest.csv')

#Combine all of the tables for fuller analysis in case of outliers
comb_tables <- power_left_join(patient_train,med_train, by = "tmpID")
comb_tables <- power_left_join(comb_tables,hosp_train, by = "tmpID")


#The first look at the tables 
#Checking the dimensionality of the tables
dim(patient_train)
dim(hosp_train)
dim(med_train)

#Checking the initial data in the tables
str(patient_train)
str(med_train)
str(hosp_train)

#Checking the information of the variables in the tables

#1
#Check the null values and then unique values to find any missing/null values
sum(is.na(patient_train))
lapply(patient_train, unique)

#As we can see there are several missing values in the data that don't have been written as null values. Change them to NA's.
patient_train$race <- gsub('[?]',NA,patient_train$race)
patient_train$payer_code <- gsub('[?]',NA,patient_train$payer_code)

#Do the same for test data
patient_test$race <- gsub('[?]',NA,patient_test$race)
patient_test$payer_code <- gsub('[?]',NA,patient_test$payer_code)

#Checking the proportion of the null values now and checking correlation between variables.
plot_missing(patient_train)
plot_correlation(patient_train)

summary(patient_train)
summary(patient_test)

mean(patient_train$age)
sd(patient_train$age)
mean(patient_train$age) + 2*sd(patient_train$age)
mean(patient_train$age) - 2*sd(patient_train$age)

max(patient_train$age)
min(patient_train$age)

#Max age is ok and cannot be considered an outlier. Nevertheless, the minimum age arouse suspicion.
#Therefore, check the information for people with minimum age (considering the possibility of newborn babies).
comb_tables[comb_tables$age==0,]
#As we can see, it is highly unlikely for people with weight 191 and 200 to be newborn babies.
#*Value as 100 will be taken as a baseline due to the fact that there can be outliers in this world and we don't want to
#*exclude them from consideration. Nevertheless, completely extreme cases should be excluded.
comb_tables[comb_tables$age<10&comb_tables$wgt>100,]

#Further analysis reveal that there are people with age <10 and weight >100, which is according to
#Centers for Disease Control and Prevention (National Center for Health Statistics)
#https://www.cdc.gov/growthcharts/html_charts/wtage.htm | https://greatist.com/health/how-much-should-i-weigh#based-on-age-and-sex
#is hightly unlikely. 
#Therefore, this age should be excluded and turned to NA values.

patient_train$age[patient_train$age<10&patient_train$wgt>100] <- NA

patient_test$age[patient_test$age<10&patient_test$wgt>100] <- NA

#Checking the weght variable
mean(patient_train$wgt)
sd(patient_train$wgt)
mean(patient_train$wgt) + 2*sd(patient_train$wgt)
mean(patient_train$wgt) - 2*sd(patient_train$wgt)

#Therefore, we can say that people with weight lower than 100 can be considered as outliers for our data.
#As we can see, people with the weight lower than 100 pounds are in the age range from 40 up to 89.
sort(comb_tables$age[comb_tables$wgt<100])


#Checking the statistical data for the normal weight for this age shows that mean weight for people in this age gap
#is around 200.9 and 194.7 with std.dev 2.0 and  1.9 accordingly.
#Hence, we can consider this weight as wrongly filled => weight values are changed for null's.
#https://www.cdc.gov/nchs/data/nhsr/nhsr122-508.pdf
patient_train$wgt[patient_train$age>40&patient_train$wgt<100] <- NA

patient_test$wgt[patient_test$age>40&patient_test$wgt<100] <- NA


#2
#Checking for any null values in the data
sum(is.na(hosp_train))
#Checking all the unique values in the columns to see if there any missing values marked differently from 'NA'.
lapply(hosp_train[,2:13], unique)

#Changing missing values for NA's
hosp_train <- mutate_all(hosp_train, ~ifelse(.=='', NA, .))
hosp_train$medical_specialty <- gsub('[?]',NA,hosp_train$medical_specialty)

hosp_test <- mutate_all(hosp_test, ~ifelse(.=='', NA, .))
hosp_test$medical_specialty <- gsub('[?]',NA,hosp_test$medical_specialty)

plot_missing(hosp_train)

summary(hosp_train)
summary(hosp_test)

#Checking all the variables
mean(hosp_train$time_in_hospital)
sd(hosp_train$time_in_hospital)
mean(hosp_train$time_in_hospital) + 2*sd(hosp_train$time_in_hospital)
mean(hosp_train$time_in_hospital) - 2*sd(hosp_train$time_in_hospital)

#--
mean(hosp_train$num_lab_procedures)
sd(hosp_train$num_lab_procedures)
mean(hosp_train$num_lab_procedures) + 2*sd(hosp_train$num_lab_procedures)
mean(hosp_train$num_lab_procedures) - 2*sd(hosp_train$num_lab_procedures)

#Checking the distribution of the number of lab procedures based on the medical speciality to check the theory that the
#biggest amount of lab.procedures are made by families.
#As we can see, it is not a majority inside the group, but still significant amount (~13.75%) compared to other groups.
addmargins(round(prop.table(table(hosp_train$medical_specialty,cut(hosp_train$num_lab_procedures, breaks = 3)),margin = 2)*100,2))

#Nevertheless, cheking the max amount of lab procedures, we can see that even if the group medical speciality belongs to families,
#it is highly unlikely to have 120 lab.procedures for only 2 visits in hospital.
#Therefore, it can be considered an outlier and converted to NA for further treatment.
comb_tables[comb_tables$num_lab_procedures==120,]
hosp_train$num_lab_procedures[hosp_train$num_lab_procedures==120] <-NA


#--

mean(hosp_train$num_procedures)
sd(hosp_train$num_procedures)
mean(hosp_train$num_procedures) + 2*sd(hosp_train$num_procedures)
mean(hosp_train$num_procedures) - 2*sd(hosp_train$num_procedures)

#As we can see, there are lots of cases where num_procedures are equal to 6. Therefore, we cannot consider them
#as outliers.
comb_tables[comb_tables$num_procedures==max(comb_tables$num_procedures),]

#--
mean(hosp_train$num_medications)
sd(hosp_train$num_medications)
mean(hosp_train$num_medications) + 2*sd(hosp_train$num_medications)
mean(hosp_train$num_medications) - 2*sd(hosp_train$num_medications)

#As it is unknown if 81 prescriptions is too many for 62 years old with these diagnoses. Therefore, it can't be considered
#as an outlier.
comb_tables[comb_tables$num_medications==81,]

#--

mean(hosp_train$number_outpatient)
sd(hosp_train$number_outpatient)
mean(hosp_train$number_outpatient) + 2*sd(hosp_train$number_outpatient)
mean(hosp_train$number_outpatient) - 2*sd(hosp_train$number_outpatient)

comb_tables[comb_tables$number_outpatient==max(comb_tables$number_outpatient),]

#--

mean(hosp_train$number_emergency)
sd(hosp_train$number_emergency)
mean(hosp_train$number_emergency) + 2*sd(hosp_train$number_emergency)
mean(hosp_train$number_emergency) - 2*sd(hosp_train$number_emergency)

#Number of emergencies equal to 42 can definitely be considered as an outlier. Especially, when we check that there is
#only one record like this and the max amount for this variable is 13 (if we exclude 42).
comb_tables[comb_tables$number_emergency==max(comb_tables$number_emergency),]
sort(unique(hosp_train$number_emergency))

hosp_train$number_emergency[hosp_train$number_emergency==42] <-NA



#--
#Nothing special was found for next variables.
mean(hosp_train$number_inpatient)
sd(hosp_train$number_inpatient)
mean(hosp_train$number_inpatient) + 2*sd(hosp_train$number_inpatient)
mean(hosp_train$number_inpatient) - 2*sd(hosp_train$number_inpatient)

comb_tables[comb_tables$number_inpatient==max(comb_tables$number_inpatient),]

mean(hosp_train$number_diagnoses)
sd(hosp_train$number_diagnoses)
mean(hosp_train$number_diagnoses) + 2*sd(hosp_train$number_diagnoses)
mean(hosp_train$number_diagnoses) - 2*sd(hosp_train$number_diagnoses)

comb_tables[comb_tables$number_diagnoses==max(comb_tables$number_diagnoses),]

#Diagnoses descriptions are excluded from the analysis as there are too many unique and long variables,
#moreover they are not duplicating each other from variable to variable.
names(hosp_train)
hosp_train <- subset(hosp_train, select=c("tmpID","admission_type_id","discharge_disposition_id",
                                          "admission_source_id","time_in_hospital","medical_specialty","num_lab_procedures",
                                          "num_procedures","num_medications","number_outpatient","number_emergency",
                                          "number_inpatient","number_diagnoses"))


names(hosp_test)
hosp_test <- subset(hosp_test, select=c("tmpID","admission_type_id","discharge_disposition_id",
                                          "admission_source_id","time_in_hospital","medical_specialty","num_lab_procedures",
                                          "num_procedures","num_medications","number_outpatient","number_emergency",
                                          "number_inpatient","number_diagnoses"))

#3
#Checking for any null values in the data
sum(is.na(med_train))
#Checking all the unique values in the columns to see if there any missing values marked differently from 'NA'.
lapply(med_train, unique)

names(med_train)
#Columns from 4 up to 21 are the names of the medicine for controlling diabetes.
#Checking the theory that variable 'diabetesMed' is built on the patients' prescriptions.
#If a person has at least one prescribed pill, 'diabetesMed' will be yes, otherwise - no.
med_train2 <- med_train %>%
  mutate(PrescribedMed = ifelse(metformin != 'No' | repaglinide != 'No'
                                | nateglinide != 'No' | chlorpropamide != 'No'| glimepiride != 'No'
                                | acetohexamide != 'No'| glipizide != 'No'
                                | glyburide != 'No' |tolbutamide != 'No' |
                                  pioglitazone != 'No'|rosiglitazone != 'No'|acarbose != 'No'|
                                  miglitol != 'No'|troglitazone != 'No'|
                                  tolazamide != 'No'|examide != 'No'|
                                  citoglipton != 'No'|insulin != 'No','Yes', 'No'))

corr_simple(med_train2, 0.5)

#As we can see, the new created based on the theory variable is correlated with diabetesMed by 0.99.
#It means these variables are the same.

#Moreover, the variable 'change' depends on the prescriptions. If the value of the prescribed medicine
#is 'Up' or 'Down', then it means that there were changes. If the values of the variables for medicine is
#'Steady' or 'No', then the value of the variable 'change' is also 'no'.
#Checking the values for each variable, shows us that the majority of the prescribed pills are either not changed,
#or were not prescribed at all. Hence, lots of them can be excluded.
lapply(med_train2[,2:23], table)

#Let's create the new variable to count the number of medications each patient has for further consideration in the model.
med_new_var <- function(data, columns) {
  for (column in columns) {
    data[[column]] <- ifelse(data[[column]] == 'No', 0, 1)
  }
  return(data)
}

med_train <- med_train %>%
  med_new_var(columns = c("metformin", "repaglinide", "nateglinide", "chlorpropamide", 
                                "glimepiride", "acetohexamide", "glipizide", "glyburide", 
                                "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", 
                                "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
                                "insulin"))

med_test <- med_test %>%
  med_new_var(columns = c("metformin", "repaglinide", "nateglinide", "chlorpropamide", 
                          "glimepiride", "acetohexamide", "glipizide", "glyburide", 
                          "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", 
                          "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
                          "insulin"))

med_train$numMed <- rowSums(med_train[,4:21])
med_test$numMed <- rowSums(med_test[,4:21])

#Therefore, as the two variables 'change' and 'diabetesMed' already includes the most important information
#from prescribed medicine, the individual medicine variables are excluded from further analysis.
names(med_train)
med_train <- subset(med_train, select=c("tmpID","max_glu_serum","A1Cresult",
                                          "change","diabetesMed","numMed"))

names(med_test)
med_test <- subset(med_test, select=c("tmpID","max_glu_serum","A1Cresult",
                                        "change","diabetesMed","numMed"))

summary(med_train)
plot_missing(med_train)
plot_correlation(med_train)


#Combining changed data tables in the final one for further modelling.
all_df <- power_left_join(patient_train,med_train, by = "tmpID")
all_df <- power_left_join(all_df,hosp_train, by = "tmpID")

plot_missing(all_df)
dim(all_df)

all_df_test <- power_left_join(patient_test,med_test, by = "tmpID")
all_df_test <- power_left_join(all_df_test,hosp_test, by = "tmpID")

plot_missing(all_df_test)
dim(all_df_test)



##### PREPARING THE VARIABLES #####


#Identify the informative and target variables for the train data
names(all_df)
targetVar       <- names(all_df)[7]
informativeVars <- names(all_df)[c(2:6, 8:24)]


#SAMPLING
#Segmenting the prep train data
set.seed(2023)
idx         <- sample(1:nrow(all_df),.1*nrow(all_df))
prepData    <- all_df[idx,]
nonPrepData <- all_df[-idx,]
nameKeeps <- all_df$tmpID[-idx] #tracking ID of the parients

#Designing a categorical treatment
plan <- designTreatmentsC(prepData, 
                          informativeVars,
                          targetVar, 1)

#Applying treatment on non-prepared data
treatedX <- prepare(plan, nonPrepData)


#Applying treatment on test data
treatedXtest <- prepare(plan, all_df_test)



## Further modifications only for train data
# Partition to avoid over fitting
set.seed(2023)
idx        <- sample(1:nrow(treatedX),.8*nrow(treatedX))
train      <- treatedX[idx,]
validation <- treatedX[-idx,]






##### OVERSAMPLING/UNDERSAMPLING ####

#Check the data for over-/under- sampling
table(train$readmitted_y)
round(prop.table(table(train$readmitted_y))*100,2)

#As we can see, our data has an oversampling problem, which is needed to be fixed.
#Use ROSE package and ovun.sample function for it.

#Changing the target variable, so that function will work.
train$readmitted_y <- ifelse(train$readmitted_y=="TRUE",1,0)

#Dealing with oversampling in the train data, so that model will be fitted correctly without bias.
train_over <- ovun.sample(readmitted_y~., data=train,
                           p=0.5, seed=1,
                           method="over")$data

#Checking for the oversampling problem again.
table(train_over$readmitted_y)
round(prop.table(table(train_over$readmitted_y))*100,2)

#Changing the target variable for validation and test datasets for further metrics applications.
validation$readmitted_y <- ifelse(validation$readmitted_y=="TRUE",1,0)
treatedXtest$readmitted_y <- ifelse(treatedXtest$readmitted_y=="TRUE",1,0)





##### TRAINING MODELS #####

##### Logistic Regression Model #####
#Fitting a logistic regression model
fit <- glm(readmitted_y ~., data = train_over, family ='binomial')
summary(fit)

#Backward Variable selection to reduce chances of multicollinearity
bestFit <- step(fit, direction='backward')
summary(bestFit)

#Comparing model size of first model and best fitted one.
length(coefficients(fit))
length(coefficients(bestFit))

#Getting predictions on validation
readmitPreds <- predict(bestFit,  validation, type='response')
tail(readmitPreds)

#Classifying on the initial level 0.5/50%
cutoff      <- 0.5
readmitClasses <- ifelse(readmitPreds >= cutoff, 1,0)


results_glm <- data.frame(actual  = validation$readmitted_y,
                      classes = readmitClasses,
                      probs   = readmitPreds)
head(results_glm)


#Geting a confusion matrix
confMat <- ConfusionMatrix(results_glm$classes, results_glm$actual)

#Checking the accuracy of the model on validation dataset
sum(diag(confMat)) / sum(confMat)
Accuracy(results_glm$classes, results_glm$actual)

#Checking the visual separation of our classes
ggplot(results_glm, aes(x=probs, color=as.factor(actual))) +
  geom_density() + 
  geom_vline(aes(xintercept = cutoff), color = 'darkgreen')

# ROC
ROCobj <- roc(results_glm$classes, results_glm$actual*1)
plot(ROCobj)

# AUC
AUC(results_glm$actual*1,results_glm$classes)




##### Random Forest Model #####

#Creating vectors to store future parameter for number of trees and oddserved error
numTreesVec <- vector()
oobError  <- vector()
nTreeSearch <- seq(from = 200, to = 600, by=20)

#Create a grid for future training grid
param_grid <- expand.grid(
  mtry = c(1, 3, 5),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 5, 10))


#Create training control object for 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

#Fit the ranger model with grid search
set.seed(2023)

for(i in 1:length(nTreeSearch)){
  print(i)
  model <- train(as.factor(readmitted_y) ~ .,
                data = train_over,
                method = "ranger",
                num.trees = nTreeSearch[i],
                trControl = train_control,
                tuneGrid = param_grid,
                importance = "permutation")
  numTreesVec[i] <- model$finalModel$num.trees
  oobError[i] <- model$finalModel$prediction.error
}  
  
#Print the best model and its performance
print(model$bestTune)
print(model)

results <- data.frame(ntrees =numTreesVec,
                      oobError = oobError)
min_obb <- results[results$oobError==min(results$oobError),][,1]
#Choosing number of trees based according to the minimal oobError.

ggplot(results, aes(x=ntrees,y=oobError)) + geom_line(alpha =0.25, color = 'red') +
  theme_gdocs()+
  geom_smooth(method = "loess")

#Store the best parameters for final training
best_params <- data.frame(mtry = model$bestTune[,1], spr = model$bestTune[,2], siz = model$bestTune[,3])


rf_model <- ranger(as.factor(readmitted_y) ~ .,
                   data = train_over,
                   num.trees = min_obb,
                   importance = "permutation",
                   mtry=best_params[1,1],
                   splitrule='gini',
                   min.node.size=best_params[1,3],
                   probability = TRUE)

#Prediction on validation part of the data to estimate the model 
valid_rf <- predict(rf_model, validation)

#Transform to 1/0 values
readmitOutcomeTest <- ifelse(valid_rf$predictions[,2]>=0.5, 1,0)

#Checking the accuracy and confusion matrix
Accuracy(as.factor(readmitOutcomeTest), 
         as.factor(validation$readmitted_y))

confusionMatrix(as.factor(readmitOutcomeTest), 
                as.factor(validation$readmitted_y))



#Looking at variance importance
varImpDF <- data.frame(variables = names(rf_model$variable.importance),
                       importance = rf_model$variable.importance,
                       row.names = NULL)
varImpDF <- varImpDF[order(varImpDF$importance, decreasing = T),]
ggplot(varImpDF, aes(x=importance, y = reorder(variables, importance))) + 
  geom_bar(stat='identity', position = 'dodge') + 
  ggtitle('Variable Importance') + 
  theme_gdocs()



##### Neural Networks model training #####


#Defining the parameter grid for tuning
param_grid_nn <- expand.grid(
  size = c(5, 10,12),
  decay = c(0, 0.1, 0.01)
)

#Creating training control object for 5-fold cross-validation
train_control_nn <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

#Fitting the neural network model with grid search
set.seed(2023)
model_nn <- train(
  as.factor(readmitted_y) ~ .,
  data = train_over,
  method = "nnet",
  trControl = train_control_nn,
  tuneGrid = param_grid_nn,
  verbose = FALSE)

#Printing the best tuning parameters and the corresponding performance metrics
print(model_nn$bestTune)
print(model_nn$results)

#Saving best parameters for final training
best_params_nn <- data.frame(mtry = model_nn$bestTune[,1], spr = model_nn$bestTune[,2])

fin_model_nn <- nnet(
  as.factor(readmitted_y) ~ .,
  data = train_over,
  size = best_params_nn[,1],
  decay = best_params_nn[,2],
  verbose = FALSE)

#Predicting on validation
valid_nn <- predict(fin_model_nn, validation)

#Creating 1/0 values
readmitOutcomeTest_nn <- ifelse(valid_nn>=0.5, 1,0)

#Checking accuracy
Accuracy(as.factor(readmitOutcomeTest_nn), 
         as.factor(validation$readmitted_y))

#Creating confusion matrix
confusionMatrix(as.factor(readmitOutcomeTest_nn), 
                as.factor(validation$readmitted_y))






##### PREDICTING READMITION ON TEST DATA ##### 

##### Logistic Regression prediction on test data #####

#Getting predictions
readmitPredsTest <- predict(bestFit,  treatedXtest, type='response')
tail(readmitPredsTest)

#Classify by 0.45 level
cutoff      <- 0.5
readmitClassesTest <- ifelse(readmitPredsTest >= cutoff, 1,0)


resultsTest <- data.frame(actual  = treatedXtest$readmitted_y,
                          classes = readmitClassesTest,
                          probs   = readmitPredsTest)
head(resultsTest)


#Getting a confusion matrix
confMatTest <- ConfusionMatrix(resultsTest$classes, resultsTest$actual)

#Checking the accuracy of the model
sum(diag(confMatTest)) / sum(confMatTest)
Accuracy(resultsTest$classes, resultsTest$actual)

#Check visually
ggplot(resultsTest, aes(x=probs, color=as.factor(actual))) +
  geom_density() + 
  geom_vline(aes(xintercept = cutoff), color = 'darkgreen')

# ROC
ROCobj <- roc(resultsTest$classes, resultsTest$actual*1)
plot(ROCobj)

# AUC
AUC(resultsTest$actual*1,resultsTest$classes)



##### Random Forest prediction on test data #####

#Getting predictions
rf_readmitPredsTest <- predict(rf_model,  treatedXtest)

#Classification by the cut level 0.5
readmitClassTest <- ifelse(rf_readmitPredsTest$predictions[,2]>=0.5, 1,0)

Accuracy(as.factor(readmitClassTest), 
         as.factor(treatedXtest$readmitted_y))

confusionMatrix(as.factor(readmitClassTest), 
                as.factor(treatedXtest$readmitted_y))




##### Neural Networks prediction on test data #####

#Getting predictions
pred_nn_res <- predict(fin_model_nn, treatedXtest)

#Classification by the cut level 0.5
readmit_nn_Test <- ifelse(pred_nn_res>=0.5, 1,0)

Accuracy(as.factor(readmit_nn_Test), 
         as.factor(treatedXtest$readmitted_y))

confusionMatrix(as.factor(readmit_nn_Test), 
                as.factor(treatedXtest$readmitted_y))




##### BENDING AND FINAL PART #####

##### Combining all the results in one table with the patients ID #####

#Creating the combined data frame
fin_df <- data.frame(cbind(patient_id = all_df_test$tmpID, readmitted = treatedXtest$readmitted_y, 
                           glm_res = rep(NA, nrow(all_df_test)), rf_res = rep(NA, nrow(all_df_test)),
                           nn_res = rep(NA, nrow(all_df_test))))

#Add values from predictions
fin_df$glm_res <- round(readmitPredsTest,8)
fin_df$rf_res <- as.numeric(rf_readmitPredsTest$predictions[,2])
fin_df$nn_res <- pred_nn_res[,1]

#Checking that no data is missing
sum(is.na(fin_df))




fit_all <- glm(readmitted ~., data = fin_df[,2:5], family ='binomial')
summary(fit_all)

#Backward Variable selection to reduce chances of multicollinearity
bestFit_all <- step(fit_all, direction='backward')
summary(bestFit_all)

#Comparing model size of first model and best fitted one.
length(coefficients(fit_all))
length(coefficients(bestFit_all))

#Getting predictions on validation
finPred <- predict(bestFit_all,  fin_df, type='response')


cutoff      <- 0.5
fin_readmitClass <- ifelse(finPred >= cutoff, 1,0)


results_glm_fin <- data.frame(actual  = fin_df$readmitted,
                          patient = fin_df$patient_id,
                          classes = fin_readmitClass,
                          probs   = finPred)
head(results_glm_fin)


#Getting a confusion matrix
confMat_fin <- ConfusionMatrix(results_glm_fin$classes, results_glm_fin$actual)

sum(diag(confMat_fin)) / sum(confMat_fin)
Accuracy(results_glm_fin$classes, results_glm_fin$actual)


##### Results of the models #####

glm_cm <- confusionMatrix(as.factor(resultsTest$classes), as.factor(treatedXtest$readmitted_y))
rf_cm <- confusionMatrix(as.factor(readmitClassTest), as.factor(treatedXtest$readmitted_y))
nn_cm <- confusionMatrix(as.factor(readmit_nn_Test), as.factor(treatedXtest$readmitted_y))
glm_all <- confusionMatrix(as.factor(results_glm_fin$classes), as.factor(treatedXtest$readmitted_y))


# Print accuracy, precision, recall and F1 score for each model
cat("Model performance:\n")
cat(paste("GLM: accuracy =", round(glm_cm$overall['Accuracy'], 3), 
          "precision =", round(glm_cm$byClass['Pos Pred Value'], 3),
          "recall =", round(glm_cm$byClass['Sensitivity'], 3),
          "F1 score =", round(glm_cm$byClass['F1'], 3), "\n"))
cat(paste("Random Forest: accuracy =", round(rf_cm$overall['Accuracy'], 3), 
          "precision =", round(rf_cm$byClass['Pos Pred Value'], 3),
          "recall =", round(rf_cm$byClass['Sensitivity'], 3),
          "F1 score =", round(rf_cm$byClass['F1'], 3), "\n"))
cat(paste("Neural Network: accuracy =", round(nn_cm$overall['Accuracy'], 3), 
          "precision =", round(nn_cm$byClass['Pos Pred Value'], 3),
          "recall =", round(nn_cm$byClass['Sensitivity'], 3),
          "F1 score =", round(nn_cm$byClass['F1'], 3), "\n"))
cat(paste("GLM on all models: accuracy =", round(glm_all$overall['Accuracy'], 3), 
          "precision =", round(glm_all$byClass['Pos Pred Value'], 3),
          "recall =", round(glm_all$byClass['Sensitivity'], 3),
          "F1 score =", round(glm_all$byClass['F1'], 3), "\n"))


#As we can see, the accuracy of the prediction based on other models is not so different from random forest.
#Therefore, the results from the random forest model will be used due to better interpretation and easiness of
#the model.
#Notice: In the real test data it will be impossible to check the accuracy of the model.
#       As there is an option to check the final accuracy on test data, this opportunity will be used
#       to choose the final model.



#Final filtering of people to extract top 100 according to the model
#Sorting the data and extracting IDs for the top 100 patients.
fin_df <- fin_df[order(fin_df$rf_res, decreasing = T),]
fin_df[order(fin_df$rf_res, decreasing = T),]

hund_patients_mod <- fin_df[1:100,1]

#Saving the top number of people that were trully readmitted and have high possibility according to the model.
true_readm <- fin_df[order(fin_df$readmitted, fin_df$rf_res, decreasing = T),]


all_info <- subset(all_df_test, all_df_test$tmpID %in% hund_patients_mod)



##### FINAL EDA ON 100 PATIENTS #####

dim(all_info)
summary(all_info)


#Creating Age_bins
all_info$age_bins <- cut(all_info$age,breaks = c(min(all_info$age)-1, 45, 60, 75, 85, max(all_info$age)))

#Fill in NA values in initial data table in numeric values by kNN Imputer
all_info <- kNN(all_info, k=5)

colnames(all_info)

#Saving df with imputation flag
all_info2_imp <- all_info

#Cutting imputation flags
all_info <- all_info[,1:25]

#####Analysis of gender#####

#Grouping by gender
gend <- all_info %>%
  group_by(gender) %>%
  summarise(counts = n())

ggplot(gend, aes(x = gender, y = counts, fill = gender)) +
  geom_bar(stat = "identity") +
  labs(title = "Gender Distribution", x = "Gender", y = "Count") +
  theme_minimal()


#Same in proportions
gend_prop <- all_info %>%
  group_by(gender) %>%
  summarise(counts = n()) %>%
  mutate(prop = prop.table(counts))


ggplot(gend, aes(x = gender, y = counts/sum(counts), fill = gender)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Gender", y = "Percentage", title = "Gender Distribution") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


##### Age #####

#Checking the age distribution
#Same in proportions
age_prop <- all_info %>%
  group_by(age_bins) %>%
  summarise(counts = n()) %>%
  mutate(prop = prop.table(counts))

ggplot(age_prop, aes(x = age_bins, y = counts/sum(counts), fill = age_bins)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Gender", y = "Percentage", title = "Gender Distribution") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))



##### Age and gender distribution #####
ggplot(all_info, aes(x = age_bins, y = ..count.., fill = gender)) +
  geom_bar() +
  labs(title = "Gender Distribution", x = "Age Bins", y = "Count") +
  scale_fill_manual(values = c("lightgreen","skyblue" )) +
  theme_minimal()






##### Number of medications, diagnoses and weight #####

#Additional graph to research weight and age distribution to understand if the patients have obesity.
#As we can see from the plot and comparing to the statistics mentioned previously,
#we can see that for the most of them their weight are equal to mean for the population.

all_info$weight_bin <- cut(all_info$wgt, breaks = seq(0, 300, 50), labels = c("0-50", "51-100", "101-150", "151-200",
                                                                              "201-250", "251-300"))

ggplot(all_info, aes(x = weight_bin, fill = age_bins)) +
  geom_bar(position = "dodge") +
  labs(title = "Count of Age Bin by Weight Bin",
       x = "Weight Bin", y = "Count") +
  scale_fill_discrete(name = "Age Bin")


#Main graph
ggplot(agg_data, aes(x = wgt, y = num_medications, color = factor(number_diagnoses))) +
  geom_point(size = 3) +
  labs(title = "Total Number of Medications by Weight and Number of Diagnoses",
       x = "Weight", y = "Number of Medications",
       color = "Number of Diagnoses") +
  theme_minimal()




##### Race #####

race_gr <- all_info %>%
  group_by(race) %>%
  summarise(counts = n())

ggplot(race_gr, aes(x = race, y = counts, fill = race)) +
  geom_bar(stat = "identity") +
  labs(title = "Gender Distribution", x = "Gender", y = "Count") +
  theme_minimal()


#Same in proportions
race_gr_prop <- all_info %>%
  group_by(race) %>%
  summarise(counts = n()) %>%
  mutate(prop = prop.table(counts))


ggplot(race_gr_prop, aes(x = race, y = counts/sum(counts), fill = race)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Gender", y = "Percentage", title = "Gender Distribution") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))


#--
#Check the distribution in the initial test dataset
race_gr_prop_all <- all_df_test %>%
  group_by(race) %>%
  summarise(counts = n()) %>%
  mutate(prop = prop.table(counts))


ggplot(race_gr_prop_all, aes(x = race, y = counts/sum(counts), fill = race)) +
  geom_bar(stat = "identity") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Gender", y = "Percentage", title = "Gender Distribution") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

#--

#Race and times in hospital
race_time_counts <- all_info %>%
  group_by(race, time_in_hospital) %>%
  summarise(counts = n()) %>%
  mutate(prop = prop.table(counts))

#Plotting
ggplot(race_time_counts, aes(x = time_in_hospital, y = counts/sum(counts), fill = race)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Time in Hospital (days)", y = "Percentage", title = "Race and Time in Hospital Distribution") +
  facet_grid(~ race) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))







##########

aggregate(readmitted_y~num_medications+wgt+number_diagnoses, all_info, sum)

aggr_readm <- aggregate(readmitted_y ~ num_medications + wgt + number_diagnoses, data = all_info, FUN = sum, na.rm = TRUE)
aggr_readm$prop_readmitted <- aggr_readm$readmitted_y / sum(all_info$readmitted_y, na.rm = TRUE)



##### Additional analysis of readmitted #####

table(all_info$readmitted_y)
prop.table(all_info$readmitted_y)


# Create scatter plot
ggplot(all_info, aes(x = age, y = gender, color = readmitted_y)) +
  geom_point() +
  labs(title = "Gender and Age Distribution", x = "Age", y = "Gender", color = "Readmission") +
  theme_minimal()


aggregate(readmitted_y~number_inpatient+number_diagnoses, all_info, sum)


#Checking readmission based on true readmitted values
table_data <- all_info %>%
  group_by(gender, age_bins, readmitted_y) %>%
  summarise(counts = n()) %>%
  mutate(prop = counts / sum(counts))

#Create the bar plot
ggplot(table_data, aes(x = age_bins, y = prop, fill = readmitted_y)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~gender) +
  labs(title = "Readmission Distribution by Gender and Age Bins", x = "Age Bins", y = "Proportion") +
  scale_fill_manual(values = c("#619CFF", "#F8766D")) +
  theme_minimal()



#3D graph for readmitted patients depending on their weight, number of medications and number of diagnoses
library(plotly)

all_info %>%
  group_by(num_medications, wgt, number_diagnoses) %>%
  summarize(prop_readmitted = mean(readmitted_y == 1, na.rm = TRUE)) %>%
  plot_ly(x = ~num_medications, y = ~wgt, z = ~number_diagnoses, color = ~prop_readmitted, colors = "Blues") %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = "Number of Medications"),
                      yaxis = list(title = "Weight"),
                      zaxis = list(title = "Number of Diagnoses"),
                      camera = list(eye = list(x = -1.8, y = -1.8, z = 0.5))),
         margin = list(l = 0, r = 0, b = 0, t = 0))


all_info %>%
  filter(readmitted_y == 1) %>%
  group_by(num_medications, wgt, number_diagnoses) %>%
  summarize(mean_readmitted = mean(readmitted_y, na.rm = TRUE)) %>%
  ggplot(aes(x = num_medications, y = mean_readmitted, color = number_diagnoses, size = wgt)) +
  geom_point() +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Mean Readmitted vs. Num Medications by Number of Diagnoses and Weight",
       x = "Num Medications", y = "Mean Readmitted",
       color = "Number of Diagnoses", size = "Weight") +
  theme_minimal()
