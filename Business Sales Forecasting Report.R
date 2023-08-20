########################################################
#####            Business Insight Report            ####
#               Authors:Polina Afanasyeva              #
########################################################


# Data Source: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

#### Description of the initial data ####
#*train.csv
#*The training data, comprising time series of features store_nbr, family, and onpromotion as well as the target sales.
#*store_nbr - identifies the store at which the products are sold.
#*family - identifies the type of product sold.
#*sales - gives the total sales for a product family at a particular store at a given date. 
#*Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese,
#*for instance, as opposed to 1 bag of chips).
#*onpromotion - gives the total number of items in a product family that were being promoted at a store at a given date.
#*
#*test.csv
#*The test data, having the same features as the training data. The dates in the test data are for the 15 days after the
#*last date in the training data.
#*
#*stores.csv
#*Store metadata, including city, state, type, and cluster.
#*cluster is a grouping of similar stores.
#*
#*holidays_events.csv
#*Holidays and Events, with metadata
#*A holiday that is transferred officially falls on that calendar day, but was moved to another date by the government.
#*A transferred day is more like a normal day than a holiday. 



#### Libraries ####
library(tidyr)
library(dplyr)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(ggthemes)
library(plotly)
library(tseries)
library(rugarch)
#install.packages('vtreat')
library(vtreat)
#install.packages('DataExplorer')
library(DataExplorer)
#install.packages('ROSE')
library(ROSE)
#install.packages('MLmetrics')
library(MLmetrics)
#install.packages('pROC')
library(pROC)


#### Uploading the data ####
main_train <- read.csv('C:/Users/polin/Desktop/HULT/Forecasting/sales forecasting/train.csv')
#main_test <- read.csv('C:/Users/polin/Desktop/HULT/Forecasting/sales forecasting/test.csv')
main_stores <- read.csv('C:/Users/polin/Desktop/HULT/Forecasting/sales forecasting/stores.csv')
#main_oil <- read.csv('C:/Users/polin/Desktop/HULT/Forecasting/sales forecasting/oil.csv')
main_holidays <- read.csv('C:/Users/polin/Desktop/HULT/Forecasting/sales forecasting/holidays_events.csv')

#* As this assignment does not require the actual test predictions, it will be skipped and
#* only validation sample will be used to evaluate the models.

#### Preprocessing the data ####

#main_train
sapply(main_train,class)
summary(main_train)
dim(main_train)
str(main_train)
sum(is.na(main_train))

#The store with the highest sales looks like an outluer 

#main_stores
sapply(main_stores,class)
summary(main_stores)
dim(main_stores)
str(main_stores)
sum(is.na(main_stores))


#main_holidays
sapply(main_holidays,class)
summary(main_holidays)
dim(main_holidays)
str(main_holidays)
sum(is.na(main_holidays))

#main_test
sapply(main_test,class)
summary(main_test)
dim(main_test)
str(main_test)
sum(is.na(main_test))




# Changing the date character type to date type
main_train$date <- as.Date(main_train$date, format = "%Y-%m-%d")
main_holidays$date <- as.Date(main_holidays$date, format = "%Y-%m-%d")

# Combination of two tables
tr_stor <- left_join(main_train, main_stores, by = "store_nbr")
tr_st_hol <- left_join(tr_stor, main_holidays, by = "date")

unique(tr_st_hol$type.y)
unique(tr_st_hol$locale)
sum(is.na(tr_st_hol))
colSums(is.na(tr_st_hol))
na_lines <- tr_st_hol[!complete.cases(tr_st_hol), ]

#Checking the earliest and latest dates for tables to understand the null values results
min(tr_stor$date)
max(tr_stor$date)
min(main_holidays$date)
max(main_holidays$date)


# Excluding columns that are not needed for further analysis
colnames(tr_st_hol)

tr_st_hol$transferred <- ifelse(tr_st_hol$transferred == "True", 1, 0)
tr_st_hol$transferred[is.na(tr_st_hol$transferred)] <- 0
tr_st_hol_f <- tr_st_hol[, c("store_nbr","family","onpromotion","city",
                             "state", "type.x","cluster","type.y","locale","transferred","sales")]

tr_st_hol_f <- tr_st_hol_f %>%
  rename(type = type.x,
         type_holiday = type.y)

# Exclude the row with the highest sales as an outlier
tr_st_hol_f <- subset(tr_st_hol_f, sales < max(tr_st_hol_f$sales))


# Counting unique value for type_holiday for further cleaning
tr_st_hol_f %>%
  count(type_holiday) %>%
  arrange(desc(n))

# Group data without date for further processing before prediction models
gr_data <- tr_st_hol_f %>%
  group_by(store_nbr, family, onpromotion, city, state, type, cluster, type_holiday, locale, transferred) %>%
  summarize(total_sales = sum(sales)) 

# Checking the number of values bigger than mean
sum(gr_data$total_sales > mean(gr_data$total_sales))
sum(gr_data$total_sales < mean(gr_data$total_sales))

# Creating boolean variable for successful shops (bigger than mean sales)
gr_data$popul_shops <- ifelse(gr_data$total_sales >  mean(gr_data$total_sales), 1, 0)
colnames(gr_data)
gr_data2 <- gr_data[, c("store_nbr","family","onpromotion","city",
                        "state", "type","cluster","type_holiday","locale","transferred","popul_shops")]

# Checking the correlation between variables with the created function
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

corr_simple(tr_st_hol_f,0.5)




#### SAMPLING ####
#Segmenting the prep train data
set.seed(2023)
idx         <- sample(1:nrow(gr_data2),.1*nrow(gr_data2))
prepData    <- gr_data2[idx,]
nonPrepData <- gr_data2[-idx,]
nameKeeps <- gr_data2$store_nbr[-idx]

# Treating the variables
colnames(gr_data2)
targetVar       <- names(gr_data2)[11]
informativeVars <- names(gr_data2)[1:10]

plan <- designTreatmentsC(prepData, 
                          informativeVars,
                          targetVar, 1)

#Applying treatment on non-prepared data
treatedX <- prepare(plan, nonPrepData)



## Further modifications only for train data
# Partition to avoid over fitting
set.seed(2023)
idx        <- sample(1:nrow(treatedX),.8*nrow(treatedX))
train      <- treatedX[idx,]
validation <- treatedX[-idx,]


#### Over-/Under- Sampling ####
#Check the data for over-/under- sampling
table(train$popul_shops)
round(prop.table(table(train$popul_shops))*100,2)

#Dealing with oversampling in the train data, so that model will be fitted correctly without bias.
train_over <- ovun.sample(popul_shops~., data=train,
                          p=0.5, seed=1,
                          method="under")$data

#Checking for the oversampling problem again.
table(train_over$popul_shops)
round(prop.table(table(train_over$popul_shops))*100,2)




#### PREDICTION MODELS ####

##### Logistic Regression Model #####
#Fitting a logistic regression model
fit <- glm(popul_shops ~., data = train_over, family ='binomial')
summary(fit)

#Backward Variable selection to reduce chances of multicollinearity
bestFit <- step(fit, direction='backward')
summary(bestFit)


# Calculating the exp for the coefficients
#store_nbr(-0.007613867)
exp(-7.643e-03)-1
#family(3.187323e+39 and 0.3016074)
exp(9.096e+01)-1
exp(2.636e-01)-1
#type_holiday(1.30712 and 24.38098)
exp(8.360e-01)-1
exp(3.234e+00)-1
#onpromotion(-0.02011494)
exp(-2.032e-02)-1
#locale(13.64355 and 10.40441)
exp(2.684e+00)-1
exp(2.434e+00)-1


#Comparing model size of first model and best fitted one.
length(coefficients(fit))
length(coefficients(bestFit))



#Getting predictions on validation
readmitPreds <- predict(bestFit,  validation, type='response')
tail(readmitPreds)

#Classifying on the initial level 0.5/50%
cutoff      <- 0.50
readmitClasses <- ifelse(readmitPreds >= cutoff, 1,0)


results_glm <- data.frame(actual  = validation$popul_shops,
                          classes = readmitClasses,
                          probs   = readmitPreds)
head(results_glm)


# Getting a confusion matrix
confMat <- ConfusionMatrix(results_glm$classes, results_glm$actual)

# Checking the accuracy of the model on validation dataset
Accuracy(results_glm$classes, results_glm$actual)

#Checking the visual separation of our classes
ggplot(results_glm, aes(x=probs, color=as.factor(actual))) +
  geom_density() + 
  geom_vline(aes(xintercept = cutoff), color = 'darkgreen')


# Getting the full results from the Confusion Matrix
confusionMatrix(data= as.factor(as.numeric(readmitPreds>0.5)),
                reference= as.factor(as.numeric(validation$popul_shops)))




#### GINI Decision Tree ####
gini_tree <- rpart(popul_shops ~ ., 
                 data=train_over, method="class" , cp=0.004)
rpart.plot(gini_tree, type=1, extra=1)

tree_predict <- predict(gini_tree, validation, type="prob")

confusionMatrix(data = as.factor(as.numeric(tree_predict[,2]>0.5)) ,
                reference= as.factor(as.numeric(validation$popul_shops)))



# Looking at variance importance
varImpDF <- data.frame(variables = names(gini_tree$variable.importance),
                       importance = gini_tree$variable.importance,
                       row.names = NULL)
varImpDF <- varImpDF[order(varImpDF$importance, decreasing = T),]
ggplot(varImpDF, aes(x=importance, y = reorder(variables, importance))) + 
  geom_bar(stat='identity', position = 'dodge') + 
  ggtitle('Variable Importance') + 
  theme_gdocs()  +
  ylab('Names of variables') +
  xlab('Importance')




#### FORECASTING PREPROCESSING ####
# Preparing data for forecasting

# Grouping data and sorting it
grouped_data <- tr_stor %>%
  group_by(store_nbr,family) %>%
  summarize(total_sales = sum(sales))

tot_sales_gr <- tr_stor %>%
  group_by(store_nbr) %>%
  summarize(total_sales = sum(sales))%>%
  arrange(desc(total_sales))

# For further analysis and forecasting only top-3 shops by total sales (44, 45, 47)
# and least-3 (32, 22, 52) will be chosen.

#For this 6 shops 3 top selling categories will be chosen for analysis.
sorted_data <- grouped_data %>%
  arrange(store_nbr, desc(total_sales)) %>%
  filter(store_nbr %in% c(44, 45, 47, 32, 22, 52))


# Choosing the top and least 3 stores
top_3_per_store <- sorted_data %>% group_by(store_nbr) %>% slice_head(n = 3)
bottom_3_per_store <- sorted_data %>% group_by(store_nbr) %>% slice_tail(n = 3)


# Checking the unique categories that fall into our filtering for these shops
unique(top_3_per_store$family)
unique(bottom_3_per_store$family)

# Therefore, the top-3 categories between these stores are 'Grocery I', 'Beverages', 'Cleaning', 'Produce'.
# The least selling categories are 'Baby Care', 'Lawn and Garden', 'Books', 'Ladieswear', 'Home appliances', 'Hardware'.

# As the least categories sales are close to 0, they won't be considered in further analysis.

### Forecasting for the top shops for top categories ###

### Filtering the data for these shops for the top categories:


##  Top stores:
# Grocery I
store_44_gr <- main_train %>%
  filter(store_nbr == 44, family == 'GROCERY I') %>%
  select(date, sales)

store_45_gr <- main_train %>%
  filter(store_nbr == 45, family == 'GROCERY I') %>%
  select(date, sales)

store_47_gr <- main_train %>%
  filter(store_nbr == 47, family == 'GROCERY I') %>%
  select(date, sales)

combined_gr_top <- bind_cols(date = store_44_gr$date, store_44 = store_44_gr$sales,
                           store_45 = store_45_gr$sales, store_47 = store_47_gr$sales)

# Beverages
store_44_bev <- main_train %>%
  filter(store_nbr == 44, family == 'BEVERAGES') %>%
  select(date, sales)

store_45_bev <- main_train %>%
  filter(store_nbr == 45, family == 'BEVERAGES') %>%
  select(date, sales)

store_47_bev <- main_train %>%
  filter(store_nbr == 47, family == 'BEVERAGES') %>%
  select(date, sales)

combined_bev_top <- bind_cols(date = store_44_bev$date, store_44 = store_44_bev$sales,
                              store_45 = store_45_bev$sales, store_47 = store_47_bev$sales)

# Produce
store_44_pr <- main_train %>%
  filter(store_nbr == 44, family == 'PRODUCE') %>%
  select(date, sales)

store_45_pr <- main_train %>%
  filter(store_nbr == 45, family == 'PRODUCE') %>%
  select(date, sales)

store_47_pr <- main_train %>%
  filter(store_nbr == 47, family == 'PRODUCE') %>%
  select(date, sales)

combined_pr_top <- bind_cols(date = store_44_pr$date, store_44 = store_44_pr$sales,
                             store_45 = store_45_pr$sales, store_47 = store_47_pr$sales)

## Bottom stores:
# Grocery I
store_22_gr <- main_train %>%
  filter(store_nbr == 22, family == 'GROCERY I') %>%
  select(date, sales)

store_32_gr <- main_train %>%
  filter(store_nbr == 32, family == 'GROCERY I') %>%
  select(date, sales)

store_52_gr <- main_train %>%
  filter(store_nbr == 52, family == 'GROCERY I') %>%
  select(date, sales)

combined_gr_low <- bind_cols(date = store_22_gr$date, store_22 = store_22_gr$sales,
                              store_32 = store_32_gr$sales, store_52 = store_52_gr$sales)

# As the third category is not matched among all the shops, only 2 will be concidered
# for the least selling stores.

# Beverages
store_22_bev <- main_train %>%
  filter(store_nbr == 22, family == 'BEVERAGES') %>%
  select(date, sales)

store_32_bev <- main_train %>%
  filter(store_nbr == 32, family == 'BEVERAGES') %>%
  select(date, sales)

store_52_bev <- main_train %>%
  filter(store_nbr == 52, family == 'BEVERAGES') %>%
  select(date, sales)

combined_bev_low <- bind_cols(date = store_22_bev$date, store_22 = store_22_bev$sales,
                              store_32 = store_32_bev$sales, store_52 = store_52_bev$sales)


### Graphs

## Graphs for top stores
# Graphs for top groceries 
compare_chart <- ggplot(data=combined_gr_top) +
  geom_line(aes(x=date, y=store_44), color="blue") +
  geom_line(aes(x=date, y=store_45), color="green") +
  geom_line(aes(x=date, y=store_47), color="red")
print(compare_chart)


# Graphs for top beverages
compare_chart2 <- ggplot(data=combined_bev_top) +
  geom_line(aes(x=date, y=store_44), color="blue") +
  geom_line(aes(x=date, y=store_45), color="green") +
  geom_line(aes(x=date, y=store_47), color="red")
print(compare_chart2)


# Graphs for top produce
compare_chart3 <- ggplot(data=combined_pr_top) +
  geom_line(aes(x=date, y=store_44), color="blue") +
  geom_line(aes(x=date, y=store_45), color="green") +
  geom_line(aes(x=date, y=store_47), color="red")
print(compare_chart3)


## Graphs for bottom stores

# For groceries
compare_chart4 <- ggplot(data=combined_gr_low) +
  geom_line(aes(x=date, y=store_22), color="blue") +
  geom_line(aes(x=date, y=store_32), color="green") +
  geom_line(aes(x=date, y=store_52), color="red")
print(compare_chart4)

# For beverages
compare_chart5 <- ggplot(data=combined_bev_low) +
  geom_line(aes(x=date, y=store_22), color="blue") +
  geom_line(aes(x=date, y=store_32), color="green") +
  geom_line(aes(x=date, y=store_52), color="red")
print(compare_chart5)

# As we can see from the data for the lowest selling stores, some of them started to sell goods
# later than others. Therefore, them being in the market for fewer time is the main reason for
# lower sales compared to other stores. Therefore, store 52 will be excluded for further analysis 
# of low selling shops.



### Checking an ADF test for non-stationarity of the data
## Top stores
# Groceries
adf.test(combined_gr_top$store_44)
adf.test(combined_gr_top$store_45)
adf.test(combined_gr_top$store_47)

# Beverages
adf.test(combined_bev_top$store_44)
adf.test(combined_bev_top$store_45)
adf.test(combined_bev_top$store_47)

# Produce
adf.test(combined_pr_top$store_44)
adf.test(combined_pr_top$store_45)
adf.test(combined_pr_top$store_47)

#p-value is not higher than 0.05 for top stores so all data is stationary

## Lowest stores
# Groceries
adf.test(combined_gr_low$store_22)
adf.test(combined_gr_low$store_32)
#adf.test(combined_gr_low$store_52)

# p-value is not higher than 0.05 for stores 22 and 32 so their data is stationary
# for shop 52 p-value is higher than 0.05, therefore data is non-stationary

# Beverages
adf.test(combined_bev_low$store_22)
adf.test(combined_bev_low$store_32)
#adf.test(combined_bev_low$store_52)

# p-value is not higher than 0.05 for stores 22 and 32 so their data is stationary
# for shop 52 p-value is higher than 0.05, therefore data is non-stationary


### Decomposition of the non-stationary data
## Top stores
# Groceries
gr_top_44 <- ts(combined_gr_top[,c("date", "store_44")], frequency = 365, start=c(2013, 1, 1))
dec1 <- decompose(gr_top_44)
plot(dec1)

gr_top_45 <- ts(combined_gr_top[,c("date", "store_45")], frequency = 365, start=c(2013, 1, 1))
dec2 <- decompose(gr_top_45)
plot(dec2)

gr_top_47 <- ts(combined_gr_top[,c("date", "store_47")], frequency = 365, start=c(2013, 1, 1))
dec3 <- decompose(gr_top_47)
plot(dec3)


# Beverages
bev_top_44 <- ts(combined_bev_top[,c("date", "store_44")], frequency = 365, start=c(2013, 1, 1))
dec4 <- decompose(bev_top_44)
plot(dec4)

bev_top_45 <- ts(combined_bev_top[,c("date", "store_45")], frequency = 365, start=c(2013, 1, 1))
dec5 <- decompose(bev_top_45)
plot(dec5)

bev_top_47 <- ts(combined_bev_top[,c("date", "store_47")], frequency = 365, start=c(2013, 1, 1))
dec6 <- decompose(bev_top_47)
plot(dec6)


# Produce
pr_top_44 <- ts(combined_pr_top[,c("date", "store_44")], frequency = 365, start=c(2013, 1, 1))
dec7 <- decompose(pr_top_44)
plot(dec7)

pr_top_45 <- ts(combined_pr_top[,c("date", "store_45")], frequency = 365, start=c(2013, 1, 1))
dec8 <- decompose(pr_top_45)
plot(dec8)

pr_top_47 <- ts(combined_pr_top[,c("date", "store_47")], frequency = 365, start=c(2013, 1, 1))
dec9 <- decompose(pr_top_47)
plot(dec9)


## Low stores
# Groceries
gr_low_22 <- ts(combined_gr_low[,c("date", "store_22")], frequency = 365, start=c(2013, 1, 1))
dec10 <- decompose(gr_low_22)
plot(dec10)

gr_low_32 <- ts(combined_gr_low[,c("date", "store_32")], frequency = 365, start=c(2013, 1, 1))
dec11 <- decompose(gr_low_32)
plot(dec11)

#gr_low_52 <- ts(combined_gr_low[,c("date", "store_52")], frequency = 365, start=c(2013, 1, 1))
#dec12 <- decompose(gr_low_52)
#plot(dec12)

# Beverages
bev_low_22 <- ts(combined_bev_low[,c("date", "store_22")], frequency = 365, start=c(2013, 1, 1))
dec13 <- decompose(bev_low_22)
plot(dec13)

bev_low_32 <- ts(combined_bev_low[,c("date", "store_32")], frequency = 365, start=c(2013, 1, 1))
dec14 <- decompose(bev_low_32)
plot(dec14)

#bev_low_52 <- ts(combined_bev_low[,c("date", "store_52")], frequency = 365, start=c(2013, 1, 1))
#dec15 <- decompose(bev_low_52)
#plot(dec15)



### Analysis of ACF - show the moving average (MA) component
## Top shops
# Groceries
acf(combined_gr_top$store_44) # 2 step
acf(combined_gr_top$store_45) # 3 step
acf(combined_gr_top$store_47) # 2 step

# Beverages
acf(combined_bev_top$store_44) # 30+ step
acf(combined_bev_top$store_45) # 30+ step
acf(combined_bev_top$store_47) # 30+ step

# Produce
acf(combined_pr_top$store_44) # 30+
acf(combined_pr_top$store_45) # 30+
acf(combined_pr_top$store_47) # 30+

## Low shops
# Groceries
acf(combined_gr_low$store_22) # 30+
acf(combined_gr_low$store_32) # 30+
acf(combined_gr_low$store_52) # 30+

# Beverages
acf(combined_bev_low$store_22) # 30+
acf(combined_bev_low$store_32) # 30+
acf(combined_bev_low$store_52) # 30+


### Analysis of pACF - show the lag - autocorrelation - AR
## Top shops
# Groceries
pacf(combined_gr_top$store_44) # 3 step
pacf(combined_gr_top$store_45) # 14 step
pacf(combined_gr_top$store_47) # 9 step

# Beverages
pacf(combined_bev_top$store_44) # 3 step
pacf(combined_bev_top$store_45) # 10 step
pacf(combined_bev_top$store_47) # 10 step

# Produce
pacf(combined_pr_top$store_44) # 5 step
pacf(combined_pr_top$store_45) # 8 step
pacf(combined_pr_top$store_47) # 11 step


## Low shops
# Groceries
pacf(combined_gr_low$store_22) # 7 step
pacf(combined_gr_low$store_32) # 6 step
pacf(combined_gr_low$store_52) # 9 step

# Beverages
pacf(combined_bev_low$store_22) # 9 step
pacf(combined_bev_low$store_32) # 9 step 
pacf(combined_bev_low$store_52) # 9 step


#### FORECASTING MODELS ####
# For simplification and taking into account that low stores started their activity much later we cannot
# accurately compare them to top stores. Therefore, they will be excluded from further analysis.
# GARCH model was chosen as the best to forecast variables

### GARCH
### Top shops

## Groceries
# Shop 44
model_param <- ugarchspec(mean.model=list(armaOrder=c(30,30)), 
                          variance.model= list(model="sGARCH", garchOrder=c(1,1)),
                          distribution.model="norm") #you can change the distribution model to t-student 
garch_model <- ugarchfit(data=combined_gr_top$store_44,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)
#sum(predicted_data$predicted_value)/sum()

store_44_gr <- store_44_gr %>%
  rename(predicted_value = sales)


combined_data_gr_44 <- rbind(store_44_gr,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_gr_44, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_44_gr) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 44, Grocery I)") +
  theme_bw()



# Shop 45
model_param <- ugarchspec(mean.model=list(armaOrder=c(30,30)), 
                          variance.model= list(model="sGARCH", garchOrder=c(1,1)),
                          distribution.model="norm") #you can change the distribution model to t-student 
garch_model <- ugarchfit(data=combined_gr_top$store_45,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)

store_45_gr <- store_45_gr %>%
  rename(predicted_value = sales)


combined_data_gr_45 <- rbind(store_45_gr,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_gr_45, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_45_gr) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 45, Grocery I)") +
  theme_bw()



# Shop 47
model_param <- ugarchspec(mean.model=list(armaOrder=c(30,30)), 
                          variance.model= list(model="sGARCH", garchOrder=c(1,1)),
                          distribution.model="norm") #you can change the distribution model to t-student 
garch_model <- ugarchfit(data=combined_gr_top$store_47,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)

store_47_gr <- store_47_gr %>%
  rename(predicted_value = sales)


combined_data_gr_47 <- rbind(store_47_gr,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_gr_47, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_47_gr) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 47, Grocery I)") +
  theme_bw()






# Beverages

# Shop 44
model_param <- ugarchspec(mean.model=list(armaOrder=c(30,40)), 
                          variance.model= list(model="sGARCH", garchOrder=c(10,10)),
                          distribution.model="norm")
garch_model <- ugarchfit(data=combined_bev_top$store_44,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)

store_44_bev <- store_44_bev %>%
  rename(predicted_value = sales)


combined_data_bev_44 <- rbind(store_44_bev,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_bev_44, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_44_bev) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 44, Beverages)") +
  theme_bw()



# Shop 45
model_param <- ugarchspec(mean.model=list(armaOrder=c(30,30)), 
                          variance.model= list(model="sGARCH", garchOrder=c(1,1)),
                          distribution.model="norm") #you can change the distribution model to t-student 
garch_model <- ugarchfit(data=combined_bev_top$store_45,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)

store_45_bev <- store_45_bev %>%
  rename(predicted_value = sales)


combined_data_bev_45 <- rbind(store_45_bev,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_bev_45, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_45_bev) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 45, Beverages)") +
  theme_bw()



# Shop 47
model_param <- ugarchspec(mean.model=list(armaOrder=c(25,30)), 
                          variance.model= list(model="sGARCH", garchOrder=c(1,1)),
                          distribution.model="norm") #you can change the distribution model to t-student 
garch_model <- ugarchfit(data=combined_bev_top$store_47,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)

store_47_bev <- store_47_bev %>%
  rename(predicted_value = sales)


combined_data_bev_47 <- rbind(store_47_bev,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_bev_47, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_47_bev) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 47, Beverages)") +
  theme_bw()



############
### Produce

# Shop 44
model_param <- ugarchspec(mean.model=list(armaOrder=c(30,30)), 
                          variance.model= list(model="sGARCH", garchOrder=c(10,10)),
                          distribution.model="norm")
garch_model <- ugarchfit(data=combined_pr_top$store_44,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)

store_44_pr <- store_44_pr %>%
  rename(predicted_value = sales)


combined_data_pr_44 <- rbind(store_44_pr,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_pr_44, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_44_pr) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 44, Produce)") +
  theme_bw()



# Shop 45
model_param <- ugarchspec(mean.model=list(armaOrder=c(30,30)), 
                          variance.model= list(model="sGARCH", garchOrder=c(1,1)),
                          distribution.model="norm") #you can change the distribution model to t-student 
garch_model <- ugarchfit(data=combined_pr_top$store_45,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)

store_45_pr <- store_45_pr %>%
  rename(predicted_value = sales)


combined_data_pr_45 <- rbind(store_45_pr,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_pr_45, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_45_pr) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 45, Produce)") +
  theme_bw()



# Shop 47
model_param <- ugarchspec(mean.model=list(armaOrder=c(30,30)), 
                          variance.model= list(model="sGARCH", garchOrder=c(1,1)),
                          distribution.model="norm") #you can change the distribution model to t-student 
garch_model <- ugarchfit(data=combined_pr_top$store_47,
                         spec=model_param, out.sample = 20)

print(garch_model)


## Forecasting a GARCH model through Bootstraping
bootstrap <- ugarchboot(garch_model, method = c("Partial", "Full")[1],
                        n.ahead = 90, n.bootpred = 500)
print(bootstrap)

predictions <- tail(bootstrap@forc@forecast$seriesFor, n = 90)

predicted_data <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
                             predicted_value = predictions)

predicted_data <- predicted_data %>%
  rename(predicted_value = X1974.07.22.20.00.00)

sum(predicted_data$predicted_value)

store_47_pr <- store_47_pr %>%
  rename(predicted_value = sales)


combined_data_pr_47 <- rbind(store_47_pr,predicted_data)

# Reset index
#predicted_data <- predicted_data %>%
#  rownames_to_column(var = "index")

ggplot(combined_data_pr_47, aes(x = date, y = predicted_value)) +
  geom_line(color = "blue", data = store_47_pr) +
  geom_line(color = "red", data = predicted_data) +
  labs(x = "Date", y = "Sales", title = "Historical Sales vs. Predicted Values (Store 47, Produce)") +
  theme_bw()











#### Appendix ####
#ARIMA shop 44
#gr_44_arma <- arma(combined_gr_top$store_44, order=c(3,0))
#summary(gr_44_arma)

#gr_44_arima <- arima(combined_gr_top$store_44, 
#                     order=c(5,0,7)) 
#predictions_44 <- predict(gr_44_arima, n.ahead =90) 

#predicted_data_44_gr <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
#                                   predicted_value = predictions_44$pred)
#ggplot() +
#  geom_line(data = combined_gr_top, aes(x = date, y = store_44), color = "blue", linetype = "solid") +
#  geom_line(data = predicted_data_44_gr, aes(x = date, y = predicted_value), color = "red", linetype = "solid") +
#  labs(x = "Date", y = "Sales", title = "ARIMA Model Predictions for Store 44") +
#  theme_bw()


# Shop 45
#gr_45_arma <- arma(combined_gr_top$store_45, order=c(3,0))
#summary(gr_45_arma)

#gr_45_arima <- arima(combined_gr_top$store_45, 
#                     order=c(3,0,7)) 
#predictions <- predict(gr_45_arima, n.ahead =90) 

#predicted_data_45_gr <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
#                                   predicted_value = predictions$pred)
#ggplot() +
#  geom_line(data = combined_gr_top, aes(x = date, y = store_45), color = "blue", linetype = "solid") +
#  geom_line(data = predicted_data_45_gr, aes(x = date, y = predicted_value), color = "red", linetype = "solid") +
#  labs(x = "Date", y = "Sales", title = "ARIMA Model Predictions for Store 44") +
#  theme_bw()


# Shop 47
#gr_47_arma <- arma(combined_gr_top$store_47, order=c(9,0))
#summary(gr_47_arma)

#gr_47_arima <- arima(combined_gr_top$store_47, 
#                     order=c(3, 0, 10)) 
#predictions <- predict(gr_47_arima, n.ahead = 90) 

#predicted_data_47_gr <- data.frame(date = seq(as.Date("2017-08-16"), by = "day", length.out = 90),
#                                   predicted_value = predictions$pred)
#ggplot() +
#  geom_line(data = combined_gr_top, aes(x = date, y = store_47), color = "blue", linetype = "solid") +
#  geom_line(data = predicted_data_47_gr, aes(x = date, y = predicted_value), color = "red", linetype = "solid") +
#  labs(x = "Date", y = "Sales", title = "ARIMA Model Predictions for Store 44") +
#  theme_bw()