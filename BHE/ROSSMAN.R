library(lubridate)
library(data.table) 
library(ggplot2)
library(h2o)
library(dplyr)


 # Section A: Pre-processing #
df_test <- read.csv("BHE-Test.csv", stringsAsFactors = FALSE)
df_train <- read.csv("BHE-Train.csv", stringsAsFactors = FALSE)
df_store <- read.csv("BHE-Stores.csv", stringsAsFactors = FALSE)
length(df_test$Store)
length(df_train$Store)
length(df_store$Store)
summary(df_store)
summary(df_train)
summary(df_test)
# add average sales to df_store
sales_by_store <- aggregate(df_train$Sales, by = list(df_train$Store), mean)
names(sales_by_store) <- c("Store", "Average.Sales")
df_store <- merge(df_store, sales_by_store, by=c('Store'))
head(df_store)

df_train <- subset(df_train, select = -c(Customers))

# processinging df_store
# impute Competition Values
df_store$CompetitionOpenSinceYear[is.na(df_store$CompetitionOpenSinceYear)] <- 1990 # Dealing with NA and outlayers
df_store$CompetitionOpenSinceMonth[is.na(df_store$CompetitionOpenSinceMonth)] <- 1 # Dealing with NA
df_store$LogCompetitionDistance <- log1p(df_store$CompetitionDistance)
df_store$StoreType <- as.factor(df_store$StoreType)
df_store$Assortment <- as.factor(df_store$Assortment)
df_store$CompetitionOpenSinceMonth <- as.factor(df_store$CompetitionOpenSinceMonth)
df_store$CompetitionOpenSinceYear <- as.factor(df_store$CompetitionOpenSinceYear)

df_store <- df_store %>% select(Store,StoreType,Assortment,CompetitionDistance,CompetitionOpenSinceMonth,CompetitionOpenSinceYear,Average.Sales,LogCompetitionDistance)

head(df_store)
str(df_store)


processing <- function(df_train) {
  # Merge df_store into df_train
  train <- merge(df_train, df_store, by=c('Store'))
  
  # add year, mth, date, day of week features
  train$Store <- as.factor(train$Store)
  
  train$Date <- as.Date(train$Date)
  train$day <- as.factor(format(train$Date, "%d"))
  train$month <- as.factor(format(train$Date, "%m"))
  train$year <- as.factor(format(train$Date, "%Y"))
  train <- subset(train, select = -c(Date))
  
  train$DayOfWeek <- as.factor(train$DayOfWeek)
  
  # Factorize
  train$Open <- as.factor(train$Open)
  train$Promo <- as.factor(train$Promo)
  train$StateHoliday <- as.factor(train$StateHoliday)
  train$SchoolHoliday <- as.factor(train$SchoolHoliday)
  return(train)
}

train = processing(df_train)
test = processing(df_test)

# Section B: EDA #
# Useful Graphs
# Sales distribution from (10,000 - 35000)
ggplot(train) + aes(Sales) + geom_histogram(binwidth = 500, fill = "blue", alpha = 0.5) + xlim(10000,35000)
ggplot(train) + aes(Promo, fill = Promo) + geom_bar() # Plot how promo affects sales
ggplot(train) + aes(Sales, fill = Promo) + geom_histogram(binwidth = 500, position = "identity", alpha = 0.5)+ xlim(10000,35000) # Distribution of Sales with Promo
ggplot(train) + aes(StoreType, Sales, fill = Assortment) + geom_boxplot(outlier.size = 1, outlier.colour = "red")
ggplot(train) + aes(StateHoliday, Sales, fill = StateHoliday) + geom_boxplot(outlier.size = 1, outlier.colour = "blue") # Sales Variation with State Holiday
ggplot(train) + aes(SchoolHoliday, Sales, fill = SchoolHoliday) + geom_boxplot( outlier.size = 1, outlier.colour = "blue")
ggplot(train) + aes(as.factor(DayOfWeek), Sales, fill = DayOfWeek) + geom_boxplot( outlier.size = 1, outlier.colour = "blue")
ggplot(train) + aes(as.factor(month), Sales, fill = as.factor(month)) + geom_boxplot( outlier.size = 1, outlier.colour = "blue")

#Section C: Utility Functions #
# rmspe
compute_rmspe <- function(predicted, expected) {
  predicted = predicted[expected != 0]
  expected = expected[expected != 0]
  mean(((predicted - expected) / expected)^2)
}

# TEST Output CSV (predicted using test)

output_to_TEST <- function(predicted) {
  write.csv(data.frame(Id=test$Id, Predicted.Sales=predicted), "Test.csv", sep = " ", row.names=F)
}

#Section D: Cross Validation #
k = 5 #Folds
set.seed(123)
# sample from 1 to k, nrow times (the number of observations in the data)
id <- sample(1:k, nrow(train), replace = TRUE)
list <- 1:k
trainingset <- subset(train, id %in% list[-1])
validationset <- subset(train, id %in% c(1))

#Section E: Benchmark #
# Predicting using the average sales per store
predict_bench = validationset$Average.Sales
rmspe_bench = compute_rmspe(predict_bench, validationset$Sales) 
rmspe_bench # 0.04365293

#Section F: Linear Model #
# Fit Sales against all variables
set.seed(123)
lm_all = lm(Sales ~ . - Sales - Store, data = trainingset)
predict_all = predict(lm_all, newdata = validationset)
rmspe_all = compute_rmspe(predict_all, validationset$Sales)  # 0.0323
output_to_TEST(predict(lm_all, newdata = test)) 
summary(lm_all)

#Section G: Variable Selection-Backward Elimination #
# after backward elimination
lm_backward_elimination <- lm(formula = "Sales~DayOfWeek+Open+Promo+StateHoliday+SchoolHoliday+StoreType+day+month+year", data = trainingset)
predict_backward_elimination = predict(lm_backward_elimination, newdata = validationset)
rmspe_backward_elimination = compute_rmspe(predict_backward_elimination, validationset$Sales)  # 0.0298
output_to_TEST(predict(lm_backward_elimination, newdata = test)) 
summary(lm_backward_elimination)
rmspe_backward_elimination
########################Section I: Random Forest ###############################
trainingset$logSales <- log1p(trainingset$Sales)
# Use H2O's random forest
# start cluster with all available threads
h2o.init(nthreads=-1,max_mem_size='6G')
# Load data into cluster from R
trainH2O<-as.h2o(trainingset)
## Set up variable to use all features other than those specified here
features<-colnames(trainingset)[!(colnames(trainingset) %in% c("Sales","logSales", "CompetitionDistance"))]

# optimize hyperparamter max_depth using validation set
rmspes = c()
depths = 1:30
for (depth in depths) {
  # Train a random forest using all default parameters
  rf_model <- h2o.randomForest(x=features,
                               y="logSales", 
                               ntrees = 100,
                               max_depth = depth,
                               nbins_cats = 1115, ## no of bins in hist
                               training_frame=trainH2O)
  
  #summary(rf_model)
  #h2o.varimp(rf_model)
  #sending the validaionset data in predict_rf
  predict_rf <- function(dataset) {
    # Load test data into cluster from R
    testH2O<-as.h2o(dataset)
    # Get predictions out; predicts in H2O, as.data.frame gets them into R
    predictions<-as.data.frame(h2o.predict(rf_model,testH2O))
    # Return the predictions to the original scale of the Sales data
    pred <- expm1(predictions[,1])
    return (pred)
  }
  predicted_rf = predict_rf(validationset)
  rmspe_rf = compute_rmspe(predicted_rf, validationset$Sales) # 0.00968 (the predicted value with validationset$Sales)
  rmspes = c(rmspes, rmspe_rf) # merging new predicted value with old one
}
h2o.varimp(rf_model)
h2o.varimp_plot(rf_model, num_of_features = NULL)
rmspes
#[1] 0.185982446 0.071531081 0.033383766 0.025505482 0.022903244 0.019826300 0.016548793 0.013951161 0.011738143
#[10] 0.010858457 0.010413946 0.010896515 0.010705461 0.010285415 0.010026126 0.010257577 0.009440238 0.010354093
#[19] 0.009495374 0.009448067 0.010339571 0.010259949 0.009505038 0.009649905 0.009452037 0.009747790 0.009955891
#[28] 0.009992118 0.009680090 0.009797381
plot(depths, rmspes, main="RMSPES for Random Forest Per Depth (with ntrees=100)")

# optimize hyperparamter ntrees using validation set
rmspes = c()
ntreesParams = (1:20)*10
for (ntreesParam in ntreesParams) {
  ## Train a random forest using all default parameters
  rf_model <- h2o.randomForest(x=features,
                               y="logSales", 
                               ntrees = ntreesParam,
                               max_depth = 30,
                               nbins_cats = 1115, ## allow it to fit store ID
                               training_frame=trainH2O)
  
  #summary(rf_model)
  predict_rf <- function(dataset) {
    # Load test data into cluster from R
    testH2O<-as.h2o(dataset)
    # Get predictions out; predicts in H2O, as.data.frame gets them into R
    predictions<-as.data.frame(h2o.predict(rf_model,testH2O))
    # Return the predictions to the original scale of the Sales data
    pred <- expm1(predictions[,1])
    return (pred)
  }
  predicted_rf = predict_rf(validationset)
  rmspe_rf = compute_rmspe(predicted_rf, validationset$Sales)
  rmspes = c(rmspes, rmspe_rf)
}
rmspes
#[1] 0.010468950 0.011053861 0.010188063 0.010576120 0.009496403 0.009470264 0.010228179
#[8] 0.009427444 0.009948769 0.009783110 0.009558485 0.009041274 0.010386415 0.009736435
#[15] 0.009785170 0.009723392 0.009870335 0.010211466 0.009344798 0.009583338
plot(ntreesParams, rmspes, main="RMSPES for Random Forest Per ntrees\n(with max_depth=30)")


# Restore trainingset to how it was before
trainingset <- subset(trainingset, select = -c(logSales))

# validation error: 0.010016293 (depth=30, ntrees=100)
write.csv(data.frame(Id=test$Id, Predicted.Sales=predict_rf(test)), "Test.csv", row.names=F)




