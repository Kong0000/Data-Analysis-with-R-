###############
# Project Notes
###############

# Problem
# The sales team of Blackwell Electronics has some concerns about ongoing product sales in one of their stores. They have been tracking the sales performance of specific product types and would like the data science team to do a product profitability prediction analysis. This will help the sales team better understand how types of products might impact sales across the enterprise.

# Goals
#  .	Predict the sales in four different product types.
#  .	Assess the impact service and customer reviews have on sales.

# Dataset Information
#  .	Existing product file contains sales performance of 80 products. This file will be used for analyzing and training.
#  .	New product file contains 24 new products that we will make the sales volume and profitability prediction on.
#  .	Product sales performance include:
#      o	Product type and number
#      o	1-5 star reviews
#      o	Positive and negative reviews
#      o	If the product was recommended and bestseller rank
#      o	Product dimensions
#      o	Product profit margin and sales volume



################
# Load packages
################

install.packages("caret", dep=TRUE)
install.packages("ggplot2")
install.packages("corrplot")
install.packages('caTools')
install.packages("readr")
install.packages("mlbench")
install.packages("doParallel")  # for Win parallel processing 
library(ggplot2)
library(caret)
library(corrplot)
library(caTools)
library(readr)
library(mlbench)
library(doParallel)             # for Win parallel processing


##############
# Import data 
##############

# import csv data
oEx <- read.csv("existingproductattributes2017.csv", stringsAsFactors = FALSE)
str(oEx)

oNew <- read.csv("newproductattributes2017.csv", stringsAsFactors = FALSE)
str(oNew)


################
# Evaluate data
################

# check missing or duplicate values
anyNA(oEx)
oEx[!complete.cases(oEx),] # BestSellerRank
anyDuplicated(oEx)

anyNA(oNew)
oNew[!complete.cases(oNew),] # BestSellerRank
anyDuplicated(oNew)

# delete features which have missing values
oEx$BestSellersRank <- NULL
str(oEx)

oNew$BestSellersRank <- NULL
str(oNew)


#############
# Preprocess
#############

# dummify categorical features
dummy <- dummyVars(~., data = oEx)
oEx_new <- data.frame(predict(dummy, newdata = oEx))
str(oEx_new) # 80 obs. of 28 variables

dummy1 <- dummyVars(~., data = oNew)
oNew_new <- data.frame(predict(dummy1, newdata = oNew))
str(oNew_new)

# check correlation and remove features with highest correlation
corr_matrix <- cor(oEx_new[, 1: 28])
h_corr <- findCorrelation(corr_matrix, cutoff = 0.8)
h_corr
colnames(oEx_new[c(15)]) #"x5StarReviews"
colnames(oEx_new[c(17)]) #"x3StarReviews"
colnames(oEx_new[c(18)]) #"x2StarReviews"
colnames(oEx_new[c(16)]) #"x4StarReviews"
colnames(oEx_new[c(21)]) #"NegativeServiceReview"
colnames(oEx_new[c(3)]) #"ProductTypeExtendedWarranty"
#Clean Training Set
oEx_cleaned <- oEx_new
oEx_cleaned$x5StarReviews <- NULL
oEx_cleaned$x3StarReviews <- NULL
oEx_cleaned$x2StarReviews <- NULL
oEx_cleaned$x4StarReviews <- NULL
oEx_cleaned$NegativeServiceReview <- NULL
oEx_cleaned$ProductTypeExtendedWarranty <- NULL
str(oEx_cleaned) # 80 obs. of 22 variables


##################
# Train/test sets
##################

# creating train and test datasets
set.seed(502)
split_data1 = sample(2, nrow(oEx_new), replace = T, prob = c(.7, .3))
oEx_new_train <- oEx_new[split_data1 == 1,]
oEx_new_test <- oEx_new[split_data1 == 2,]

set.seed(502)
split_data2 = sample(2, nrow(oEx_cleaned), replace = T, prob = c(.7, .3))
oEx_cleaned_train <- oEx_cleaned[split_data2 == 1,]
oEx_cleaned_test <- oEx_cleaned[split_data2 == 2,]


################
# Train control
################

# since dataset doesn't have enough observations, we use k-fold cross validation
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)



###############
# Train models
###############

# svm
svm_1 <- train(Volume ~., data = oEx_cleaned_train, method = "svmRadial", trControl = train_control, preProcess = c("center"))
svm_1

# Support Vector Machines with Radial Basis Function Kernel 
# 61 samples
# 21 predictors
# Pre-processing: centered (21) 
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 55, 55, 56, 54, 55, 54, ... 
# Resampling results across tuning parameters:
#   C     RMSE      Rsquared   MAE     
# 0.25  1297.726  0.2896879  759.8402
# 0.50  1297.043  0.2896879  759.4556
# 1.00  1295.680  0.2896879  758.6865
# Tuning parameter 'sigma' was held constant at a value of 0.0001258594
# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were sigma = 0.0001258594 and C = 1.

# results are not good enough, try to auto-tune the model
svm_2 <- train(Volume ~., data = oEx_cleaned_train, method = "svmRadial", trControl = train_control, preProcess = c("center"), tuneLength = 10)
svm_2
# Support Vector Machines with Radial Basis Function Kernel 
# 61 samples
# 21 predictors
# Pre-processing: centered (21) 
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 55, 54, 55, 56, 55, 54, ... 
# Resampling results across tuning parameters:
#   C       RMSE      Rsquared   MAE     
# 0.25  1308.812  0.3220466  760.0128
# 0.50  1308.090  0.3220466  759.6244
# 1.00  1306.648  0.3220466  758.8475
# 2.00  1303.776  0.3220466  757.2936
# 4.00  1298.071  0.3220466  754.1860
# 8.00  1286.859  0.3220466  747.8225
# 16.00  1265.394  0.3220466  735.2459
# 32.00  1238.275  0.3330054  713.2529
# 64.00  1199.175  0.3608002  679.4109
# 128.00  1147.495  0.4107993  637.9846
# Tuning parameter 'sigma' was held constant at a value of 0.0001111544
# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were sigma = 0.0001111544 and C = 128.

svm_2$bestTune
# sigma   C
# 10 0.0001111544 128

# both results are not good, then changing the dataset without cleanup
svm_3 <- train(Volume ~., data = oEx_new_train, method = "svmRadial", trControl = train_control, preProcess = c("center"))
svm_3

# Support Vector Machines with Radial Basis Function Kernel 
# 
# 61 samples
# 27 predictors
# 
# Pre-processing: centered (27) 
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 55, 56, 55, 55, 54, 55, ... 
# Resampling results across tuning parameters:
#   
#   C     RMSE      Rsquared   MAE     
# 0.25  1306.070  0.6582805  757.8316
# 0.50  1304.033  0.6582805  756.4155
# 1.00  1299.963  0.6582805  753.5835
# 
# Tuning parameter 'sigma' was held constant at a value of 1.63909e-05
# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were sigma = 1.63909e-05 and C = 1.

svm_4 <- train(Volume ~., data = oEx_new_train, method = "svmRadial", trControl = train_control, preProcess = c("center"), tuneLength = 10)
svm_4

# Support Vector Machines with Radial Basis Function Kernel 
# 
# 61 samples
# 27 predictors
# 
# Pre-processing: centered (27) 
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 54, 56, 55, 56, 55, 55, ... 
# Resampling results across tuning parameters:
#   
#   C       RMSE      Rsquared   MAE     
# 0.25  1333.982  0.5200549  778.2137
# 0.50  1332.702  0.5200549  777.3067
# 1.00  1330.147  0.5200549  775.4928
# 2.00  1325.052  0.5200549  771.8651
# 4.00  1314.931  0.5200549  764.6095
# 8.00  1294.942  0.5200549  750.0870
# 16.00  1257.052  0.5209621  720.7624
# 32.00  1209.742  0.5384743  672.6811
# 64.00  1133.196  0.5701446  609.2821
# 128.00  1046.030  0.5936868  516.8558
# 
# Tuning parameter 'sigma' was held constant at a value of 8.701488e-05
# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were sigma = 8.701488e-05 and C = 128.

# using the best model to predict
svm_test_pred <- predict(svm_4, newdata = oEx_new_test)
# RMSE and R Square
postResample(svm_test_pred, oEx_new_test$Volume)

# RMSE    Rsquared         MAE 
# 431.0752024   0.3809941 255.0646055 



#rf
rf_1 <- train(Volume ~., data = oEx_cleaned_train, method = "rf", trControl = train_control)
rf_1
# Random Forest 
# 
# 61 samples
# 21 predictors
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 54, 55, 55, 55, 55, 55, ... 
# Resampling results across tuning parameters:
#   
#   mtry  RMSE      Rsquared   MAE     
# 2    949.7524  0.6652122  586.1563
# 11    926.4036  0.8215465  460.0870
# 21    957.8996  0.8294684  455.4349
# 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 11.

rf_2 <- train(Volume ~., data = oEx_new_train, method = "rf", trControl = train_control)
rf_2
# Random Forest 
# 
# 61 samples
# 27 predictors
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 55, 55, 55, 54, 54, 54, ... 
# Resampling results across tuning parameters:
#   
#   mtry  RMSE      Rsquared   MAE     
#   2    814.0784  0.8948517  456.2205
#   14    651.1319  0.9622482  292.5617
#   27    555.8869  0.9779738  244.2778
# 
# RMSE was used to select the optimal model using the smallest value.
# The final value used for the model was mtry = 27.

# using the best model to predict
rf_test_pred <- predict(rf_2, newdata = oEx_new_test)
postResample(rf_test_pred, oEx_new_test$Volume)
# RMSE   Rsquared        MAE 
# 98.8419723  0.9672875 53.5944561

# gbm
gbm_1 <- train(Volume ~., data = oEx_cleaned_train, method = "gbm", trControl = train_control)
gbm_1

# Stochastic Gradient Boosting 
# 
# 61 samples
# 21 predictors
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 55, 54, 55, 55, 54, 56, ... 
# Resampling results across tuning parameters:
#   
#   interaction.depth  n.trees  RMSE       Rsquared   MAE     
# 1                   50       959.5007  0.7864574  599.2614
# 1                  100      1034.1824  0.7284344  658.0402
# 1                  150      1092.5704  0.7135048  712.3185
# 2                   50       966.1797  0.7832324  603.7908
# 2                  100      1025.3362  0.7389678  652.4931
# 2                  150      1080.8370  0.6871343  706.8393
# 3                   50       953.0270  0.7928220  581.8977
# 3                  100      1027.6097  0.7419838  660.4419
# 3                  150      1074.6654  0.7028233  713.2857
# 
# Tuning parameter 'shrinkage' was held constant at a value of 0.1
# Tuning parameter 'n.minobsinnode' was held constant at a value of 10
# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were n.trees = 50, interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.

gbm_2 <- train(Volume ~., data = oEx_new_train, method = "gbm", trControl = train_control)
gbm_2

# Stochastic Gradient Boosting 
# 
# 61 samples
# 27 predictors
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 54, 54, 54, 56, 55, 55, ... 
# Resampling results across tuning parameters:
#   
#   interaction.depth  n.trees  RMSE       Rsquared   MAE     
# 1                   50       916.6727  0.8418274  548.4059
# 1                  100       992.4464  0.7907232  618.2123
# 1                  150      1038.6859  0.7577176  657.9111
# 2                   50       915.4734  0.8305785  560.6369
# 2                  100       986.3015  0.7870500  630.6769
# 2                  150      1030.3598  0.7475758  669.0954
# 3                   50       940.5054  0.8091629  579.6015
# 3                  100       998.7545  0.7668762  637.5238
# 3                  150      1027.8779  0.7301153  663.6970
# 
# Tuning parameter 'shrinkage' was held constant at a value of 0.1
# Tuning parameter 'n.minobsinnode' was held constant at a value of 10
# RMSE was used to select the optimal model using the smallest value.
# The final values used for the model were n.trees = 50, interaction.depth = 2, shrinkage = 0.1 and n.minobsinnode = 10.

# using the best model to predict
gbm_test_pred <- predict(gbm_2, newdata = oEx_new_test)
postResample(gbm_test_pred, oEx_new_test$Volume)

# RMSE    Rsquared         MAE 
# 226.6275792   0.9189546 164.0285773




#run rfe to select the most important features for cleaned dataset
set.seed(100)

control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 5,
                      number = 10)

result_rfe1 <- rfe(oEx_cleaned[, 1:21],
                   oEx_cleaned[, 22],
                   sizes = c(1:21),
                   rfeControl = control)

predictors(result_rfe1)
#following features are import:
#[1] "PositiveServiceReview"  "x1StarReviews"          "ProductNum"           
#[4] "ProductTypeGameConsole" "ShippingWeight"         "Recommendproduct"     
#[7] "ProductWidth"           "ProductDepth"           "ProductHeight"         
#[10] "ProductTypePrinter"     "Price"


varImp(result_rfe1)
# Overall
# PositiveServiceReview      18.353922
# x1StarReviews               9.125475
# ProductTypeGameConsole      4.176588
# ProductNum                  3.795871
# ShippingWeight              2.884187
# Recommendproduct            2.658032
# ProductDepth                2.429582
# ProductTypeAccessories      2.428781
# ProductTypeTablet           2.371065
# ProductWidth                2.328994
# ProfitMargin                2.185214
# ProductHeight               2.071762
# Price                       2.056758
# ProductTypeDisplay          2.044144
# ProductTypeLaptop           1.896317
# ProductTypePrinter          1.890386
# ProductTypeSmartphone       1.639584
# ProductTypePC               1.591599
# ProductTypePrinterSupplies  1.415563
# ProductTypeSoftware         1.094535
varimp_data <- data.frame(feature = row.names(varImp(result_rfe1))[1:11],
                          importance = varImp(result_rfe1)[1:11, 1])
# 11 most important features
ggplot(data = varimp_data,
       aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
  geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") +
  geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) +
  theme_bw() + theme(legend.position = "none")

#run rfe to select the most important features for not cleaned dataset
set.seed(100)

control2 <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 5,
                      number = 10)

result_rfe2 <- rfe(oEx_new[, 1:26],
                   oEx_new[, 27],
                   sizes = c(1:26),
                   rfeControl = control2)

predictors(result_rfe2)
# [1] "ProductTypeAccessories"      "ProductDepth"                "ProductHeight"               "ProductWidth"               
# [5] "Price"                       "ProductTypeExtendedWarranty" "ProductTypePrinterSupplies"  "ProductNum"                 
# [9] "ShippingWeight"              "ProductTypeSoftware"



postResample(predict(result_rfe1, oEx_cleaned), oEx_cleaned$Volume)
# RMSE    Rsquared         MAE 
# 609.3985171   0.8939873 183.0878279 



# # following is overfitted, since the most important feature is only x5start
# plot(result_rfe1, type=c("g", "o"))
# 
# result_rfe <- rfe(oEx_new[, 1:27],
#                   oEx_new[, 28],
#                   sizes = c(1:27),
#                   rfeControl = control)
# 
# predictors(result_rfe)
# # [1] "x5StarReviews"
# plot(result_rfe, type=c("g", "o"))
# str(oEx_new)


##################
# Model selection
##################

# now using the best model to do the prediction
# svm
svm_optimal <- train(Volume ~ PositiveServiceReview*x1StarReviews*ProductNum*ProductTypeGameConsole*ShippingWeight*Recommendproduct*ProductWidth*ProductDepth*ProductHeight*ProductTypePrinter*Price, data = oEx_cleaned_train, method = "svmRadial", trControl = train_control, preProcess = c("center"))

# using the best model to predict
svm_optimal_pred <- predict(svm_optimal, newdata = oEx_cleaned_test)
# RMSE and R Square
postResample(svm_optimal_pred, oEx_cleaned_test$Volume)
# RMSE     Rsquared          MAE 
# 573.17756340   0.04898625 339.20361663 

#rf
rf_optimal <- train(Volume ~ PositiveServiceReview*x1StarReviews*ProductNum*ProductTypeGameConsole*ShippingWeight*Recommendproduct*ProductWidth*ProductDepth*ProductHeight*ProductTypePrinter*Price, data = oEx_cleaned_train, method = "rf", trControl = train_control)

# using the best model to predict
rf_optimal_pred <- predict(rf_optimal, newdata = oEx_cleaned_test)
postResample(rf_optimal_pred, oEx_cleaned_test$Volume)
# RMSE    Rsquared         MAE 
# 517.7960120   0.4255789 475.8688842 

# gbm
gbm_optimal <- train(Volume ~ PositiveServiceReview*x1StarReviews*ProductNum*ProductTypeGameConsole*ShippingWeight*Recommendproduct*ProductWidth*ProductDepth*ProductHeight*ProductTypePrinter*Price, data = oEx_cleaned_train, method = "gbm", trControl = train_control)

# using the best model to predict
gbm_optimal_pred <- predict(gbm_optimal, newdata = oEx_cleaned_test)
postResample(gbm_optimal_pred, oEx_cleaned_test$Volume)
# RMSE    Rsquared         MAE 
# 315.3719770   0.8894306 269.9768967 


#################
# Predict testSet
#################

# from above result, rf still gives the best prediction
# using rf_optimal to do the prediction based on new dataset
rf_optimal_pred_result <- predict(rf_optimal, newdata = oNew_new)
print(rf_optimal_pred_result)
# 1         2         3         4         5         6         7         8         9        10        11        12        13        14        15 
# 1268.0972 1194.0869  932.6561  606.3401  483.9418  645.5631 1193.6775  787.5023  482.5617 1071.5841 1727.4823  616.1942  629.4192  532.9620  599.3969 
# 16        17        18        19        20        21        22        23        24 
# 3078.9346  570.6705  518.1091  544.1174  529.4520  456.4536  485.2989  616.2359 3377.7802 

oNew_new$Volume <- rf_optimal_pred_result
str(oNew_new)

# calculate the profit based on predicted volume and draw the profit per product type
oNew_new$profit <- oNew_new$Volume * oNew_new$Price * oNew_new$ProfitMargin
str(oNew_new)
aggregate(oNew_new$profit, list(oNew_new$ProductTypeSoftware), FUN=sum) #7517.159
aggregate(oNew_new$profit, list(oNew_new$ProductTypeAccessories), FUN=sum) #1006.255
aggregate(oNew_new$profit, list(oNew_new$ProductTypeDisplay), FUN=sum) #3994.693
aggregate(oNew_new$profit, list(oNew_new$ProductTypeExtendedWarranty), FUN=sum) #24646.97
aggregate(oNew_new$profit, list(oNew_new$ProductTypeGameConsole), FUN=sum) #327673.4
aggregate(oNew_new$profit, list(oNew_new$ProductTypeLaptop), FUN=sum) # 443377.7
aggregate(oNew_new$profit, list(oNew_new$ProductTypeNetbook), FUN=sum) #108362.9
aggregate(oNew_new$profit, list(oNew_new$ProductTypePC), FUN=sum) #426982.9
aggregate(oNew_new$profit, list(oNew_new$ProductTypePrinter), FUN=sum) #82157.54
aggregate(oNew_new$profit, list(oNew_new$ProductTypePrinterSupplies), FUN=sum) #3055.927
aggregate(oNew_new$profit, list(oNew_new$ProductTypeSmartphone), FUN=sum) #48881.27
aggregate(oNew_new$profit, list(oNew_new$ProductTypeTablet), FUN=sum) #136156.4

# profit_data <- data.frame(productType=c("Software", 
#                                        "Accessories", 
#                                        "Display", 
#                                        "ExtendedWarranty", 
#                                        "GameConsole", 
#                                        "Laptop", 
#                                        "Netbook", 
#                                        "PC", 
#                                        "Printer",
#                                        "PrinterSupplies",
#                                        "Smartphone",
#                                        "Tablet"),
#                          profit = c(7517.159,
#                                     1006.255,
#                                     3994.693,
#                                     24646.97,
#                                     327673.4,
#                                     443377.7,
#                                     108362.9,
#                                     426982.9,
#                                     82157.54,
#                                     3055.927,
#                                     48881.27,
#                                     136156.4))

profit_data <- data.frame(productType=c("Laptop",
                                       "Netbook",
                                       "PC",
                                       "Smartphone"),
                         profit = c(443377.7,
                                    108362.9,
                                    426982.9,
                                    48881.27))
ggplot(profit_data, aes(x = productType, y = profit)) + geom_col()
