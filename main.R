setwd('d:/DS projects/kaggle/car_insurance')
library(readr)
library(ggplot2)
library(dplyr)
library(caret)
library(xgboost)
library(PRROC)
library(mlr)


insurance_data = read_csv('Car_Insurance_Claim.csv')
data = na.omit(insurance_data)

str(data)
sapply(data, class)

data = data%>%mutate_if(is.character, as.factor)
data$VEHICLE_OWNERSHIP = as.factor(data$VEHICLE_OWNERSHIP)
data$MARRIED = as.factor(data$MARRIED)
data$CHILDREN = as.factor(data$CHILDREN)
data$OUTCOME = as.numeric(data$OUTCOME) 
str(data)
summary(data)

hist(data$SPEEDING_VIOLATIONS)
hist(data$CREDIT_SCORE)
hist(data$ANNUAL_MILEAGE)
hist(data$PAST_ACCIDENTS)
hist(data$DUIS)


table(data$GENDER, data$OUTCOME)
table(data$AGE, data$OUTCOME)
table(data$DRIVING_EXPERIENCE, data$OUTCOME)
table(data$VEHICLE_OWNERSHIP, data$OUTCOME)
table(data$INCOME, data$OUTCOME)
table(data$VEHICLE_YEAR, data$OUTCOME)
table(data$ANNUAL_MILEAGE, data$OUTCOME)



colnames(data)

set.seed(101)
trainIndex = sample(1:nrow(data), size = round(0.7*nrow(data)), replace=FALSE)
train = data[trainIndex ,]
test = data[-trainIndex ,]

str(train)

#preparation data
mod_cols = c('AGE', 'GENDER', 'DRIVING_EXPERIENCE', 'VEHICLE_OWNERSHIP',
             'SPEEDING_VIOLATIONS', 'DUIS')

labels <- train$OUTCOME 
ts_label <- test$OUTCOME
new_tr <- model.matrix(~.+0,data = train[, mod_cols, with=F]) 
new_ts <- model.matrix(~.+0,data = test[, mod_cols,with=F])

dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)


xgbcv <- xgb.cv(data = dtrain, nrounds = 100, 
                nfold = 5, showsd = T, stratified = T, 
                print_every_n = 10, early_stopping_rounds = 20, 
                maximize = F)
xgbcv


#model #1
xgb1 = xgb.train(data = dtrain, 
                 max.depth = 2, 
                 eta = 1, 
                 nthread = 2, 
                 nrounds = 10, 
                 objective = "binary:logistic", 
                 verbose = 0)

xgbpred1 <- predict (xgb1,dtest)
xgbpred_res1 <- as.numeric(xgbpred1 > 0.45,1,0)

CM = confusionMatrix(as.factor(xgbpred_res1), as.factor(ts_label))
CM

mat1 <- xgb.importance (feature_names = colnames(new_tr), model = xgb1)
xgb.plot.importance (importance_matrix = mat1) 


PRROC_obj_xgboost = roc.curve(scores.class0 = xgbpred1, 
                              weights.class0 = ts_label,
                              curve=TRUE,
                              rand.compute = TRUE)
plot(PRROC_obj_xgboost, rand.plot = TRUE, maxminrand.col = "blue")


#mlr
#create tasks
train_copy = train
train_copy$OUTCOME = as.factor(train_copy$OUTCOME)
test_copy = test
test_copy$OUTCOME = as.factor(test_copy$OUTCOME)


traintask = makeClassifTask (data = train_copy,target = "OUTCOME")
testtask = makeClassifTask (data = test_copy,target = "OUTCOME")

#do one hot encoding
traintask = createDummyFeatures (obj = traintask) 
testtask = createDummyFeatures (obj = testtask)

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "prob")
lrn$par.vals = list( objective="binary:logistic", 
                      eval_metric="error", nrounds=100L, eta=0.1)

#set parameter space
params = makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")), 
                          makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                          makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1),
                        makeNumericParam("eta", lower = 0.001,upper = 0.1))

#set resampling strategy
rdesc = makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl = makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune = tuneParams(learner = lrn, task = traintask, 
                     resampling = rdesc, measures = acc, 
                     par.set = params, control = ctrl, 
                     show.info = T)
mytune$x

#set hyperparameters
lrn_tune = setHyperPars(lrn, par.vals = mytune$x)


#train model
xgb2 = train(learner = lrn_tune,task = traintask)
xgb2$learner

class(xgb2)
#predict model
xgbpred2 = predict(xgb2,testtask)


CM2 = confusionMatrix(xgbpred2$data$response, xgbpred2$data$truth)
CM2

df_roc_m2 = generateThreshVsPerfData(xgbpred2, measures = list(fpr, tpr, acc)) 
plotROCCurves(df_roc_m2)
mlr::performance(xgbpred2, mlr::auc)

#stop parallelization
parallelStop()
#


xgb3 = xgb.train(data = dtrain, params = mytune$x, 
                 nrounds=100,
                 objective="binary:logistic", 
                 eval_metric="error")

xgbpred3 = predict (xgb3, dtest)
xgbpred_res3 = as.numeric(xgbpred3 > 0.45,1,0)

CM3 = confusionMatrix(as.factor(xgbpred_res3), as.factor(ts_label))
CM3


PRROC_obj_xgboost = roc.curve(scores.class0 = xgbpred3, 
                              weights.class0 = ts_label,
                              curve=TRUE,
                              rand.compute = TRUE)
plot(PRROC_obj_xgboost, rand.plot = TRUE, maxminrand.col = "blue")


xgb.plot.multi.trees(model = xgb3, feature_names = xgb3$feature_names)
