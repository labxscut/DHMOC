#' Lightgbm training test model
#'
#' @param train_data :training dataset
#' @param test_data :testing dataset
#' @param positive_class :Positive label
#'
#' @return :Returns the prediction label of the test.
#' @export
#'
#' @examples DHMOC.lightgbm_train_test_ovr_model(train,test,class)
lightgbm_train_test_ovr_model <- function(train_data, test_data, positive_class) {

  train_data$label <- ifelse(train_data$type == positive_class, 1, 0)
  test_data$label <- ifelse(test_data$type == positive_class, 1, 0)

  dtrain <- lgb.Dataset(data = as.matrix(train_data[, -c(1, ncol(train_data))]), label = train_data$label)

  params <- list(objective = "binary", metric = "binary_logloss", boosting_type = "gbdt")

  cv_results <- lgb.cv(params,
                       dtrain,
                       nfold = 10,
                       nrounds = 100)
  best_iter <- cv_results$best_iter
  model <- lgb.train(params = params, data = dtrain, num_boost_round = best_iter)

  cv_preds <- predict(model, as.matrix(train_data[, -c(1, ncol(train_data))]))
  test_preds <- predict(model, as.matrix(test_data[, -c(1, ncol(test_data))]))

  return(list(cv_preds = cv_preds, test_preds = test_preds))
}
