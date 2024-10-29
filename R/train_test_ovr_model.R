#' Training set and test set model
#'
#' @param train_data  :training dataset
#' @param test_data :testing dataset
#' @param positive_class :
#'
#' @return :Returns a list containing training labels and test labels.
#' @export
#'
#' @examples DHMOC.train_test_ovr_model(train,test,calss)
train_test_ovr_model <- function(train_data, test_data, positive_class) {
  # Set the label to 1 for positive samples and 0 for other samples
  train_data$label <- ifelse(train_data$type == positive_class, 1, 0)
  test_data$label <- ifelse(test_data$type == positive_class, 1, 0)

  # Training the lasso-logistic model
  model <- cv.glmnet(as.matrix(train_data[, -c(1, ncol(train_data))]), train_data$label, alpha = 1)

  # Perform 10-fold cross validation
  train_preds <- predict(model, s = "lambda.min", newx = as.matrix(train_data[, -c(1, ncol(train_data))]), type = "response")

  # Get test set probability predictions
  test_preds <- predict(model, s = "lambda.min", newx = as.matrix(test_data[, -c(1, ncol(test_data))]), type = "response")

  return(list(train_preds = train_preds, test_preds = test_preds))
}
