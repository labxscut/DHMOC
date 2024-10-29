#' SVM model
#'
#' @param train_data :Training dataset
#' @param test_data :Testing data set
#'
#' @return :Returns a list of training sets, test sets, test labels and mixing matrices.
#' @export
#'
#' @examples DHMOC.svm_model(train,test)
svm_model <- function(train_data, test_data) {
  x_train <- train_data[, -1]
  y_train <- train_data[, 1]
  x_test <- test_data[, -1]
  y_test <- test_data[, 1]

  # Storing predicted probabilities and AUC values
  train_predictions <- matrix(0, nrow = nrow(train_data), ncol = length(unique(y_train)))
  test_predictions <- matrix(0, nrow = nrow(test_data), ncol = length(unique(y_train)))
  svm_aucs <- numeric(length(unique(y_train)))

  for (class in unique(y_train)) {

    y_train_ovr <- ifelse(y_train == class, 1, 0)
    y_test_ovr <- ifelse(y_test == class, 1, 0)

    model <- svm(x_train, y_train_ovr, probability = TRUE)

    prob1 <- predict(model, x_train, probability = TRUE)
    prob2 <- predict(model, x_test, probability = TRUE)

    train_predictions[, class] <- prob1
    test_predictions[, class] <- prob2
  }

  svm_pred <- max.col(test_predictions, "first")
  svm_confusion <- table(Actual = y_test, Predicted = svm_pred)

  return(list(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, model = model,
              train_predictions = train_predictions, test_predictions = test_predictions,
              confusion = svm_confusion))
}
