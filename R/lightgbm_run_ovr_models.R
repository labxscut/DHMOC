#' Lightgbm model
#'
#' @param train_data :Training data set
#' @param test_data :Testing data set
#'
#' @return :List containing prediction labels and confusion matrices
#' @export
#'
#' @examples DHMOC.lightgbm_run_ovr_models(train,test)
lightgbm_run_ovr_models <- function(train_data, test_data) {
  classes <- unique(train_data$type)
  num_classes <- length(classes)

  cv_predictions <- matrix(0, nrow(train_data), num_classes)
  test_predictions <- matrix(0, nrow(test_data), num_classes)

  for (i in 1:num_classes) {
    class <- classes[i]
    ovr_result <- lightgbm_train_test_ovr_model(train_data, test_data, class)

    cv_predictions[, i] <- ovr_result$cv_preds
    test_predictions[, i] <- ovr_result$test_preds
  }

  final_predictions <- max.col(test_predictions)

  confusion <- table(test_data$type, final_predictions)

  return(list(cv_predictions = cv_predictions, test_predictions = test_predictions,
              confusion = confusion))
}
