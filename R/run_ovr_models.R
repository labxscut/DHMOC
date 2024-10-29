#' Function to run the model
#'
#' @param train_data :Training data set
#' @param test_data : Testing data set
#'
#' @return : Returns a list containing training labels, test labels and confusion matrices.
#' @export
#'
#' @examples DHMOC.run_ovr_models(train,test)
run_ovr_models <- function(train_data, test_data) {
  classes <- unique(train_data$type)
  num_classes <- length(classes)

  # Storing cross-validation results and test set predictions
  train_predictions <- matrix(0, nrow(train_data), num_classes)
  test_predictions <- matrix(0, nrow(test_data), num_classes)

  for (i in 1:num_classes) {
    class <- classes[i]
    ovr_result <- train_test_ovr_model(train_data, test_data, class)

    train_predictions[, i] <- ovr_result$train_preds
    test_predictions[, i] <- ovr_result$test_preds
  }

  # Getting the final prediction
  final_predictions <- max.col(rbind(train_predictions, test_predictions))

  # Creating a Confusion Matrix
  confusion <- table(c(train_data$type, test_data$type), final_predictions)

  return(list(train_predictions = train_predictions, test_predictions = test_predictions,
              confusion = confusion))
}
