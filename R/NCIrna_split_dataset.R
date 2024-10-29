#' Function of dividing single cell RNA sequence of gastric cancer cell line NCI-N87
#'
#' @param data :Single cell RNA sequence of gastric cancer cell line NCI-N87
#' @param seed :random seed
#'
#' @return :Returns a data list containing training set, test set and verification set.
#' @export
#'
#' @examples DHMOC.NCIrna_split_dataset(data,seed)
NCIrna_split_dataset <- function(data, seed = 123) {
  set.seed(seed)

  type_1_indices <- which(data$Type == 1)
  type_2_indices <- which(data$Type == 2)
  type_3_indices <- which(data$Type == 3)

  selected_indices <- c(
    sample(type_1_indices, 802),
    sample(type_2_indices, 77),
    sample(type_3_indices, 422)
  )

  remaining_indices <- setdiff(1:nrow(data), selected_indices)
  test_indices <- sample(remaining_indices, length(remaining_indices) / 2)
  validation_indices <- setdiff(remaining_indices, test_indices)

  train_set <- data[selected_indices, ]
  test_set <- data[test_indices, ]
  validation_set <- data[validation_indices, ]

  dataset_list <- list(
    train = train_set,
    test = test_set,
    validation = validation_set
  )

  return(dataset_list)
}
