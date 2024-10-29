#' Function of single cell RNA sequence data set for dividing lymphocytes
#'
#' @param data :Single cell RNA sequence data set of lymphocytes
#' @param seed :random seed
#'
#' @return :Contains a list of verification sets, training sets and test sets.
#' @export
#'
#' @examples DHMOC.lymphoid_split_dataset(data,123)
lymphoid_split_dataset <- function(data, seed = 123) {
  set.seed(seed)

  type_1_indices <- which(data$type == 0)
  type_2_indices <- which(data$type == 1)
  type_3_indices <- which(data$type == 2)
  type_4_indices <- which(data$type == 3)
  type_5_indices <- which(data$type == 4)

  selected_indices <- c(
    sample(type_1_indices, 1095),
    sample(type_2_indices, 2625),
    sample(type_3_indices, 2249),
    sample(type_4_indices, 953),
    sample(type_5_indices, 319)
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
