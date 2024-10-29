#' Function of dividing GC data set
#'
#' @param data : dataset
#' @param seed :random seed
#'
#' @return :Returns a list of training sets, test sets and verification sets.
#' @export
#'
#' @examples DHMOC.gc_split_dataset(data,seed)
gc_split_dataset <- function(data, seed = 123) {
  set.seed(seed)

  type_1_indices <- which(data$type == 1)
  type_2_indices <- which(data$type == 2)
  type_3_indices <- which(data$type == 3)
  type_4_indices <- which(data$type == 4)

  selected_indices <- c(
    sample(type_1_indices, 100),
    sample(type_2_indices, 100),
    sample(type_3_indices, 100),
    sample(type_4_indices, 100)
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
