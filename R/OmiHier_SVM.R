#' OmiHier SVM
#'
#' This function performs OmiHier analysis using SVM for multiclass classification.
#'
#' @param data Limit its first column category variable (numeric type, limit it must start with 1), 2-n column feature variable.
#' @return This function does not return any value. It plots the OmiHier_SVM dendrogram.
#'
#' @import e1071
#' @import pROC
#' @import dplyr
#' @import readr
#' @import caret
#' @import dendextend
#' @import ggdendro
#' @import stats
#'
#' @examples
#' data_file <- system.file("data", "Faecal_svm.csv", package = "OmiHier",mustWork = TRUE)
#' data <- read.csv(data_file)
#' omihier_svm(data)
#'
#' @export
omihier_svm <- function(data){
  library(e1071)
  library(pROC)
  library(dplyr)
  library(readr)
  library(caret)
  library(dendextend)
  library(ggdendro)
  library(stats)
  
  split_data <- function(data) {
    set.seed(123)  # 设置种子以确保结果可重现
    
    # 找出data[,1]列中的唯一值
    unique_values <- unique(data[,1])
    
    # 创建空的数据框来存储合并后的训练集和测试集
    combined_train_data <- data.frame()
    combined_test_data <- data.frame()
    
    # 对于每个唯一值，按照70:30的比例分割数据集并合并
    for (value in unique_values) {
      # 根据当前唯一值筛选数据
      subset_data <- data[data[,1] == value, ]
      
      # 计算分割点
      split_point <- floor(0.7 * nrow(subset_data))
      
      # 分割数据集
      train_subset <- subset_data[1:split_point, ]
      test_subset <- subset_data[(split_point + 1):nrow(subset_data), ]
      
      # 合并训练集和测试集
      combined_train_data <- rbind(combined_train_data, train_subset)
      combined_test_data <- rbind(combined_test_data, test_subset)
    }
    
    # 将合并后的训练集和测试集存储在列表中
    data_list <- list(train = combined_train_data, test = combined_test_data)
    
    return(data_list)
  }
  
  # Call the function and pass in the dataset brca
  data1 <- split_data(data)
  
  svm_model <- function(train_data, test_data) {
    x_train <- train_data[, -1]
    y_train <- train_data[, 1]
    x_test <- test_data[, -1]
    y_test <- test_data[, 1]
    
    # Storing predicted probabilities and AUC values
    train_predictions <- matrix(0, nrow = nrow(train_data), ncol = length(unique(y_train)))
    test_predictions <- matrix(0, nrow = nrow(test_data), ncol = length(unique(y_train)))
    svm_aucs <- numeric(length(unique(y_train)))
    
    for (class in sort(unique(y_train))) {
      
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
  
  # Construct the OVR model on the training set and apply it on the test set to output the confusion matrix
  result1 <- svm_model(data1$train, data1$test)
  
  # Calculate the similarity matrix with the distance matrix and plot the dendrogram for the next steps
  A <- result1$confusion
  
  # normalization
  normalized_A <- t(apply(A, 1, function(row) row / max(row)))
  
  # Constructing a similarity matrix
  symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
  symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
  
  # Constructing the distance matrix
  affinity_matrix <- symmetric_affinity_matrix
  affinity_matrix[affinity_matrix == 0] <- 0.0001
  distance_matrix <- 1 / affinity_matrix
  
  # Normalize and construct a symmetric distance matrix
  normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
  symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
  symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
  diag(symmetric_distance_matrix) <- 0
  
  # hierarchical clustering
  hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")
  
  # plot
  dend <- as.dendrogram(hc)
  dend <- color_branches(dend, k = length(unique(data[,1]))-1)
  dend <- set(dend, "labels_cex", 0.8)
  plot(dend, main = "OmiHier", xlab = "type")
  
  rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
     symmetric_affinity_matrix,symmetric_distance_matrix)
}