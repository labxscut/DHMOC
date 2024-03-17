#' OmiHier LightGBM
#'
#' This function performs OmiHier analysis using LightGBM for multiclass classification.
#'
#' @param data Limit its first column category variable (numeric type), 2-n column feature variable.
#' @return This function does not return any value. It plots the OmiHier_LightGBM dendrogram.
#'
#' @import lightgbm
#' @import pROC
#' @import dplyr
#' @import readr
#' @import caret
#' @import dendextend
#' @import ggdendro
#' @import stats
#'
#' @examples
#' data_file <- system.file("data", "Faecal.csv", package = "OmiHier",mustWork = TRUE)
#' data <- read.csv(data_file)
#' omihier_lightgbm(data)
#'
#' @export
omihier_lightgbm <- function(data){
  library(lightgbm)
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
  
  lightgbm_train_test_ovr_model <- function(train_data, test_data, positive_class) {
    
    train_data$label <- ifelse(train_data[,1] == positive_class, 1, 0)
    test_data$label <- ifelse(test_data[,1] == positive_class, 1, 0)
    
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
  
  lightgbm_run_ovr_models <- function(train_data, test_data) {
    classes <- sort(unique(train_data[,1]))
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
    
    confusion <- table(test_data[,1], final_predictions)
    
    return(list(cv_predictions = cv_predictions, test_predictions = test_predictions, 
                confusion = confusion))
  }
  
  # Construct the OVR model on the training set and apply it on the test set to output the confusion matrix
  result1 <- lightgbm_run_ovr_models(data1$train, data1$test)
  
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