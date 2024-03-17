#' OmiHier Lasso Function
#'
#' This function performs OmiHier analysis using lasso logistic regression.
#'
#' @param data Limit its first column category variable (numeric type), 2-n column feature variable.
#'
#' @return This function does not return any value. It plots the OmiHier_Lasso dendrogram.
#'
#' @import glmnet pROC dplyr readr caret dendextend ggdendro stats
#'
#' @examples
#' data_file <- system.file("data", "Faecal.csv", package = "OmiHier")
#' data <- read.csv(data_file)
#' omihier_lasso(data)
#'
#' @export
omihier_lasso <- function(data){
    library(glmnet)
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
    
    # Define functions for training and testing of individual OVR models
    train_test_ovr_model <- function(train_data, test_data, positive_class) {
      # Set the label to 1 for positive samples and 0 for other samples
      train_data$label <- ifelse(train_data[,1] == positive_class, 1, 0)
      test_data$label <- ifelse(test_data[,1] == positive_class, 1, 0)
      
      # Training the lasso-logistic model
      model <- cv.glmnet(as.matrix(train_data[, -c(1, ncol(train_data))]), train_data$label, alpha = 1)
      
      # Perform 10-fold cross validation
      train_preds <- predict(model, s = "lambda.min", newx = as.matrix(train_data[, -c(1, ncol(train_data))]), type = "response")
      
      # Get test set probability predictions
      test_preds <- predict(model, s = "lambda.min", newx = as.matrix(test_data[, -c(1, ncol(test_data))]), type = "response")
      
      return(list(train_preds = train_preds, test_preds = test_preds))
    }
    
    # Define the function that runs the OVR model and outputs the confusion matrix
    run_ovr_models <- function(train_data, test_data) {
      classes <- sort(unique(train_data[,1]))
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
      confusion <- table(c(train_data[,1], test_data[,1]), final_predictions)
      
      return(list(train_predictions = train_predictions, test_predictions = test_predictions, 
                  confusion = confusion))
    }
    
    # Construct the OVR model on the training set and apply it on the test set to output the confusion matrix
    result1 <- run_ovr_models(data1$train, data1$test)
    
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
