library(glmnet)
library(pROC)
library(ggplot2)
library(ggsci)
library(dplyr)
library(readr)
library(caret)
library(dendextend)
library(ggdendro)
library(stats)
library(e1071)
library(lightgbm)

##############################################
#                 BRCA-1                     #
##############################################

brca_split_dataset <- function(data, seed = 900) {
  set.seed(seed)
  
  # Get an index of each type
  type_1_indices <- which(data$type == 1)
  type_2_indices <- which(data$type == 2)
  type_3_indices <- which(data$type == 3)
  type_4_indices <- which(data$type == 4)
  
  # Randomly selected sample indexes for each type to meet the requirement of 200 samples of each type
  selected_indices <- c(
    sample(type_1_indices, 200),
    sample(type_2_indices, 200),
    sample(type_3_indices, 200),
    sample(type_4_indices, 200)
  )
  
  # Get the indexes of the test set and validation set
  remaining_indices <- setdiff(1:nrow(data), selected_indices)
  test_indices <- sample(remaining_indices, length(remaining_indices) / 2)
  validation_indices <- setdiff(remaining_indices, test_indices)
  
  # Create training, test and validation sets
  train_set <- data[selected_indices, ]
  test_set <- data[test_indices, ]
  validation_set <- data[validation_indices, ]
  
  # Storing datasets in lists
  dataset_list <- list(
    train = train_set,
    test = test_set,
    validation = validation_set
  )
  
  return(dataset_list)
}

# Call the function and pass in the dataset brca
brca_data1 <- brca_split_dataset(brca)

# View the dimensions of each collection
lapply(brca_data1, dim)

# Define functions for training and testing of individual OVR models
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

# Define the function that runs the OVR model and outputs the confusion matrix
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

# Construct the OVR model on the training set and apply it on the test set to output the confusion matrix
brca_result1 <- run_ovr_models(brca_data1$train, brca_data1$test)
print(brca_result1$confusion)

# Calculate the similarity matrix with the distance matrix and plot the dendrogram for the next steps
A <- brca_result1$confusion
colnames(A) <- c('Basal','Her2','LumA','LumB')
rownames(A) <- c('Basal','Her2','LumA','LumB')

# normalization
normalized_A <- t(apply(A, 1, function(row) row / max(row)))

# Constructing a similarity matrix
symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

# Constructing the distance matrix
affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

# Normalize and construct a symmetric distance matrix
normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

# hierarchical clustering
hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

# plot
pdf("~/R/DHMOC/Figures/brca_dendrogram1.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-BRCA", xlab = "Intrinsic Subtype")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                  BRCA-2                    #
##############################################
brca_data2 <- brca_data1

brca_data2$train$type <- ifelse(brca_data2$train$type == 4, 3, brca_data2$train$type)
brca_data2$test$type <- ifelse(brca_data2$test$type == 4, 3, brca_data2$test$type)
brca_data2$validation$type <- ifelse(brca_data2$validation$type == 4, 3, brca_data2$validation$type)

brca_result2 <- run_ovr_models(brca_data2$train, brca_data2$test)
print(brca_result2$confusion)

A <- brca_result2$confusion
colnames(A) <- c('Basal','Her2','LumA_LumB')
rownames(A) <- c('Basal','Her2','LumA_LumB')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/brca_dendrogram2.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-BRCA", xlab = "Intrinsic Subtype")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                  BRCA-3                    #
##############################################
brca_data3 <- brca_data2

brca_data3$train$type <- ifelse(brca_data3$train$type == 3, 2, brca_data3$train$type)
brca_data3$test$type <- ifelse(brca_data3$test$type == 3, 2, brca_data3$test$type)
brca_data3$validation$type <- ifelse(brca_data3$validation$type == 3, 2, brca_data3$validation$type)

brca_result3 <- run_ovr_models(brca_data3$train, brca_data3$test)
print(brca_result3$confusion)


A <- brca_result3$confusion
colnames(A) <- c('Basal','Her2_LumA_LumB')
rownames(A) <- c('Basal','Her2_LumA_LumB')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/brca_dendrogram3.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 2)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-BRCA", xlab = "Intrinsic Subtype")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)











##############################################
#                   GC-1                     #
##############################################

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

gc_data1 <- gc_split_dataset(gc)
lapply(gc_data1, dim)

gc_result1 <- run_ovr_models(gc_data1$train, gc_data1$test)
print(gc_result1$confusion)

A <- gc_result1$confusion
colnames(A) <- c('CIN','GS','MSI','EBV')
rownames(A) <- c('CIN','GS','MSI','EBV')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/gc_dendrogram1.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-GC", xlab = "Intrinsic Subtype")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                   GC-2                     #
##############################################
gc_data2 <- gc_data1

gc_data2$train$type <- ifelse(gc_data2$train$type == 2, 1, gc_data2$train$type)
gc_data2$train$type <- ifelse(gc_data2$train$type == 3, 2, gc_data2$train$type)
gc_data2$train$type <- ifelse(gc_data2$train$type == 4, 3, gc_data2$train$type)
gc_data2$test$type <- ifelse(gc_data2$test$type == 2, 1, gc_data2$test$type)
gc_data2$test$type <- ifelse(gc_data2$test$type == 3, 2, gc_data2$test$type)
gc_data2$test$type <- ifelse(gc_data2$test$type == 4, 3, gc_data2$test$type)
gc_data2$validation$type <- ifelse(gc_data2$validation$type == 2, 1, gc_data2$validation$type)
gc_data2$validation$type <- ifelse(gc_data2$validation$type == 3, 2, gc_data2$validation$type)
gc_data2$validation$type <- ifelse(gc_data2$validation$type == 4, 3, gc_data2$validation$type)

gc_result2 <- run_ovr_models(gc_data2$train, gc_data2$test)
print(gc_result2$confusion)

A <- gc_result2$confusion
colnames(A) <- c('CIN_GS','MSI','EBV')
rownames(A) <- c('CIN_GS','MSI','EBV')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/gc_dendrogram2.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-GC", xlab = "Intrinsic Subtype")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                   GC-3                     #
##############################################
gc_data3 <- gc_data2

gc_data3$train$type <- ifelse(gc_data3$train$type == 2, 1, gc_data3$train$type)
gc_data3$train$type <- ifelse(gc_data3$train$type == 3, 2, gc_data3$train$type)
gc_data3$test$type <- ifelse(gc_data3$test$type == 2, 1, gc_data3$test$type)
gc_data3$test$type <- ifelse(gc_data3$test$type == 3, 2, gc_data3$test$type)
gc_data3$validation$type <- ifelse(gc_data3$validation$type == 2, 1, gc_data3$validation$type)
gc_data3$validation$type <- ifelse(gc_data3$validation$type == 3, 2, gc_data3$validation$type)

gc_result3 <- run_ovr_models(gc_data3$train, gc_data3$test)
print(gc_result3$confusion)


A <- gc_result3$confusion
colnames(A) <- c('CIN_GS_MSI','EBV')
rownames(A) <- c('CIN_GS_MSI','EBV')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/gc_dendrogram3.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 2)
dend <- set(dend, "labels_cex", 0.8) 
plot(dend, main = "DHMOC-GC", xlab = "Intrinsic Subtype")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)












##############################################
#                 Faecal-1                   #
##############################################

ktuple_split_dataset <- function(data, seed = 123) {
  set.seed(seed)
  
  type_1_indices <- which(data$type == 0)
  type_2_indices <- which(data$type == 1)
  type_3_indices <- which(data$type == 2)
  
  selected_indices <- c(
    sample(type_1_indices, 38),
    sample(type_2_indices, 27),
    sample(type_3_indices, 29)
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

ktuple_data1 <- ktuple_split_dataset(ktuple)
lapply(ktuple_data1, dim)

lightgbm_train_test_ovr_model <- function(train_data, test_data, positive_class) {
  
  train_data$label <- ifelse(train_data$type == positive_class, 1, 0)
  test_data$label <- ifelse(test_data$type == positive_class, 1, 0)
  
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

ktuple_result1 <- lightgbm_run_ovr_models(ktuple_data1$train, ktuple_data1$test)
print(ktuple_result1$confusion)

A <- ktuple_result1$confusion
colnames(A) <- c('Normal','Benign','Malignant')
rownames(A) <- c('Normal','Benign','Malignant')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/ktuple_dendrogram1.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-Ktuple", xlab = "Type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)









##############################################
#                   HCC-1                    #
##############################################

HCC_split_dataset <- function(data, seed = 123) {
  set.seed(seed)
  
  type_1_indices <- which(data$V3 == 1)
  type_2_indices <- which(data$V3 == 2)
  type_3_indices <- which(data$V3 == 3)
  type_4_indices <- which(data$V3 == 4)
  
  selected_indices <- c(
    sample(type_1_indices, 26),
    sample(type_2_indices, 2136),
    sample(type_3_indices, 1076),
    sample(type_4_indices, 2078)
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

HCC_data1 <- HCC_split_dataset(HCC)
lapply(HCC_data1, dim)

HCC_result1 <- lightgbm_run_ovr_models(HCC_data1$train[,-1:-2], HCC_data1$test[,-1:-2])
print(HCC_result1$confusion)

A <- HCC_result1$confusion
colnames(A) <- c('Immune','Normal','Stromal','Tumor')
rownames(A) <- c('Immune','Normal','Stromal','Tumor')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/HCC_dendrogram1.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 4)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-Spatial", xlab = "Type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                   HCC-2                    #
##############################################
HCC_data2 <- HCC_data1

HCC_data2$train$V3 <- ifelse(HCC_data2$train$V3 == 3, 1, HCC_data2$train$V3)
HCC_data2$train$V3 <- ifelse(HCC_data2$train$V3 == 4, 3, HCC_data2$train$V3)
HCC_data2$test$V3 <- ifelse(HCC_data2$test$V3 == 3, 1, HCC_data2$test$V3)
HCC_data2$test$V3 <- ifelse(HCC_data2$test$V3 == 4, 3, HCC_data2$test$V3)
HCC_data2$validation$V3 <- ifelse(HCC_data2$validation$V3 == 3, 1, HCC_data2$validation$V3)
HCC_data2$validation$V3 <- ifelse(HCC_data2$validation$V3 == 4, 3, HCC_data2$validation$V3)

HCC_result2 <- lightgbm_run_ovr_models(HCC_data2$train[,-1:-2], HCC_data2$test[,-1:-2])
print(HCC_result2$confusion)

A <- HCC_result2$confusion
colnames(A) <- c('Immune_Stromal','Normal','Tumor')
rownames(A) <- c('Immune_Stromal','Normal','Tumor')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/HCC_dendrogram2.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-Spatial", xlab = "Type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                   HCC-3                    #
##############################################
HCC_data3 <- HCC_data2

HCC_data3$train$V3 <- ifelse(HCC_data3$train$V3 == 2, 1, HCC_data3$train$V3)
HCC_data3$train$V3 <- ifelse(HCC_data3$train$V3 == 3, 2, HCC_data3$train$V3)
HCC_data3$test$V3 <- ifelse(HCC_data3$test$V3 == 2, 1, HCC_data3$test$V3)
HCC_data3$test$V3 <- ifelse(HCC_data3$test$V3 == 3, 2, HCC_data3$test$V3)
HCC_data3$validation$V3 <- ifelse(HCC_data3$validation$V3 == 2, 1, HCC_data3$validation$V3)
HCC_data3$validation$V3 <- ifelse(HCC_data3$validation$V3 == 3, 2, HCC_data3$validation$V3)

HCC_result3 <- lightgbm_run_ovr_models(HCC_data3$train[,-1:-2], HCC_data3$test[,-1:-2])
print(HCC_result3$confusion)

A <- HCC_result3$confusion
colnames(A) <- c('Immune_Stromal_Normal','Tumor')
rownames(A) <- c('Immune_Stromal_Normal','Tumor')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/HCC_dendrogram3.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 2)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-Spatial", xlab = "Type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)











##############################################
#                 NCI-RNA-2                  #
##############################################

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

NCIrna_data1 <- NCIrna_split_dataset(NCIrna)

lapply(NCIrna_data1, dim)

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

NCIrna_result1 <- svm_model(NCIrna_data1$train, NCIrna_data1$test)

print(NCIrna_result1$confusion)

A <- NCIrna_result1$confusion
colnames(A) <- c('1','2','3')
rownames(A) <- c('1','2','3')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/NCIrna_dendrogram1.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-NCIrna", xlab = "Type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                NCI-RNA-2                   #
##############################################
NCIrna_data2 <- NCIrna_data1

NCIrna_data2$train$Type <- ifelse(NCIrna_data2$train$Type == 3, 2, NCIrna_data2$train$Type)
NCIrna_data2$test$Type <- ifelse(NCIrna_data2$test$Type == 3, 2, NCIrna_data2$test$Type)
NCIrna_data2$validation$Type <- ifelse(NCIrna_data2$validation$Type == 3, 2, NCIrna_data2$validation$Type)

NCIrna_result2 <- svm_model(NCIrna_data2$train, NCIrna_data2$test)
print(NCIrna_result2$confusion)

A <- NCIrna_result2$confusion
colnames(A) <- c('1','2_3')
rownames(A) <- c('1','2_3')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/NCIrna_dendrogram2.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 2)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-NCIrna", xlab = "Type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)









##############################################
#                Lymphoid-1                  #
##############################################

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

lymphoid_data1 <- lymphoid_split_dataset(lymphoid)

lapply(lymphoid_data1, dim)

lymphoid_result1 <- svm_model(lymphoid_data1$train, lymphoid_data1$test)

print(lymphoid_result1$confusion)

A <- lymphoid_result1$confusion
colnames(A) <- c('CD4 T cell','NK cell','NK T cell','CD8 T cell','B cell')
rownames(A) <- c('CD4 T cell','NK cell','NK T cell','CD8 T cell','B cell')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/lymphoid_dendrogram1.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 4)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-lymphoid", xlab = "type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                Lymphoid-2                  #
##############################################
lymphoid_data2 <- lymphoid_data1

lymphoid_data2$train$type <- ifelse(lymphoid_data2$train$type == 2, 1, lymphoid_data2$train$type)
lymphoid_data2$train$type <- ifelse(lymphoid_data2$train$type == 3, 2, lymphoid_data2$train$type)
lymphoid_data2$train$type <- ifelse(lymphoid_data2$train$type == 4, 3, lymphoid_data2$train$type)
lymphoid_data2$test$type <- ifelse(lymphoid_data2$test$type == 2, 1, lymphoid_data2$test$type)
lymphoid_data2$test$type <- ifelse(lymphoid_data2$test$type == 3, 2, lymphoid_data2$test$type)
lymphoid_data2$test$type <- ifelse(lymphoid_data2$test$type == 4, 3, lymphoid_data2$test$type)
lymphoid_data2$validation$type <- ifelse(lymphoid_data2$validation$type == 2, 1, lymphoid_data2$validation$type)
lymphoid_data2$validation$type <- ifelse(lymphoid_data2$validation$type == 3, 2, lymphoid_data2$validation$type)
lymphoid_data2$validation$type <- ifelse(lymphoid_data2$validation$type == 4, 3, lymphoid_data2$validation$type)

lymphoid_result2 <- svm_model(lymphoid_data2$train, lymphoid_data2$test)
print(lymphoid_result2$confusion)

A <- lymphoid_result2$confusion
colnames(A) <- c('CD4 T cell','NK_NKT cell','CD8 T cell','B cell')
rownames(A) <- c('CD4 T cell','NK_NKT cell','CD8 T cell','B cell')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/lymphoid_dendrogram2.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-lymphoid", xlab = "type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)


##############################################
#                Lymphoid-3                  #
##############################################
lymphoid_data3 <- lymphoid_data2

lymphoid_data3$train$type <- ifelse(lymphoid_data3$train$type == 1, 0, lymphoid_data3$train$type)
lymphoid_data3$train$type <- ifelse(lymphoid_data3$train$type == 2, 1, lymphoid_data3$train$type)
lymphoid_data3$train$type <- ifelse(lymphoid_data3$train$type == 3, 2, lymphoid_data3$train$type)
lymphoid_data3$test$type <- ifelse(lymphoid_data3$test$type == 1, 0, lymphoid_data3$test$type)
lymphoid_data3$test$type <- ifelse(lymphoid_data3$test$type == 2, 1, lymphoid_data3$test$type)
lymphoid_data3$test$type <- ifelse(lymphoid_data3$test$type == 3, 2, lymphoid_data3$test$type)
lymphoid_data3$validation$type <- ifelse(lymphoid_data3$validation$type == 1, 0, lymphoid_data3$validation$type)
lymphoid_data3$validation$type <- ifelse(lymphoid_data3$validation$type == 2, 1, lymphoid_data3$validation$type)
lymphoid_data3$validation$type <- ifelse(lymphoid_data3$validation$type == 3, 2, lymphoid_data3$validation$type)

lymphoid_result3 <- svm_model(lymphoid_data3$train, lymphoid_data3$test)
print(lymphoid_result3$confusion)

A <- lymphoid_result3$confusion
colnames(A) <- c('CD4T_NK_NKT cell','CD8 T cell','B cell')
rownames(A) <- c('CD4T_NK_NKT cell','CD8 T cell','B cell')

normalized_A <- t(apply(A, 1, function(row) row / max(row)))

symmetric_affinity_matrix <- (normalized_A + t(normalized_A)) / 2
symmetric_affinity_matrix <- round(symmetric_affinity_matrix, 3)
print(symmetric_affinity_matrix)

affinity_matrix <- symmetric_affinity_matrix
affinity_matrix[affinity_matrix == 0] <- 0.0001
distance_matrix <- 1 / affinity_matrix

normalized_distance_matrix <- t(apply(distance_matrix, 1, function(row) row / max(row)))
symmetric_distance_matrix <- (normalized_distance_matrix + t(normalized_distance_matrix)) / 2
symmetric_distance_matrix <- round(symmetric_distance_matrix, 3)
diag(symmetric_distance_matrix) <- 0
print(symmetric_distance_matrix)

hc <- hclust(as.dist(symmetric_distance_matrix), method = "single")

pdf("~/R/DHMOC/Figures/lymphoid_dendrogram3.pdf")
dend <- as.dendrogram(hc)
dend <- color_branches(dend, k = 3)
dend <- set(dend, "labels_cex", 0.8)
plot(dend, main = "DHMOC-lymphoid", xlab = "type")
dev.off()

rm(A,dend,distance_matrix,hc,affinity_matrix,normalized_A,normalized_distance_matrix,
   symmetric_affinity_matrix,symmetric_distance_matrix)
