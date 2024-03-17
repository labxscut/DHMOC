type3_split_dataset <- function(data, seed = 123) {
  set.seed(seed)
  
  type_1_indices <- which(data$X1 == 1)
  type_2_indices <- which(data$X1 == 2)
  type_3_indices <- which(data$X1 == 3)
  
  selected_indices <- c(
    sample(type_1_indices, 233),
    sample(type_2_indices, 233),
    sample(type_3_indices, 233)
  )
  
  test_indices <- setdiff(1:nrow(data), selected_indices)
  
  train_set <- data[selected_indices, ]
  test_set <- data[test_indices, ]
  
  dataset_list <- list(
    train = train_set,
    test = test_set
  )
  
  return(dataset_list)
}

type4_split_dataset <- function(data, seed = 123) {
  set.seed(seed)
  
  type_1_indices <- which(data$X1 == 1)
  type_2_indices <- which(data$X1 == 2)
  type_3_indices <- which(data$X1 == 3)
  type_4_indices <- which(data$X1 == 4)
  
  selected_indices <- c(
    sample(type_1_indices, 200),
    sample(type_2_indices, 200),
    sample(type_3_indices, 200),
    sample(type_4_indices, 200)
  )
  
  test_indices <- setdiff(1:nrow(data), selected_indices)
  
  train_set <- data[selected_indices, ]
  test_set <- data[test_indices, ]
  
  dataset_list <- list(
    train = train_set,
    test = test_set
  )
  
  return(dataset_list)
}

type6_split_dataset <- function(data, seed = 123) {
  set.seed(seed)
  
  type_1_indices <- which(data$X1 == 1)
  type_2_indices <- which(data$X1 == 2)
  type_3_indices <- which(data$X1 == 3)
  type_4_indices <- which(data$X1 == 4)
  type_5_indices <- which(data$X1 == 5)
  type_6_indices <- which(data$X1 == 6)
  
  selected_indices <- c(
    sample(type_1_indices, 120),
    sample(type_2_indices, 120),
    sample(type_3_indices, 120),
    sample(type_4_indices, 120),
    sample(type_5_indices, 120),
    sample(type_6_indices, 120)
  )
  
  test_indices <- setdiff(1:nrow(data), selected_indices)
  
  train_set <- data[selected_indices, ]
  test_set <- data[test_indices, ]
  
  dataset_list <- list(
    train = train_set,
    test = test_set
  )
  
  return(dataset_list)
}

type10_split_dataset <- function(data, seed = 123) {
  set.seed(seed)
  
  type_1_indices <- which(data$X1 == 1)
  type_2_indices <- which(data$X1 == 2)
  type_3_indices <- which(data$X1 == 3)
  type_4_indices <- which(data$X1 == 4)
  type_5_indices <- which(data$X1 == 5)
  type_6_indices <- which(data$X1 == 6)
  type_7_indices <- which(data$X1 == 7)
  type_8_indices <- which(data$X1 == 8)
  type_9_indices <- which(data$X1 == 9)
  type_10_indices <- which(data$X1 == 10)
  
  selected_indices <- c(
    sample(type_1_indices, 80),
    sample(type_2_indices, 80),
    sample(type_3_indices, 80),
    sample(type_4_indices, 80),
    sample(type_5_indices, 80),
    sample(type_6_indices, 80),
    sample(type_7_indices, 80),
    sample(type_8_indices, 80),
    sample(type_9_indices, 80),
    sample(type_10_indices, 80)
  )
  
  test_indices <- setdiff(1:nrow(data), selected_indices)
  
  train_set <- data[selected_indices, ]
  test_set <- data[test_indices, ]
  
  dataset_list <- list(
    train = train_set,
    test = test_set
  )
  
  return(dataset_list)
}

s3_data <- list()
s4_data <- list()
s6_data <- list()
s10_data <- list()
for (i in 1:length(sd)){
  s3_data[[i]] <- type3_split_dataset(data.frame(s3[[i]]))
  s4_data[[i]] <- type4_split_dataset(data.frame(s4[[i]]))
  s6_data[[i]] <- type6_split_dataset(data.frame(s6[[i]]))
  s10_data[[i]] <- type10_split_dataset(data.frame(s10[[i]]))
}


##########################################################
#                       OVR model                        #
##########################################################
svm_model <- function(num_categories, train_data, test_data) {
  
  x_train <- train_data[, -1]
  y_train <- train_data[, 1]
  x_test <- test_data[, -1]
  y_test <- test_data[, 1]
  
  train_predictions <- matrix(0, nrow = nrow(train_data), ncol = length(unique(y_train)))
  test_predictions <- matrix(0, nrow = nrow(test_data), ncol = length(unique(y_train)))
  
  for (class in unique(y_train)) {
    
    y_train_ovr <- ifelse(y_train == class, 1, 0)
    y_test_ovr <- ifelse(y_test == class, 1, 0)
    
    model <- svm(x_train, y_train_ovr, probability = TRUE)
    
    prob1 <- predict(model, x_train, probability = TRUE)
    prob2 <- predict(model, x_test, probability = TRUE)
    
    train_predictions[, class] <- prob1
    test_predictions[, class] <- prob2
  }
  colnames(train_predictions) <- 1:num_categories
  colnames(test_predictions) <- 1:num_categories
  
  svm_pred <- max.col(test_predictions, "first")
  svm_confusion <- table(Actual = y_test, Predicted = svm_pred)
  
  train_roc <- multiclass.roc(y_train, train_predictions, levels = 1:num_categories)
  test_roc <- multiclass.roc(y_test, test_predictions, levels = 1:num_categories)
  train_auc <- train_roc$auc
  test_auc <- test_roc$auc
  
  return(list(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, model = model,
              train_predictions = train_predictions, test_predictions = test_predictions, 
              confusion = svm_confusion, train_auc = train_auc, test_auc = test_auc))
}

s3_result <- list()
s4_result <- list()
s6_result <- list()
s10_result <- list()
for (i in 6:10) {
  s3_result[[i]] <- svm_model(3, s3_data[[i]]$train, s3_data[[i]]$test)
  s4_result[[i]] <- svm_model(4, s4_data[[i]]$train, s4_data[[i]]$test)
  s6_result[[i]] <- svm_model(6, s6_data[[i]]$train, s6_data[[i]]$test)
  s10_result[[i]] <- svm_model(10, s10_data[[i]]$train, s10_data[[i]]$test)
}






##########################################################
#   Calculate the distance between similarity matrices   #
##########################################################
library(phangorn)

tree_distance <- function(x, y){
  B <- t(apply(y, 1, function(row) row / max(row)))
  B <- (B + t(B)) / 2
  B[B == 0] <- 0.0001
  B <- 1 / B
  B <- t(apply(B, 1, function(row) row / max(row)))
  B <- (B + t(B)) / 2
  diag(B) <- 0
  
  hc <- hclust(as.dist(B), method = "single")
  tree2 <- as.phylo(hc)
  
  rf_distance <- RF.dist(x, tree2, normalize = TRUE, check.labels = TRUE, rooted = TRUE)

  return(rf_distance)
}

s3_tree_distance <- list()
s4_tree_distance <- list()
s6_tree_distance <- list()
s10_tree_distance <- list()
for (i in 1:length(sd)){
  s3_tree_distance[[i]] <- tree_distance(s3_tree, s3_result[[i]]$confusion)
  s4_tree_distance[[i]] <- tree_distance(s4_tree, s4_result[[i]]$confusion)
  s6_tree_distance[[i]] <- tree_distance(s6_tree, s6_result[[i]]$confusion)
  s10_tree_distance[[i]] <- tree_distance(s10_tree, s10_result[[i]]$confusion)
}
for (i in 1:length(sd)){
  if (all(diag(s3_result[[i]][["confusion"]]) != 0) && all(s3_result[[i]][["confusion"]] - diag(diag(s3_result[[i]][["confusion"]])) == 0)){
    s3_tree_distance[[i]] <- 0
  }
  if (all(diag(s4_result[[i]][["confusion"]]) != 0) && all(s4_result[[i]][["confusion"]] - diag(diag(s4_result[[i]][["confusion"]])) == 0)){
    s4_tree_distance[[i]] <- 0
  }
  if (all(diag(s6_result[[i]][["confusion"]]) != 0) && all(s6_result[[i]][["confusion"]] - diag(diag(s6_result[[i]][["confusion"]])) == 0)){
    s6_tree_distance[[i]] <- 0
  }
  if (all(diag(s10_result[[i]][["confusion"]]) != 0) && all(s10_result[[i]][["confusion"]] - diag(diag(s10_result[[i]][["confusion"]])) == 0)){
    s10_tree_distance[[i]] <- 0
  }
}






##########################################################
#                         Plot                           #
##########################################################
AUC <- matrix(0, 4, length(sd))
for (i in 1:length(sd)){
  AUC[1,i] <- s3_result[[i]]$test_auc
  AUC[2,i] <- s4_result[[i]]$test_auc
  AUC[3,i] <- s6_result[[i]]$test_auc
  AUC[4,i] <- s10_result[[i]]$test_auc
}
sd <- sd[-8:-10]
AUC <- AUC[,-8:-10]

data1 <- data.frame(sd, AUC = AUC[1,])
data2 <- data.frame(sd, AUC = AUC[2,])
data3 <- data.frame(sd, AUC = AUC[3,])
data4 <- data.frame(sd, AUC = AUC[4,])
par(mfrow = c(1,2))

plot(0, 0, xlim = range(log10(sd)), ylim = range(c(0.1,data1$AUC, data2$AUC, data3$AUC, data4$AUC)), 
     xlab = "data variability (log10(λ))", ylab = "AUC", main = "Classification Performance (AUC)")

lines(log10(data1$sd), data1$AUC, col = "red", type = "b", lwd = 2, lty = 1, pch = 1)
lines(log10(data2$sd), data2$AUC, col = "blue", type = "b", lwd = 2, lty = 2, pch = 2)
lines(log10(data3$sd), data3$AUC, col = "orange", type = "b", lwd = 2, lty = 3, pch = 3)
lines(log10(data4$sd), data4$AUC, col = "purple", type = "b", lwd = 2, lty = 4, pch = 4)

legend("bottomright", legend = c("K=3", "K=4", "K=6", "K=10"),
       col = c("red", "blue", "orange", "purple"), lwd = 2, lty = c(1, 2, 3, 4), pch = c(1, 2, 3, 4))


data1 <- data.frame(sd, similarity = do.call(rbind, s3_tree_distance[1:7]))
data2 <- data.frame(sd, similarity = do.call(rbind, s4_tree_distance[1:7]))
data3 <- data.frame(sd, similarity = do.call(rbind, s6_tree_distance[1:7]))
data4 <- data.frame(sd, similarity = do.call(rbind, s10_tree_distance[1:7]))

plot(0, 0, xlim = range(log10(sd)), ylim = range(c(0,1)), 
     xlab = "data variability (log10(λ))", ylab = "NormalRFDistance", main = "Hierarchy Correctness \n (Normalized Robinson-Foulds distance)")

lines(log10(data1$sd), data1$similarity, col = "red", type = "b", lwd = 2, lty = 1, pch = 1)
lines(log10(data2$sd), data2$similarity, col = "blue", type = "b", lwd = 2, lty = 2, pch = 2)
lines(log10(data3$sd), data3$similarity, col = "orange", type = "b", lwd = 2, lty = 3, pch = 3)
lines(log10(data4$sd), data4$similarity, col = "purple", type = "b", lwd = 2, lty = 4, pch = 4)

legend("topright", legend = c("K=3", "K=4", "K=6", "K=10"),
       col = c("red", "blue", "orange", "purple"), lwd = 2, lty = c(1, 2, 3, 4), pch = c(1, 2, 3, 4))
