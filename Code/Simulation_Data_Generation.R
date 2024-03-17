##############################################
#              Normal - 3 types              #
##############################################

# Load required library
library(mvtnorm)
library(ape)

# Set the number of categories, samples, and features
num_categories <- 3
num_samples <- 999
num_features <- 1000

# Define the means and standard deviations for each category-feature combination
means <- matrix(0, nrow = num_categories, ncol = num_features)

# Generate uniform data for the means
for (j in 1:num_categories){
  set.seed(123*j)
  means[j, ] <- runif(num_features, min = 0.5, max = 1.5)
}

d_values <- matrix(0, nrow = num_categories, ncol = num_categories)
for (i in 1:(num_categories - 1)) {
  for (j in (i + 1):num_categories) {
    d_value <- sum(means[i,] * means[j,]) / (sqrt(sum(means[i,]^2)) * sqrt(sum(means[j,]^2)))
    d_values[i, j] <- d_value
    d_values[j, i] <- d_value  # Since the distance matrix is symmetric
  }
}
distance_matrix_s3 <- matrix(0, nrow = num_categories, ncol = num_categories)
for (i in 1:(num_categories - 1)) {
  for (j in (i + 1):num_categories) {
    distance_matrix_s3[i, j] <- d_values[i, j]
    distance_matrix_s3[j, i] <- d_values[j, i]
  }
}
rownames(distance_matrix_s3) <- 1:num_categories
colnames(distance_matrix_s3) <- 1:num_categories

hc <- hclust(as.dist(distance_matrix_s3), method = "single")
s3_tree <- as.phylo(hc)

sd <- c(1/100, 1/10, 1/6, 1/4, 1/3, 1, 3)
std_devs <- list()
for (i in 1:length(sd)){
  std_devs[[i]] <- matrix(sd[i], nrow = num_categories, ncol = num_features)
}

# Generate data for each category-feature combination
s3 <- list()
for (k in 1:length(sd)){
  simulated_data_list <- list()
  for (i in 1:num_categories) {
    simulated_data_i <- matrix(0, nrow = num_samples / num_categories, ncol = num_features)
    for (j in 1:num_features) {
      set.seed(123*j)
      simulated_data_i[, j] <- rnorm(num_samples / num_categories, mean = means[i, j], sd = std_devs[[k]][i, j])
    }
    simulated_data_list[[i]] <- simulated_data_i
  }
  simulated_data <- do.call(rbind, simulated_data_list)
  simulated_data <- cbind(rep(1:num_categories, each = num_samples / num_categories), simulated_data)
  s3[[k]] <- simulated_data
}

rm(distance_matrix, simulated_data, simulated_data_i, simulated_data_list, means, std_devs)












##############################################
#              Normal - 4 types              #
##############################################

# Load required library
library(mvtnorm)

# Set the number of categories, samples, and features
num_categories <- 4
num_samples <- 1000
num_features <- 1000

# Define the means and standard deviations for each category-feature combination
means <- matrix(0, nrow = num_categories, ncol = num_features)

# Generate uniform data for the means
for (j in 1:num_categories){
  set.seed(123*j)
  means[j, ] <- runif(num_features, min = 0.5, max = 1.5)
}

d_values <- matrix(0, nrow = num_categories, ncol = num_categories)
for (i in 1:(num_categories - 1)) {
  for (j in (i + 1):num_categories) {
    d_value <- sum(means[i,] * means[j,]) / (sqrt(sum(means[i,]^2)) * sqrt(sum(means[j,]^2)))
    d_values[i, j] <- d_value
    d_values[j, i] <- d_value  # Since the distance matrix is symmetric
  }
}
distance_matrix_s4 <- matrix(0, nrow = num_categories, ncol = num_categories)
for (i in 1:(num_categories - 1)) {
  for (j in (i + 1):num_categories) {
    distance_matrix_s4[i, j] <- d_values[i, j]
    distance_matrix_s4[j, i] <- d_values[j, i]
  }
}
rownames(distance_matrix_s4) <- 1:num_categories
colnames(distance_matrix_s4) <- 1:num_categories

hc <- hclust(as.dist(distance_matrix_s4), method = "single")
s4_tree <- as.phylo(hc)

sd <- c(1/100, 1/10, 1/6, 1/4, 1/3, 1, 3)
std_devs <- list()
for (i in 1:length(sd)){
  std_devs[[i]] <- matrix(sd[i], nrow = num_categories, ncol = num_features)
}

# Generate data for each category-feature combination
s4 <- list()
for (k in 1:length(sd)){
  simulated_data_list <- list()
  for (i in 1:num_categories) {
    simulated_data_i <- matrix(0, nrow = num_samples / num_categories, ncol = num_features)
    for (j in 1:num_features) {
      set.seed(123*j)
      simulated_data_i[, j] <- rnorm(num_samples / num_categories, mean = means[i, j], sd = std_devs[[k]][i, j])
    }
    simulated_data_list[[i]] <- simulated_data_i
  }
  simulated_data <- do.call(rbind, simulated_data_list)
  simulated_data <- cbind(rep(1:num_categories, each = num_samples / num_categories), simulated_data)
  s4[[k]] <- simulated_data
}

rm(distance_matrix, simulated_data, simulated_data_i, simulated_data_list, means, std_devs)














##############################################
#              Normal - 6 types              #
##############################################

# Load required library
library(mvtnorm)

# Set the number of categories, samples, and features
num_categories <- 6
num_samples <- 996
num_features <- 1000

# Define the means and standard deviations for each category-feature combination
means <- matrix(0, nrow = num_categories, ncol = num_features)

# Generate uniform data for the means
for (j in 1:num_categories){
  set.seed(123*j)
  means[j, ] <- runif(num_features, min = 0.5, max = 1.5)
}

d_values <- matrix(0, nrow = num_categories, ncol = num_categories)
for (i in 1:(num_categories - 1)) {
  for (j in (i + 1):num_categories) {
    d_value <- sum(means[i,] * means[j,]) / (sqrt(sum(means[i,]^2)) * sqrt(sum(means[j,]^2)))
    d_values[i, j] <- d_value
    d_values[j, i] <- d_value  # Since the distance matrix is symmetric
  }
}
distance_matrix_s6 <- matrix(0, nrow = num_categories, ncol = num_categories)
for (i in 1:(num_categories - 1)) {
  for (j in (i + 1):num_categories) {
    distance_matrix_s6[i, j] <- d_values[i, j]
    distance_matrix_s6[j, i] <- d_values[j, i]
  }
}
rownames(distance_matrix_s6) <- 1:num_categories
colnames(distance_matrix_s6) <- 1:num_categories

hc <- hclust(as.dist(distance_matrix_s6), method = "single")
s6_tree <- as.phylo(hc)

sd <- c(1/100, 1/10, 1/6, 1/4, 1/3, 1, 3)
std_devs <- list()
for (i in 1:length(sd)){
  std_devs[[i]] <- matrix(sd[i], nrow = num_categories, ncol = num_features)
}

# Generate data for each category-feature combination
s6 <- list()
for (k in 1:length(sd)){
  simulated_data_list <- list()
  for (i in 1:num_categories) {
    simulated_data_i <- matrix(0, nrow = num_samples / num_categories, ncol = num_features)
    for (j in 1:num_features) {
      set.seed(123*j)
      simulated_data_i[, j] <- rnorm(num_samples / num_categories, mean = means[i, j], sd = std_devs[[k]][i, j])
    }
    simulated_data_list[[i]] <- simulated_data_i
  }
  simulated_data <- do.call(rbind, simulated_data_list)
  simulated_data <- cbind(rep(1:num_categories, each = num_samples / num_categories), simulated_data)
  s6[[k]] <- simulated_data
}

rm(distance_matrix, simulated_data, simulated_data_i, simulated_data_list, means, std_devs)














##############################################
#              Normal - 10 types             #
##############################################
# Load required library
library(mvtnorm)

# Set the number of categories, samples, and features
num_categories <- 10
num_samples <- 1000
num_features <- 1000

# Define the means and standard deviations for each category-feature combination
means <- matrix(0, nrow = num_categories, ncol = num_features)

# Generate uniform data for the means
for (j in 1:num_categories){
  set.seed(123*j)
  means[j, ] <- runif(num_features, min = 0.5, max = 1.5)
}

d_values <- matrix(0, nrow = num_categories, ncol = num_categories)
for (i in 1:(num_categories - 1)) {
  for (j in (i + 1):num_categories) {
    d_value <- sum(means[i,] * means[j,]) / (sqrt(sum(means[i,]^2)) * sqrt(sum(means[j,]^2)))
    d_values[i, j] <- d_value
    d_values[j, i] <- d_value  # Since the distance matrix is symmetric
  }
}
distance_matrix_s10 <- matrix(0, nrow = num_categories, ncol = num_categories)
for (i in 1:(num_categories - 1)) {
  for (j in (i + 1):num_categories) {
    distance_matrix_s10[i, j] <- d_values[i, j]
    distance_matrix_s10[j, i] <- d_values[j, i]
  }
}
rownames(distance_matrix_s10) <- 1:num_categories
colnames(distance_matrix_s10) <- 1:num_categories

hc <- hclust(as.dist(distance_matrix_s10), method = "single")
s10_tree <- as.phylo(hc)

sd <- c(1/100, 1/10, 1/6, 1/4, 1/3, 1, 3)
std_devs <- list()
for (i in 1:length(sd)){
  std_devs[[i]] <- matrix(sd[i], nrow = num_categories, ncol = num_features)
}

# Generate data for each category-feature combination
s10 <- list()
for (k in 1:length(sd)){
  simulated_data_list <- list()
  for (i in 1:num_categories) {
    simulated_data_i <- matrix(0, nrow = num_samples / num_categories, ncol = num_features)
    for (j in 1:num_features) {
      set.seed(123*j)
      simulated_data_i[, j] <- rnorm(num_samples / num_categories, mean = means[i, j], sd = std_devs[[k]][i, j])
    }
    simulated_data_list[[i]] <- simulated_data_i
  }
  simulated_data <- do.call(rbind, simulated_data_list)
  simulated_data <- cbind(rep(1:num_categories, each = num_samples / num_categories), simulated_data)
  s10[[k]] <- simulated_data
}

rm(distance_matrix, simulated_data, simulated_data_i, simulated_data_list, means, std_devs)
