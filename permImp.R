#' permImp
#'
#' @description Perform permutation importace on encoder mpdel data.
#'
#' @param encoderModel output from autoBuild function.
#' @param num_permutations The number of permutations to preform
#' @param MSE If TRUE, then use MSE. If FALSE, then use RMSE

permImp <- function(encoderModel, num_permutations = 4, MSE = TRUE) {
  encoder <- encoderModel[[1]] # get encoder model
  x_test <- encoderModel[[2]] # get test data

  # Encode the original test data
  original_encoded_imgs <- predict(encoder, x_test)

  # Initialize matrices to store importance
  varimp <- matrix(0, nrow = 784, ncol = 32)  # Initialize with 0 for summing

  # Permutation importance loop
  for (i in 1:dim(x_test)[2]) { # Loop over each feature
    print(paste0('dimension ', i, ' computed'))

    for (n in 1:num_permutations) {
      # Copy the original test data
      permuted_x_test <- x_test

      # Permute the values in the i-th column (feature)
      permuted_x_test[, i] <- sample(permuted_x_test[, i])

      # Encode the permuted data
      permuted_encoded_imgs <- predict(encoder, permuted_x_test, verbose = F)

      for(j in 1:32){
        varimp[i,j] <- varimp[i,j] + errorMetric(original_encoded_imgs[,j], permuted_encoded_imgs[,j], MSE = MSE)
      }
    }
  }

  # Average the importance over the number of permutations
  varimp <- varimp / num_permutations

  # Store results in a list
  all_dimensions_importance <- list()

  for(j in 1:32){
    # Convert to matrix for each dimension
    feature_importance_matrix <- matrix(varimp[,j], nrow = 28, ncol = 28)

    # Convert the matrix to a data frame suitable for ggplot
    feature_importance_df <- as.data.frame(as.table(feature_importance_matrix))
    all_dimensions_importance[[as.character(j)]] <- feature_importance_df
  }

  return(all_dimensions_importance)
}








#
# permImp <- function(encoderModel, numPerm = 4) {
#   encoder <- encoderModel[[1]] # get encoder model
#   x_test <- encoderModel[[2]] # get test data
#
#   # Encode the original test data
#   original_encoded_imgs <- predict(encoder, x_test)
#
#   # Initialize matrices to store importance
#   varimp <- matrix(NA, nrow = 784, ncol = 32)
#
#   # Permutation importance loop
#   for (i in 1:dim(x_test)[2]) { # Loop over each feature
#     print(paste0('dimension ', i, ' computed'))
#
#     for (n in 1:numPerm) {
#       print(paste0('PERMUTATION ', n, ' BEING COMPUTED'))
#     # Copy the original test data
#     permuted_x_test <- x_test
#
#     # Permute the values in the i-th column (feature)
#     permuted_x_test[, i] <- sample(permuted_x_test[, i])
#
#     # Encode the permuted data
#     permuted_encoded_imgs <- predict(encoder, permuted_x_test, verbose = F)
#
#     for(j in 1:32){
#       varimp[i,j] <- errorMetric(original_encoded_imgs[,j], permuted_encoded_imgs[,j], MSE = F)
#     }
#     }
#   }
#
#   # Store results in a list
#   all_dimensions_importance <- list()
#
#   for(j in 1:32){
#     # Average over dimensions
#     feature_importance_matrix <- matrix(varimp[,j], nrow = 28, ncol = 28)
#
#     # Convert the matrix to a format suitable for ggplot
#     feature_importance_df <- as.data.frame(as.table(feature_importance_matrix))
#     all_dimensions_importance[[as.character(j)]] <- feature_importance_df
#   }
#
#   return(all_dimensions_importance)
# }








#
# permImp <- function(encoderModel){
#
#   encoder<- encoderModel[[1]] # get encoder model
#   x_test <- encoderModel[[2]] # get test data
#
#   # Encode the original test data
#   original_encoded_imgs <- predict(encoder, x_test)
#
#   # Initialise a vector to store the importance of each feature
#   feature_importance <- numeric(length = dim(x_test)[2]) # 784 for MNIST
#
#   varimp <- matrix(NA,nrow = 784, ncol = 32)
#
#   # Permutation importance loop
#   for (i in 1:dim(x_test)[2]) { # Loop over each feature
#     print(paste0('dimension ', i, ' computed'))
#     # Copy the original test data
#     permuted_x_test <- x_test
#
#     # Permute the values in the i-th column (feature)
#     permuted_x_test[, i] <- sample(permuted_x_test[, i])
#
#     # Encode the permuted data
#     permuted_encoded_imgs <- predict(encoder, permuted_x_test, verbose = F)
#
#
#     for(j in 1:32){
#       varimp[i,j] <- errorMetric(original_encoded_imgs[,j], permuted_encoded_imgs[,j], MSE = F)
#       }
#     }
#
#
#
#   for(j in 1:32){
#     # plot --------------------------------------------------------------------
#     feature_importance_matrix <- matrix(varimp[,j], nrow = 28, ncol = 28)
#
#     # Convert the matrix to a format suitable for ggplot
#     feature_importance_df <- as.data.frame(as.table(feature_importance_matrix))
#     lim <-   range(feature_importance_df$Freq)
#     lim_final <-  range(labeling::rpretty(lim[1], lim[2]))
#
#
#
#     # Plotting
#     p <- ggplot(feature_importance_df, aes(Var1, Var2, fill = Freq)) +
#       geom_tile() +
#       scale_fill_gradient(low = "white",
#                           high = "red",
#                           name = 'MSE',
#                           limits = lim_final,
#                           guide = guide_colorbar(
#                             frame.colour = "black",
#                             ticks.colour = "black"
#                           )) +
#       theme_void() +
#       labs(x = "Pixel Row", y = "Pixel Column", title = paste0('Dimension ', j)) +
#       coord_fixed()
#
#
# # Average over dimensions -------------------------------------------------
#
#     # average
#     row_averages <- rowMeans(varimp)
#     feature_importance_matrix <- matrix(row_averages, nrow = 28, ncol = 28)
#
#     # Convert the matrix to a format suitable for ggplot
#     feature_importance_df <- as.data.frame(as.table(feature_importance_matrix))
#
#     # Plotting
#     pp <- ggplot(feature_importance_df, aes(Var1, Var2, fill = Freq)) +
#       geom_tile() +
#       scale_fill_gradient(low = "white",
#                           high = "red",
#                           name = 'MSE',
#                           #limits = c(0, 0.0132),
#                           guide = guide_colorbar(
#                             frame.colour = "black",
#                             ticks.colour = "black"
#                           )) +
#       theme_void() +
#       labs(x = "Pixel Row", y = "Pixel Column", title = 'Average') +
#       coord_fixed()
#   }
# }
