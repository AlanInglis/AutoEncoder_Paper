library(keras)
library(png)
library(fields)

set.seed(2468)
tensorflow::tf$random$set_seed(2468)

# Load images -------------------------------------------------------------

# load and preprocess images function
load_and_preprocess_image <- function(image_path) {
  img <- png::readPNG(image_path)
  img <- img[,,1:3] # Keep only the RGB channels, discard alpha
  return(img)
}

image_directory <- "/Users/alaninglis/Desktop/autoencoder_stuff/MNIST_speech/col_32"
image_files <- list.files(path = image_directory, pattern = "\\.png$", full.names = TRUE)

#set.seed(2468)
#image_files <- image_files[1:50]#sample(image_files, size = 10)

# load images
pb <- txtProgressBar(min = 0, max = length(image_files), style = 3)
images <- lapply(image_files, function(file) {
  img <- load_and_preprocess_image(file)
  setTxtProgressBar(pb, which(image_files == file))
  return(img)
})
close(pb)


# -------------------------------------------------------------------------
# Smooth images -----------------------------------------------------------


# Function to smooth and average the channels of a single image
smooth_and_average_channels <- function(image_data) {
  # Initialize an empty matrix to store the averaged smoothed channels
  smoothed_image = matrix(0, nrow = 32, ncol = 32)

  # Define the grid coordinates
  x_grid <- rep(1:32, each = 32)
  y_grid <- rep(1:32, times = 32)

  # Convert the 3D image data array to a list of vectors, one for each channel
  channel_vectors <- lapply(1:3, function(channel) as.vector(image_data[,,channel]))

  # Apply smooth.2d to each color channel and then average the results
  for (i in 1:3) {
    # Apply smooth.2d to the vector with coordinates
    smoothed_vector <- smooth.2d(Y = channel_vectors[[i]], x = cbind(x_grid, y_grid), nrow = 32, ncol = 32, theta = 0.1, surface = TRUE)$z

    # Accumulate the smoothed results
    smoothed_image <- smoothed_image + smoothed_vector / 3
  }

  return(smoothed_image)
}

#  apply the function to each image
# Create a progress bar
pb <- txtProgressBar(min = 0, max = length(images), style = 3)

# Initialize a counter to track the progress
current_progress <- 0

# Define a wrapper function to update the progress bar as each image is processed
process_image_with_progress <- function(image) {
  result <- smooth_and_average_channels(image)
  # Increment the progress counter
  current_progress <<- current_progress + 1
  # Update the progress bar based on the current progress
  setTxtProgressBar(pb, current_progress)
  return(result)
}

# Apply the function to all images, with progress update
smoothed_images <- lapply(images, process_image_with_progress)


# Close the progress bar
close(pb)


# display the smoothed images
par(mfrow = c(1, 5))
for (i in 5001:5005) {
  image(1:32, 1:32, smoothed_images[[i]], col = viridis(256), main = paste("Image", i))
}


# preprocess data ---------------------------------------------------------

images <- smoothed_images
# Apply min-max normalisation to each image
images_normalized <- lapply(images, function(img) {
  img_min <- min(img)
  img_max <- max(img)
  (img - img_min) / (img_max - img_min)
})



# Ensure the data is reshaped properly:
images_flattened <- lapply(images_normalized, function(img) {
  array_reshape(img, c(1, prod(dim(img))))  # Reshape to include batch dimension
})

# Convert to matrix, ensuring it maintains a 2D shape
image_matrix <- do.call(rbind, images_flattened)

# Check the shape of the inputs
cat("Shape of the image matrix: ", dim(image_matrix), "\n")

#  split
set.seed(2468)
indices <- sample(1:nrow(image_matrix), nrow(image_matrix))
split_index <- floor(0.8 * length(indices))
train_indices <- indices[1:split_index]
test_indices <- indices[(split_index + 1):length(indices)]

# Create training and test
train_images <- image_matrix[train_indices, ]
test_images <- image_matrix[test_indices, ]
image_files <- basename(image_files)
labels <- sapply(strsplit(image_files, "_"), function(x) as.integer(x[1]))
y_test <- labels[test_indices]
x_test <- test_images


#  confirm the input shape:
cat("Shape of train_images: ", dim(train_images), "\n")
cat("Shape of test_images: ", dim(test_images), "\n")


# Autoencoder -------------------------------------------------------------


input_size <- 32 * 32

input_img <- layer_input(shape = c(input_size))

# Encoder
encoded <- input_img %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 450, activation = "relu") %>% # Additional layer
  layer_dense(units = 384, activation = "relu") %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 192, activation = "relu") %>% # Additional layer
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 96, activation = "relu") %>%  # Additional layer
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu")

# Decoder
decoded <- encoded %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 96, activation = "relu") %>%  # Additional layer
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 192, activation = "relu") %>% # Additional layer
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 384, activation = "relu") %>%
  layer_dense(units = 450, activation = "relu") %>% # Additional layer
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = input_size, activation = "sigmoid")


# Autoencoder model
autoencoder <- keras_model(inputs = input_img, outputs = decoded)

autoencoder %>% compile(
  loss =  'binary_crossentropy',
  optimizer = 'adam'
)

# Summary of the model
summary(autoencoder)

# Train the model
autoencoder %>% fit(
  train_images, train_images,
  epochs = 100,
  batch_size = 256,
  shuffle = T,
  validation_data = list(test_images, test_images)
)



# -------------------------------------------------------------------------
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)
# Predicting the reconstructed images from the test set
reconstructed_images <- autoencoder %>% predict(test_images)
# Reshape the flattened images back to 32x32 format
original_reshaped <- array_reshape(test_images, c(length(test_indices), 32, 32))
reconstructed_reshaped <- array_reshape(reconstructed_images, c(length(test_indices), 32, 32))
# Define the number of images to display
num_images_to_display <- 10


# Function to flip the matrix vertically
flip_matrix_vertically <- function(mat) {
  mat[, seq(ncol(mat), 1)]
}

# Set up the plotting window
par(mfrow = c(2, 10), mar = c(1, 1, 1, 1))

# Plotting original and reconstructed images side by side
for (i in 1:10) {
  # Plot original image
  image(1:32, 1:32, flip_matrix_vertically(original_reshaped[i,,]), col = viridis(255), main = paste("Original", i), axes = FALSE)

  # Plot reconstructed image
  image(1:32, 1:32, flip_matrix_vertically(reconstructed_reshaped[i,,]), col = viridis(255), main = paste("Reconstructed", i), axes = FALSE)
}


# -------------------------------------------------------------------------


encoded_imgs <- predict(encoder, x_test)
decoded_imgs <- predict(autoencoder, x_test)

# Plot Encoded Space
col_sds <- apply(encoded_imgs, 2, 'sd')
o <- order(col_sds, decreasing = TRUE)
encoded_df <- as.data.frame(encoded_imgs[,o[1:2]])
encoded_df$digit <- y_test

ggplot(encoded_df, aes(x = V1, y = V2, color = as.factor(digit))) +
  geom_point() +
  scale_color_discrete(name = "Digit") +
  labs(title = "",
       x = paste("Dimension",o[1]),
       y = paste("Dimension",o[2])) +
  theme_bw()


# no idea why 2 rows are identical
df <- as.data.frame(encoded_imgs)
duplicates <- duplicated(df) | duplicated(df, fromLast = TRUE)
df[duplicates, ]
# Display duplicate rows
# df[duplicates, ] <- df[duplicates, ] + 0.000001
# df[5752, ] = df[5752, ] + 0.000001
#
#
#
# set.seed(2468)
# # Checking the number of unique rows
# unique_encoded_imgs <- unique(df)
# cat("Original rows: ", nrow(encoded_imgs), "\n")
# cat("Unique rows: ", nrow(unique_encoded_imgs), "\n")

# Proceed with unique rows
tsne_results <- Rtsne::Rtsne(df, dims = 2, perplexity = 30, verbose = TRUE)
tsne_data <- as.data.frame(tsne_results$Y)
tsne_data$digit <- y_test

ggplot(tsne_data, aes(x = V1, y = V2,  label = as.factor(digit), color = as.factor(digit))) +
  #geom_point(alpha = 0.5) +
  geom_text(alpha = 0.5) +
  labs(title = "",
       x = "t-SNE 1",
       y = "t-SNE 2",
       color = "Digit") +
  theme_bw() +
  theme(legend.position = 'none')



# UMAP --------------------------------------------------------------------

library(umap)


# Perform UMAP dimensionality reduction
umap_results <- umap(df, n_neighbors = 15, min_dist = 0.1, n_components = 2)

# Convert UMAP results to a data frame
umap_data <- as.data.frame(umap_results$layout)
umap_data$digit <- y_test

# Plot the UMAP results
ggplot(umap_data, aes(x = V1, y = V2, label = as.factor(digit), color = as.factor(digit))) +
  #geom_point(alpha = 0.5) +
  geom_text(alpha = 0.5) +
  labs(title = "",
       x = "UMAP 1",
       y = "UMAP 2",
       color = "Digit") +
  theme_bw() +
  theme(legend.position = 'none')

# -------------------------------------------------------------------------


library(ggplot2)
library(autoImp)
library(dplyr)
library(shiny)



#' Variable Importance for Encoded Autoencoder
#'
#' @description
#' Computes the importance of each feature in the encoding process
#' of the MNIST data set using a provided encoder model. It uses permutation-based
#' importance measurement.
#'
#' @details
#' The function computes the permutation importance
#' by permuting each feature across a specified number
#' of permutations and calculating the impact on the encoding output.
#'
#'
#' @param encoder An encoder model
#' @param testData The test data used to build the model
#' @param num_permutations The number of permutations to use for calculating
#'        importance. Default is 4.
#' @param metric  Character indicating the type of error metric to return.
#' If \code{metric = 'MSE'}, (default), the function returns the Mean Squared Error (MSE).
#' If \code{metric = 'RMSE'}, it returns the Root Mean Squared Error (RMSE).
#'
#' @return A list containing two elements:
#' \itemize{
#'   \item{"Vimp"}{ - A list of data frames, each representing the feature importance
#'           matrix for one encoded dimension.}
#'   \item{"selectedNumber"}{ - The number selected in the encoder model, if any.}
#' }
#'
#'
#' @examples
#' # Train model using only digit 3 for 10 epochs
#' encoderModel <- mnistAuto(numSel = 3, epochs = 10)
#' # Calculate importance
#' vimpEncoded(encoderModel)
#'
#' @export


vimpEncodedtt <- function(encoder, testData, num_permutations = 4, metric = "MSE") {


  encoder <- encoder
  x_test <- testData

  # Encode the original test data
  original_encoded_imgs <- predict(encoder, x_test, verbose = FALSE)

  # Initialize matrices to store importance
  varimp <- matrix(0, nrow = 1024, ncol = encoding_dim) # Initialize with 0 for summing

  # Set up the progress bar
  total_iterations <- dim(x_test)[2]
  pb <- txtProgressBar(min = 0, max = total_iterations, style = 3)

  # Permutation importance loop
  for (i in 1:dim(x_test)[2]) { # Loop over each feature

    # progress bar
    setTxtProgressBar(pb, i)

    for (n in 1:num_permutations) {
      # Copy the original test data
      permuted_x_test <- x_test

      # Permute the values in the i-th column (feature)
      permuted_x_test[, i] <- sample(permuted_x_test[, i])

      # Encode the permuted data
      permuted_encoded_imgs <- predict(encoder, permuted_x_test, verbose = F)

      for (j in 1:encoding_dim) {
        varimp[i, j] <- varimp[i, j] + errorMetric(original_encoded_imgs[, j], permuted_encoded_imgs[, j], metric = "MSE")
      }
    }
  }

  # Close the progress bar
  close(pb)

  # Average the importance over the number of permutations
  varimp <- varimp / num_permutations

  # Store results in a list
  all_dimensions_importance <- list()

  for (j in 1:encoding_dim) {
    # Convert to matrix for each dimension
    feature_importance_matrix <- matrix(varimp[, j], nrow = 32, ncol = 32)


    # Convert the matrix to a data frame suitable for ggplot
    feature_importance_df <- as.data.frame(as.table(feature_importance_matrix))
    all_dimensions_importance[[as.character(j)]] <- feature_importance_df
  }

  myList <- list(
    "Vimp" = all_dimensions_importance
  )

  return(myList)
}



# -------------------------------------------------------------------------
# -------------------------------------------------------------------------



vimpDecodedNewtt <- function(encoder,
                             x_test,
                             y_test,
                             autoencoder,
                             digits,
                             num_permutations = 4,
                             encodingDim = 32,
                             errorMetric = "MSE",
                             letter = FALSE) {



  encoded_input <- keras::layer_input(shape = c(encodingDim))
  decoder_layer <- autoencoder$layers[[11]] # decoder layer
  decoder <- keras::keras_model(inputs = encoded_input, outputs = decoder_layer(encoded_input))


  permImpInternal <- function(encoder, autoencoder, x_test, y_test, digit, num_permutations = 4, em = errorMetric) {
    digit_indices <- which(y_test == digit)
    digit_x_test <- x_test[digit_indices, ]

    # original
    original_encoded_imgs <- predict(encoder, digit_x_test, verbose = F)
    decoded_imgs <- predict(decoder, original_encoded_imgs, verbose = F)

    varimp <- matrix(0, nrow = encodingDim, ncol = 1)

    for (i in 1:encodingDim) {
      for (n in 1:num_permutations) {
        # OG
        permuted_encoded_imgs <- original_encoded_imgs

        # Permute
        permuted_encoded_imgs[, i] <- sample(permuted_encoded_imgs[, i])

        # Decode permuted encoded images
        permuted_decoded_imgs <- predict(decoder, permuted_encoded_imgs, verbose = F)

        # error
        varimp[i] <- varimp[i] + errorMetric(decoded_imgs, permuted_decoded_imgs, metric = "MSE")
      }
    }
    # Average importance over the number of permutations
    varimp <- varimp / num_permutations
    return(varimp)
  }

  # progress bar setup
  pb <- txtProgressBar(min = 0, max = length(digits), style = 3)

  all_digits_importance <- list()
  for (digit_idx in 1:length(digits)) {
    digit <- digits[digit_idx]
    # Update progress bar
    setTxtProgressBar(pb, digit_idx)

    digit_importance <- permImpInternal(encoder, autoencoder, x_test, y_test, digit)
    all_digits_importance[[as.character(digit)]] <- digit_importance
  }

  close(pb)


  importance_df <- data.frame(
    Digit = rep(digits, each = encodingDim),
    EncodedDimension = rep(1:encodingDim, times = length(digits)),
    Importance = unlist(all_digits_importance)
  )


  return(importance_df)
}

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------



autoLmtt <- function(encoder, x_test, threshold = NULL){

  # encoder and test data
  encoder <- encoder
  x_test <- x_test

  # encoder <- autoencoder$encoder
  # x_test <- autoencoder$testData

  # predict
  encoded_imgs <- predict(encoder, x_test, verbose = F)

  # covert to df
  x_test_df <- as.data.frame(x_test)
  encoded_imgs_df <- as.data.frame(encoded_imgs)

  # rename cols
  colnames(encoded_imgs) <- paste0("encoded_dim_", 1:encoding_dim) # 32 latent dims


  # Initialize matrix to hold the coefficients using 32 latent dims
  model_summaries <- matrix(0, nrow = 1024, ncol = encoding_dim)


  # Set up the progress bar
  total_iterations <- dim(x_test)[2]
  pb <- txtProgressBar(min = 0, max = ncol(encoded_imgs_df), style = 3)


  # Loop through each encoded dimension
  for (i in 1:ncol(encoded_imgs_df)) {
    #print(paste0("Processing dimension ", i))
    # progress bar
    setTxtProgressBar(pb, i)


    for(j in 1:ncol(x_test_df)){
      # Fit linear regression model: Response ~ All Pixels
      model <- lm(encoded_imgs_df[, i] ~ x_test_df[,j])
      # get slope
      model_summaries[j, i] <- model$coefficients[2]
    }
  }

  close(pb)
  # Convert all NA values in model_summaries to 0
  model_summaries[is.na(model_summaries)] <- 0


  # Set all negative values to -1, positive values to +1, and keep zeros unchanged
  model_summaries_adjusted <- model_summaries
  model_summaries_adjusted[model_summaries < 0] <- -1
  model_summaries_adjusted[model_summaries > 0] <- 1

  # threshold values

  if(!is.null(threshold)){
    # Set values below the threshold to zero
    model_summaries_adjusted[abs(model_summaries_adjusted) < threshold] <- 0
  }

  return(model_summaries_adjusted)

}


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------



decodedLmt <- function(encodedVimp,
                       decodedVimp,
                       lmSummary,
                       digit,
                       topX = NULL,
                       sortVimp = FALSE,
                       filterZero = FALSE) {



  # Decoded plot ------------------------------------------------------------

  # -------------------------------------------------------------------------
  # Filter data for the selected digits
  data_digit <- subset(decodedVimp, Digit == digit)

  if (filterZero) {
    # remove rows with zeros
    data_digit <- subset(data_digit, Importance != 0)
  }

  if (sortVimp) {
    # Select the top x values by the absolute magnitude of ImportanceDifference
    data_digit <- data_digit |>
      arrange(desc(abs(Importance)))
  }

  if (!is.null(topX)) {
    data_digit <- data_digit |>
      head(topX) # top x selected
  }

  # Turn encoded list of dfs into matrix ------------------------------------
  vimp_matrix <- matrix(nrow = 1024, ncol = encoding_dim)

  # Loop through each data frame in the list
  for (i in 1:encoding_dim) {
    # Extract the 'Freq' column from each data frame
    freq_column <- encodedVimp$Vimp[[as.character(i)]]$Freq
    # Fill the corresponding column in the matrix with the 'Freq' values
    vimp_matrix[, i] <- freq_column
  }


  # overlay lm summary onto vimps -------------------------------------------

  vimp_lm_mat <- vimp_matrix * lmSummary


  # match lmsum to decoded ------------------------------------------------

  # Vector containing the numbers of the data frames you want to keep
  myVec <- data_digit$EncodedDimension

  # Filter the lm summaries to match
  colnames(vimp_lm_mat) <- c(1:encoding_dim)
  filtered_vimp <- vimp_lm_mat[,myVec]
  colnames(filtered_vimp) <- as.character(myVec)

  # -------------------------------------------------------------------------
  # decoded plot set up -----------------------------------------------------

  # Reorder the factor levels based on subset
  data_digit$EncodedDimension <- factor(data_digit$EncodedDimension,
                                        levels = rev(data_digit$EncodedDimension))




  # plot
  # p_dec <- ggplot(data_digit, aes(
  #   x = as.factor(EncodedDimension),
  #   y = Importance,
  #   fill = Importance > 0
  # )) +
  #   geom_bar(stat = "identity", position = "dodge") +
  #   coord_flip() +
  #   scale_fill_manual(values = c("TRUE" = "firebrick", "FALSE" = "steelblue")) +
  #   labs(x = "Encoded Dimension", y = "Importance") +
  #   theme_bw() +
  #   #ggtitle(paste0('Importance for digit: ', digit)) +
  #   theme(legend.position = "none", aspect.ratio = 1)
  p_dec <- ggplot(data_digit, aes(
    x = EncodedDimension,
    y = Importance,
    fill = Importance
  )) +
    geom_bar(stat = "identity", position = "dodge") +
    coord_flip() +
    scale_fill_gradient(
      low = "steelblue",
      high = "red",
      name = "Encoded \nDimension \nVimp",
      guide = guide_colorbar(
        frame.colour = "black",
        ticks.colour = "black"
      )
    ) +
    # scale_fill_manual(values = c("TRUE" = "firebrick", "FALSE" = "steelblue")) +
    labs(x = "Encoded Dimension", y = "Importance") +
    theme_bw() +
    #ggtitle(paste0('Importance for letter: ', digit)) +
    theme(legend.position = "none", aspect.ratio = 1)







  # -------------------------------------------------------------------------

  # filter plots with only zeros
  if (filterZero) {
    # ID columns that are not all zeros
    columns_to_keep <- apply(vimp_matrix, 2, function(x) any(x != 0))
    colnames(vimp_matrix) <- c(1:encoding_dim)
    # Subset the matrix to remove columns that are all zeros
    vimp_matrix <- vimp_matrix[, columns_to_keep]
  }




  # plot heatmaps -----------------------------------------------------------
  max_val <- max(filtered_vimp)
  min_val <- min(filtered_vimp)

  # plotting function
  plotFun <- function(data, legendTitle){
    # Prepare a list to hold ggplot objects
    plots <- list()

    for (i in 1:ncol(data)) {
      mat <- matrix(data[, i], ncol = 32, nrow = 32)
      mat <- t(apply(mat, 2, rev))
      # mat <- t(apply(mat, 2, rev))
      mat <- mat[nrow(mat):1, ]  # Flip the image vertically
      # Convert the matrix to a data frame for ggplot
      mat_long <- reshape2::melt(mat)

      # Create a plot for this matrix
      p <- ggplot(mat_long, aes(x = Var1, y = Var2, fill = value)) +
        geom_tile() +
        scale_y_discrete(limits = rev(levels(mat_long$Var2))) +
        scale_fill_gradient2(
          low = 'blue',
          high = 'red',
          mid = 'white',
          #limits = c(min_val, max_val),
          name = legendTitle,
          guide = guide_colorbar(
            order = 1,
            frame.colour = "black",
            ticks.colour = "black"
          ),
          oob = scales::squish
        ) +
        ggtitle(paste(myVec[i])) +
        theme_void() +
        theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
        theme(plot.title = element_text(hjust = 0.5)) +
        coord_fixed()

      # Add the plot to the list
      plots[[i]] <- p
    }
    # Get legend
    legend <- cowplot::get_legend(plots[[2]])

    # remove legend from all plots
    all_plots_no_leg <- lapply(plots, function(x) x + theme(legend.position = "none"))

    myList <- list(plots = all_plots_no_leg, legend = legend)
    return(myList)
  }

  pp <- plotFun(data = filtered_vimp, legendTitle = 'Vimp \nDirection')




  # plot everything on grid -------------------------------------------------

  #plot on a grid
  n <- length(pp$plots)
  nRow <- floor(sqrt(n))
  all_plot_grid <- gridExtra::arrangeGrob(grobs = pp$plots, nrow = nRow)
  suppressMessages(
    gridExtra::grid.arrange(p_dec, all_plot_grid,
                           # pp$legend,
                            ncol = 3,
                            widths = c(5, 5,0.5)
    )
  )



}








# -------------------------------------------------------------------------

# -------------------------------------------------------------------------


shinyDecodedLmt <- function(decodedVimp, lmSummary, encodedVimp) {
  # UI ----------------------------------------------------------------------


  ui <- fluidPage(
    titlePanel("MNIST Autoencoder Analysis"),

    # Row for digit selection and threshold
    fluidRow(
      column(
        4,
        selectInput("digit", "Select digit:", choices = 0:9, selected = 0)
      )
    ),

    # Row for additional controls
    fluidRow(
      column(
        4,
        numericInput("topX", "Number of Top Importances:", value = 5, min = 1, max = 100)
      ),
      column(
        4,
        checkboxInput("filterZeros", "Filter Zero Importance Differences", value = TRUE)
      ),
      column(
        4,
        checkboxInput("sortVimp", "Sort by Importance", value = TRUE)
      )
    ),

    # Row for the plot
    fluidRow(
      column(
        12,
        plotOutput("combinedPlot", width = "100%", height = "600px")
      )
    )
  )



  # server logic ------------------------------------------------------------


  server <- function(input, output) {
    output$combinedPlot <- renderPlot({
      decodedLmt(
        decodedVimp = decodedVimp,
        lmSummary = lmSummary,
        encodedVimp = encodedVimp,
        digit = input$digit,
        topX = input$topX,
        filterZero = input$filterZeros,
        sortVimp = input$sortVimp
      )
    })
  }


  # launch ------------------------------------------------------------------

  shinyApp(ui = ui, server = server)
}


# -------------------------------------------------------------------------


# -------------------------------------------------------------------------

encoding_dim = 32
set.seed(2468)
ae_vimp_enc <- vimpEncodedtt(encoder = encoder, testData = x_test, num_permutations = 4)
toneR::tone(2)
set.seed(2468)
ae_vimp_dec <- vimpDecodedNewtt(encoder = encoder,
                                x_test =  x_test,
                                y_test = y_test,
                                autoencoder = autoencoder,
                                digits = c(0:9),
                                num_permutations = 4,
                                encodingDim = 32)
set.seed(2468)
lm_sum_2<- autoLmtt(encoder = encoder,
                    x_test =  x_test)

save.image(file = 'mnist_speech.RData')
load("/Users/alaninglis/Desktop/Autoencoder Paper/saved_objects/mnist_speech.RData")


plotEncoded(ae_vimp_enc, legendName = 'Vimp')
plotDecoded(data = ae_vimp_dec) + ggtitle('')  +  theme(
  axis.title = element_text(size = 10),
  axis.text.x = element_text(size = 8,
                             angle = 90,
                             hjust = 1,
                             vjust = 1),
  axis.text.y = element_text(size = 10)
)
shinyDecodedLmt(decodedVimp = ae_vimp_dec, lmSummary = lm_sum_2, encodedVimp = ae_vimp_enc)

plotAvgDecoded(ae_vimp_dec)
# -------------------------------------------------------------------------

decodedLmt(encodedVimp = ae_vimp_enc,
           decodedVimp = ae_vimp_dec,
           lmSummary = lm_sum_2,
           digit = 8,
           sortVimp = T,
           filterZero = T,
           topX = 10)







# ] -----------------------------------------------------------------------

library(tidyr)
library(dplyr)


# Reshape data to wide format
ae_vimp_dec_wide <- ae_vimp_dec %>%
  pivot_wider(names_from = EncodedDimension, values_from = Importance, names_prefix = "Dim") %>%
  replace(is.na(.), 0)  # Replace NA values with 0 if any


# Compute the distance matrix
dist_matrix <- dist(ae_vimp_dec_wide[-1], method = "euclidean")  # Exclude the first column as it's the Digit identifier

# Hierarchical clustering using Ward's method
hc <- hclust(dist_matrix, method = "complete")

# Set labels directly corresponding to the digits (if they are already from 0 to 9)
hc$labels <- as.character(ae_vimp_dec_wide$Digit)  # Ensure that labels are from 0 to 9

# If the row names or labels aren't set or need to be corrected to start from 0:
hc$labels <- as.character(0:9)  # Only correct if you have exactly 10 labels corresponding from 0 to 9


# Plot the dendrogram
plot(hc, main = "Hierarchical Clustering of Digits based on Dimension Importance")




# Set up the plot parameters
par(mar = c(5, 4, 4, 2) + 0.1)  # Adjust the margins if necessary

# Plot the dendrogram without the y-axis
plot(hc, main = "Hierarchical Clustering of Digits", xlab = "", ylab = "",
     sub = "", yaxt = 'n', labels = F, hang = -1,
     labels.cex = 0.8, labels.col = "blue")

# Adding colored labels manually if desired
if(!is.null(hc$labels)) {
  text(x = seq_along(hc$labels), y = -1, labels = hc$labels, col = "blue", xpd = TRUE, srt = 45, adj = 1)
}

# If you want to add color to branches based on a cut
rect.hclust(hc, k = 4, border = "red")  # Adds colored rectangles for 4 clusters


# -------------------------------------------------------------------------


library(dendextend)
library(dendextend)

# Create a dendrogram object from hclust
dend <- as.dendrogram(hc)

# Define a vector of vibrant colors
vibrant_colors <- c("firebrick", "lightgreen", "steelblue", "chartreuse3")

# Color branches by cluster with custom vibrant colors
dend <- color_branches(dend, k = 4, col = vibrant_colors)

# Plot the dendrogram with customized branch colors
plot(dend, main = "")

# Generate a gradient of colors
gradient_colors <- colorRampPalette(c("#FF5733", "#3357FF"))(5)

# Apply gradient colors to branches
dend <- color_branches(dend, k = 5, col = gradient_colors)

# Plot the dendrogram
plot(dend, main = "Hierarchical Clustering with Gradient Colors")


# -------------------------------------------------------------------------


# Applying cutree to see the cluster assignments
cluster_assignments <- cutree(hc, k = 4)

# View the assignments
print(cluster_assignments)

# Optionally, check the heights
print(hc$height)


# Manually setting colors based on cluster assignments
manual_colors <- vibrant_colors[cluster_assignments]




# -------------------------------------------------------------------------



# Cutting the tree into k clusters
k <- 4  # You can choose k based on the dendrogram or domain knowledge
clusters <- cutree(hc, k)

# Adding cluster assignments back to the data
ae_vimp_dec_wide$Cluster <- as.factor(clusters)

# Summarize the data by cluster
cluster_summary <- ae_vimp_dec_wide %>%
  group_by(Cluster) %>%
  summarise(across(starts_with("Dim"), mean, na.rm = TRUE))  # Calculating mean importance per dimension in each cluster

print(cluster_summary)

# -------------------------------------------------------------------------


library(keras)


encoder_new <- keras_model(inputs = autoencoder$input, outputs = get_layer(autoencoder, "dense")$output)

encoded_representations <- predict(encoder_new, test_images)


# Convert the encoded representations to a data frame
encoded_representations_df <- as.data.frame(encoded_representations)

# Add the digit labels
encoded_representations_df$Digit <- y_test

# Add a SampleID column to keep track of the samples
encoded_representations_df$SampleID <- 1:nrow(encoded_representations_df)

# Reorder columns to have SampleID and Digit at the beginning
encoded_representations_df <- encoded_representations_df %>%
  select(SampleID, Digit, everything())

# Print the resulting data frame
print(encoded_representations_df)



# -------------------------------------------------------------------------

# Assuming you already have the `ae_vimp_dec` data frame with permutation importance data
# Here's a brief summary for the top 3 important dimensions per digit
top_3_importances <- ae_vimp_dec %>%
  group_by(Digit) %>%
  top_n(3, wt = Importance) %>%
  arrange(Digit, desc(Importance))

print(top_3_importances, n = 100)




# # Extract important dimensions for digit 1 (example)
# important_dims_digit_1 <- c(20, 7, 30)
#
# # Update the column names to match your actual data frame
# important_dim_cols <- paste0("V", important_dims_digit_1)
#
# # Filter encoded representations for Digit 1
# digit_1_representations <- encoded_representations_df %>%
#   filter(Digit == 1) %>%
#   select(SampleID, Digit, all_of(important_dim_cols))
#
# # Plot the encoded dimensions
# ggplot(digit_1_representations, aes_string(x = important_dim_cols[1],
#                                            y = important_dim_cols[2],
#                                            color = "SampleID")) +
#   geom_point() +
#   labs(title = "Encoded Dimensions for Digit 1",
#        x = important_dim_cols[1],
#        y = important_dim_cols[2], color = "SampleID") +
#   theme_minimal()




# -------------------------------------------------------------------------

#
# plot_important_dimensions <- function(digit, important_dims) {
#   important_dim_cols <- paste0("V", important_dims)
#
#   digit_representations <- encoded_representations_df %>%
#     filter(Digit == digit) %>%
#     select(SampleID, Digit, all_of(important_dim_cols))
#
#   p <- ggplot(digit_representations, aes_string(x = important_dim_cols[1], y = important_dim_cols[2], color = "SampleID")) +
#     geom_point() +
#     labs(title = paste("Encoded Dimensions for Digit", digit), x = important_dim_cols[1], y = important_dim_cols[2], color = "SampleID") +
#     theme_minimal()
#
#   print(p)
# }
#
# # Example: Plot for digit 1
# plot_important_dimensions(1, c(28, 20, 9))


library(ggplot2)
library(dplyr)

# Define a list with the top 3 important dimensions for each digit
top_3_dims <- list(
  `0` = c(3, 25, 2),
  `1` = c(24, 2, 15),
  `2` = c(3, 2, 25),
  `3` = c(5, 17, 1),
  `4` = c(24, 2, 30),
  `5` = c(2, 24, 15),
  `6` = c(25, 14, 2),
  `7` = c(25, 3, 15),
  `8` = c(5, 14, 2),
  `9` = c(24, 2, 13)
)


# Combine the important dimensions for all digits into a single data frame
combined_data <- bind_rows(
  lapply(names(top_3_dims), function(digit) {
    dims <- top_3_dims[[digit]]
    important_dim_cols <- paste0("V", dims)

    encoded_representations_df %>%
      filter(Digit == as.integer(digit)) %>%
      select(SampleID, Digit, all_of(important_dim_cols)) %>%
      mutate(Dimension1 = get(important_dim_cols[1]),
             Dimension2 = get(important_dim_cols[2]),
             Dimension3 = get(important_dim_cols[3])) %>%
      select(SampleID, Digit, Dimension1, Dimension2, Dimension3)
  })
)

# Create facet wrap plot
ggplot(combined_data, aes(x = Dimension1, y = Dimension2)) +
  geom_point(color = 'steelblue', size = 0.7) +
  facet_wrap(~ Digit, scales = "free") +
  labs(title = "",
       x = "Dimension X",
       y = "Dimension Y",
       color = "SampleID") +
  theme_bw() +
  theme(legend.position = 'none')


# -------------------------------------------------------------------------

library(plotly)

plot_3d_important_dimensions <- function(digit, important_dims) {
  important_dim_cols <- paste0("V", important_dims)

  digit_representations <- encoded_representations_df %>%
    filter(Digit == digit) %>%
    select(SampleID, Digit, all_of(important_dim_cols))

  p <- plot_ly(digit_representations, x = ~get(important_dim_cols[1]), y = ~get(important_dim_cols[2]), z = ~get(important_dim_cols[3]),
               color = ~factor(SampleID), colors = c('#636EFA','#EF553B','#00CC96','#AB63FA'),
               marker = list(size = 3)) %>%
    add_markers() %>%
    layout(title = paste("Encoded Dimensions for Digit", digit),
           scene = list(xaxis = list(title = important_dim_cols[1]),
                        yaxis = list(title = important_dim_cols[2]),
                        zaxis = list(title = important_dim_cols[3])))

  p
}

# Example: 3D Plot for digit 1
plot_3d_important_dimensions(1, c(20, 7, 30))











