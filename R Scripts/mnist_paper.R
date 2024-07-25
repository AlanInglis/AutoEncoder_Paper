# Load Libraries
library(keras)
library(ggplot2)
library(autoImp)
library(shiny)
library(dplyr)

set.seed(101)
tensorflow::tf$random$set_seed(101)

# Load Data
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

considered_digits <- c(1, 0)
train_indices <- which(y_train %in% considered_digits)
x_train <- x_train[train_indices, , ]
y_train <- y_train[train_indices]

# Filter for odd digits in the test data
test_indices <- which(y_test %in% considered_digits)
x_test <- x_test[test_indices, , ]
y_test <- y_test[test_indices]

# Preprocess Data
x_train <- array_reshape(x_train / 255, c(nrow(x_train), 784))
x_test <- array_reshape(x_test / 255, c(nrow(x_test), 784))

# Filter to include only digit '3'
#x_train <- x_train[y_train == 3, ]
#x_test <- x_test[y_test == 3, ]
#y_test <- y_test[y_test == 3]


# Define Autoencoder Model
encoding_dim <- 12
input_img <- layer_input(shape = c(784))

encoded <- input_img %>%
  layer_dense(units = encoding_dim, activation = 'relu')

decoded <- encoded %>%
  layer_dense(units = 784, activation = 'sigmoid')

autoencoder <- keras_model(inputs = input_img, outputs = decoded)

# Compile and Fit the Model
autoencoder %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy'
)

autoencoder %>% fit(
  x_train, x_train,
  epochs = 50,
  batch_size = 256,
  validation_data = list(x_test, x_test)
)

# Create Encoder Model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)
encoded_imgs <- predict(encoder, x_test)
decoded_imgs <- predict(autoencoder, x_test)




# ------------------------------------------------------------------------
# -------------------------------------------------------------------------




# Adjusting plot layout and improving image quality
par(mfrow = c(2, 10), mar = c(0.5, 0.5, 2, 0.5), oma = c(2, 2, 2, 2))
for (i in 1001:1010) {
  # Original image
  img <- matrix(x_test[i,], ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev)) # rotating the image for correct orientation
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n', main = paste("Original", i), cex.main = 0.8)

  # Reconstructed image
  img <- matrix(decoded_imgs[i,], ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev)) # rotating the image for correct orientation
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n', main = paste("Reconstructed", i), cex.main = 0.8)
}

# -------------------------------------------------------------------------

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
  varimp <- matrix(0, nrow = 784, ncol = encoding_dim) # Initialize with 0 for summing

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
    #feature_importance_matrix <- matrix(varimp[, j], nrow = 32, ncol = 32)
    feature_importance_matrix <- matrix(varimp[, j], nrow = 28, ncol = 28)
    #feature_importance_matrix <- matrix(varimp[, j], nrow = 64, ncol = 64)

    # Convert the matrix to a data frame suitable for ggplot
    feature_importance_df <- as.data.frame(as.table(feature_importance_matrix))
    all_dimensions_importance[[as.character(j)]] <- feature_importance_df
  }

  myList <- list(
    "Vimp" = all_dimensions_importance
    #"selectedNumber" = encoderModel$selectedNumber
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


  # if(encoderModel$type == "mnist"){
  #   # extract info from autoencoder output
  #   encoder <- encoderModel[[1]]
  #   x_test <- encoderModel[[2]]
  #   y_test <- encoderModel[[3]]
  #   autoencoder <- encoderModel[[6]]
  #   digits <- encoderModel$selectedNumber
  #   if (is.null(digits)) {
  #     digits <- c(0:9)
  #   }
  #
  # }else if(encoderModel$type == "emnist"){
  #   encoder <- encoderModel[[1]]
  #   x_test <- encoderModel[[2]]
  #   y_test <- encoderModel[[3]]
  #   autoencoder <- encoderModel[[5]]
  #   digits <- c(10:35)
  # }

  encodingDim <- encoding_dim
  encoded_input <- keras::layer_input(shape = c(encodingDim))
  decoder_layer <- autoencoder$layers[[3]] # need to generalise this
  decoder <- keras::keras_model(inputs = encoded_input, outputs = decoder_layer(encoded_input))


  permImpInternal <- function(encoder, autoencoder, x_test, y_test, digit, num_permutations = 4, em = errorMetric) {
    digit_indices <- which(y_test == digit)
    digit_x_test <- x_test[digit_indices, ]

    # original
    original_encoded_imgs <- predict(encoder, digit_x_test, verbose = F)
    decoded_imgs <- predict(decoder, original_encoded_imgs, verbose = F)

    varimp <- matrix(0, nrow = encodingDim, ncol = 1)

    for (i in 1:encodingDim) {
      # print(paste0("Encoded dimension ", i, " computed"))


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
    # print(paste0("Running digit ", digit))

    digit_importance <- permImpInternal(encoder, autoencoder, x_test, y_test, digit)
    all_digits_importance[[as.character(digit)]] <- digit_importance
  }

  close(pb)


  importance_df <- data.frame(
    Digit = rep(digits, each = encodingDim),
    EncodedDimension = rep(1:encodingDim, times = length(digits)),
    Importance = unlist(all_digits_importance)
  )

  if(letter){
    num_to_letter <- setNames(as.character(LETTERS), digits)
    #Transform the Digit column
    importance_df <- importance_df |>
      dplyr::mutate(Digit = num_to_letter[as.character(Digit)])
  }
  # if(encoderModel$type == "emnist"){
  #   num_to_letter <- setNames(as.character(LETTERS), 10:35)
  #
  #   # Transform the Digit column
  #   importance_df <- importance_df |>
  #     dplyr::mutate(Digit = num_to_letter[as.character(Digit)])
  # }

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
  # and MNIST images at 28x28 (=784)
  #model_summaries <- matrix(0, nrow = 40encoding_dim, ncol = 32)
  model_summaries <- matrix(0, nrow = 784, ncol = encoding_dim)


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
  #vimp_matrix <- matrix(nrow = 40encoding_dim, ncol = 32)
  vimp_matrix <- matrix(nrow = 784, ncol = encoding_dim)

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
  p_dec <- ggplot(data_digit, aes(
    x = as.factor(EncodedDimension),
    y = Importance,
    fill = Importance > 0
  )) +
    geom_bar(stat = "identity", position = "dodge") +
    coord_flip() +
    scale_fill_manual(values = c("TRUE" = "firebrick", "FALSE" = "steelblue")) +
    labs(x = "Encoded Dimension", y = "Importance") +
    theme_bw() +
    ggtitle(paste0('Importance for digit: ', digit)) +
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
      # Reshape the column into a 28x28 matrix
      #mat <- matrix(data[, i], ncol = 64, nrow = 64)
      mat <- matrix(data[, i], ncol = 28, nrow = 28)
      mat <- t(apply(mat, 2, rev))
      mat <- t(apply(mat, 2, rev))
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
                            pp$legend,
                            ncol = 3,
                            widths = c(5, 5, 1)
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


set.seed(101)
ae_vimp_enc <- vimpEncodedtt(encoder = encoder, testData = x_test, num_permutations = 4)

set.seed(101)
ae_vimp_dec <- vimpDecodedNewtt(encoder = encoder,
                                x_test =  x_test,
                                y_test = y_test,
                                autoencoder = autoencoder,
                                digits = considered_digits,
                                num_permutations = 4,
                                letter = F)
set.seed(101)
lm_sum_2<- autoLmtt(encoder = encoder,
                    x_test =  x_test)
toneR::tone(2)

plotEncoded(ae_vimp_enc, legendName = 'Vimp')
plotDecoded(data = ae_vimp_dec)
shinyDecodedLmt(decodedVimp = ae_vimp_dec, lmSummary = lm_sum_2, encodedVimp = ae_vimp_enc)
# -------------------------------------------------------------------------

decodedLmt(encodedVimp = ae_vimp_enc,
           decodedVimp = ae_vimp_dec,
           lmSummary = lm_sum_2,
           digit = 3,
           sortVimp = F,
           filterZero = F)

decodedLm_single(encodedVimp = ae_vimp_enc,
                 decodedVimp = ae_vimp_dec,
                 lmSummary = lm_sum_2,
                 digit = 3,
                 sortVimp = T,
                 filterZero = F)

plotEncoded(ae_vimp_enc, legendName = 'Vimp', sort = T)


plotDecoded(data = ae_vimp_dec)

shinyDecodedLmt(decodedVimp = ae_vimp_dec, lmSummary = lm_sum_2, encodedVimp = ae_vimp_enc)




# -------------------------------------------------------------------------


plotEncoded <- function(digitResults,
                        legendName = "Error",
                        sort = FALSE,
                        colours = c("white", "red"),
                        plotSel = NULL
) {
  if (sort) {
    # remove empty dfs from encoded vimps
    # Access list of data frames
    vimp_dfs <- digitResults$Vimp

    # Identify dfs with only zeros in 'Freq'
    dfs_to_keep <- sapply(vimp_dfs, function(df) {
      if ("Freq" %in% names(df)) {
        return(any(df$Freq != 0)) # Return TRUE if there's any non-zero value
      } else {
        return(TRUE) # Keep the data frame if it doesn't have 'Freq' column
      }
    })

    # Remove dfs with only zeros
    vimp_dfs <- vimp_dfs[dfs_to_keep]

    # Update the original list
    digitResults$Vimp <- vimp_dfs

    # Calculate range for each data frame
    ranges <- sapply(digitResults$Vimp, function(df) {
      if ("Freq" %in% names(df)) {
        return(max(df$Freq, na.rm = TRUE) - min(df$Freq, na.rm = TRUE))
      } else {
        return(0) # Assign 0 range for data frames without 'Freq' column
      }
    })

    # Sort the dfs by ranges
    sorted_indices <- order(ranges, decreasing = TRUE)
    sorted_dfs <- digitResults$Vimp[sorted_indices]

    # Update the original list
    digitResults$Vimp <- sorted_dfs
  }


  digit <- digitResults$selectedNumber
  if (length(digit) < 10) {
    digit <- paste(digit, collapse = ",")
  }
  digitResults <- digitResults$Vimp
  nam <- names(digitResults)


  all_plots <- list()


  # get range for legend
  max_val <- max(sapply(digitResults, function(df) max(df$Freq)))
  min_val <- 0
  max_val = 0.101



  for (j in 1:length(digitResults)) {
    feature_importance_df <- digitResults[[j]]

    # create all plots
    p <- ggplot(feature_importance_df, aes(Var1, Var2, fill = Freq)) +
      geom_tile() +
      scale_y_discrete(limits = rev(levels(feature_importance_df$Var2))) +
      scale_fill_gradient(
        low = colours[1],
        high = colours[2],
       limits = c(min_val, max_val),
        name = legendName,
        guide = guide_colorbar(
          order = 1,
          frame.colour = "black",
          ticks.colour = "black"
        ),
        oob = scales::squish
      ) +
      theme_void() +
      theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
      # labs(title = if(length(digit) < 10) as.character(j) else paste0(digit, ':', j)) +
      labs(title = paste0(nam[j])) +
      # labs(title = paste0(digit, ':', j)) +
      theme(plot.title = element_text(hjust = 0.5)) +
      coord_fixed()

    all_plots[[j]] <- p
  }

  if (!is.null(plotSel)) {
    all_plots <- all_plots[1:plotSel]
  }

  # Get legend
  legend <- cowplot::get_legend(all_plots[[1]])

  # remoe legend from all plots
  all_plots_no_leg <- lapply(all_plots, function(x) x + theme(legend.position = "none"))


  # plot on a grid
  n <- length(all_plots)
  nRow <- floor(sqrt(n))
  all_plot_grid <- gridExtra::arrangeGrob(grobs = all_plots_no_leg, nrow = nRow)
  suppressMessages(
    gridExtra::grid.arrange(all_plot_grid,
                            legend,
                            ncol = 2,
                            widths = c(10, 1)
    )
  )
}

plotEncoded(ae_vimp_enc, legendName = 'Vimp')



decodedLm_single <- function(encodedVimp,
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
  #vimp_matrix <- matrix(nrow = 40encoding_dim, ncol = 32)
  vimp_matrix <- matrix(nrow = 784, ncol = encoding_dim)

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
  p_dec <- ggplot(data_digit, aes(
    x = as.factor(EncodedDimension),
    y = Importance,
    fill = Importance > 0
  )) +
    geom_bar(stat = "identity", position = "dodge") +
    coord_flip() +
    scale_fill_manual(values = c("TRUE" = "firebrick", "FALSE" = "steelblue")) +
    labs(x = "Encoded Dimension", y = "Importance") +
    theme_bw() +
    ggtitle(paste0('Importance for digit: ', digit)) +
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
      # Reshape the column into a 28x28 matrix
      #mat <- matrix(data[, i], ncol = 64, nrow = 64)
      mat <- matrix(data[, i], ncol = 28, nrow = 28)
      mat <- t(apply(mat, 2, rev))
      mat <- t(apply(mat, 2, rev))
      mat <- mat[nrow(mat):1, ]  # Flip the image vertically
      # Convert the matrix to a data frame for ggplot
      mat_long <- reshape2::melt(mat)

      # ggplot(feature_importance_df, aes(Var1, Var2, fill = Freq)) +
      #   geom_tile() +
      #   scale_y_discrete(limits = rev(levels(feature_importance_df$Var2))) +
      #   scale_fill_gradient(
      #     low = colours[1],
      #     high = colours[2],
      #     limits = c(min_val, max_val),
      #     name = legendName,
      #     guide = guide_colorbar(
      #       order = 1,
      #       frame.colour = "black",
      #       ticks.colour = "black"
      #     ),
      #     oob = scales::squish
      #   ) +
      #   theme_void() +
      #   theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)) +
      #   # labs(title = if(length(digit) < 10) as.character(j) else paste0(digit, ':', j)) +
      #   labs(title = paste0(nam[j])) +
      #   # labs(title = paste0(digit, ':', j)) +
      #   theme(plot.title = element_text(hjust = 0.5)) +
      #   coord_fixed()

      # Create a plot for this matrix
      p <- ggplot(mat_long, aes(x = Var1, y = Var2, fill = value)) +
        geom_tile() +
        scale_y_discrete(limits = rev(levels(mat_long$Var2))) +
        scale_fill_gradient2(
          low = 'blue',
          high = 'red',
          mid = 'white',
          limits = c(min_val, max_val),
          name = legendTitle,
          guide = guide_colorbar(
            order = 1,
            frame.colour = "black",
            ticks.colour = "black"
          ),
          oob = scales::squish
        ) +
        labs(title = paste(myVec[i])) +
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

  pp <- plotFun(data = filtered_vimp, legendTitle = 'Pixel \nVimp \nDirection')




  # plot everything on grid -------------------------------------------------

  #plot on a grid
  n <- length(pp$plots)
  nRow <- floor(sqrt(n))
  all_plot_grid <- gridExtra::arrangeGrob(grobs = pp$plots, nrow = nRow)
  suppressMessages(
    gridExtra::grid.arrange(all_plot_grid,
                            pp$legend,
                            ncol = 2,
                            widths = c(10, 1)
    )
  )



}

decodedLm_single(ae_vimp_enc, ae_vimp_dec, lm_sum_2, digit = 1, sortVimp = F)


save.image(file = 'mnist_ex.RData')
load("/Users/alaninglis/Desktop/Autoencoder Paper/saved_objects/mnist_ex.RData")




