# Load necessary libraries
library(keras)
library(tensorflow)
library(ggplot2)
library(dplyr)
library(shiny)
library(autoImp)
set.seed(1234)
tensorflow::tf$random$set_seed(1234)


# Load MNIST fashion data
fashion_mnist <- dataset_fashion_mnist()
x_train <- fashion_mnist$train$x
y_train <- fashion_mnist$train$y
x_test <- fashion_mnist$test$x
y_test <- fashion_mnist$test$y

# Preprocess Data
x_train <- array_reshape(x_train / 255, c(nrow(x_train), 784))
x_test <- array_reshape(x_test / 255, c(nrow(x_test), 784))

# Define Autoencoder Model
encoding_dim <- 32
edim <- encoding_dim
input_img <- layer_input(shape = c(784))

encoded <- input_img %>%
  layer_dense(units = encoding_dim, activation = 'relu')

decoded <- encoded %>%
  layer_dense(units = 784, activation = 'sigmoid')

autoencoder <- keras_model(inputs = input_img, outputs = decoded)

# Compile and Fit the Model
autoencoder %>% compile(
  #optimizer = keras$optimizers$legacy$Adam(learning_rate = 0.01),
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



# Set up the plotting area
par(mfrow = c(2, 8),    # 2 rows, 10 columns layout
    mar = c(0.5, 0.5, 2, 0.5),  # Margins around each plot
    oma = c(1, 0, 0, 0),  # Outer margins for the whole plot area
    mgp = c(1, 0.5, 0),  # Margin line for title, labels, and axis
    cex.main = 1.2)  # Main title size

for (i in 1:8) {
  # Original image
  img <- x_test[i,]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Input", i))

  # Reconstructed image
  img <- decoded_imgs[i,]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Output", i))
}


# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
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
  varimp <- matrix(0, nrow = 784, ncol = edim) # Initialize with 0 for summing

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

      for (j in 1:edim) {
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

  for (j in 1:edim) {
    # Convert to matrix for each dimension
    feature_importance_matrix <- matrix(varimp[, j], nrow = 28, ncol = 28)

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
                             encodingDim = edim,
                             errorMetric = "MSE") {

  encodingDim = edim

  encoded_input <- keras::layer_input(shape = c(encodingDim))
  decoder_layer <- autoencoder$layers[[3]] # decoder layer
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


  # List of words associated with each digit
  fashion_names <- c('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                     'Shirt', 'Sneaker', 'Bag', 'Ankle_boot')

  # Map the names to the digits
  importance_df$Digit <- fashion_names[importance_df$Digit + 1]


  return(importance_df)
}

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------



autoLmtt <- function(encoder, x_test, threshold = NULL){



  # predict
  encoded_imgs <- predict(encoder, x_test, verbose = F)

  # covert to df
  x_test_df <- as.data.frame(x_test)
  encoded_imgs_df <- as.data.frame(encoded_imgs)

  # rename cols
  colnames(encoded_imgs) <- paste0("encoded_dim_", 1:edim) # edim latent dims


  # Initialize matrix to hold the coefficients using edim latent dims
  # and MNIST images at 2edimx2edim (=7edim4)
  #model_summaries <- matrix(0, nrow = 40edim, ncol = edim)
  model_summaries <- matrix(0, nrow = 784, ncol = edim)


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
  #vimp_matrix <- matrix(nrow = 40edim, ncol = edim)
  vimp_matrix <- matrix(nrow = 784, ncol = edim)

  # Loop through each data frame in the list
  for (i in 1:edim) {
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
  colnames(vimp_lm_mat) <- c(1:edim)
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
  #  # ggtitle(paste0('Importance for: ', digit)) +
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
    colnames(vimp_matrix) <- c(1:edim)
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
      # Reshape the column into a 2edimx2edim matrix
      #mat <- matrix(data[, i], ncol = 64, nrow = 64)
      mat <- matrix(data[, i], ncol = 28, nrow = 28)
      mat <- t(apply(mat, 2, rev))
      mat <- t(apply(mat, 2, rev))
      #mat <- mat[nrow(mat):1, ]  # Flip the image vertically
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
          # limits = c(min_val, max_val),
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
    # gridExtra::grid.arrange(p_dec, all_plot_grid,
    #                         pp$legend,
    #                         ncol = 3,
    #                         widths = c(5, 5, 1)
    # )
    gridExtra::grid.arrange(p_dec, all_plot_grid,
                          #  pp$legend,
                            ncol = 3,
                            widths = c(5, 5, 0.5)
    )
  )



}








# -------------------------------------------------------------------------

# -------------------------------------------------------------------------


shinyDecodedLm_test <- function(decodedVimp, lmSummary, encodedVimp) {
  # UI ----------------------------------------------------------------------

  # List of words associated with each digit
  fashion_names <- c('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                     'Shirt', 'Sneaker', 'Bag', 'Ankle_boot')


  ui <- fluidPage(
    titlePanel("Autoencoder Analysis"),

    # Row for letter selection and threshold
    fluidRow(
      column(
        4,
        selectInput("digit", "Select Item:", choices = fashion_names, selected = "T-Shirt")
      )
    ),

    # Row for additional controls
    fluidRow(
      column(
        4,
        numericInput("topX", "Number of Top Importances:", value = 5)
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


plotDecodedtt <- function(data) {
  #  plot
  p <- ggplot(data, aes(x = as.factor(EncodedDimension), y = Importance, fill = Importance)) +
    geom_bar(stat = "identity") +
    scale_fill_gradient(
      low = "steelblue",
      high = "red",
      name = "Vimp",
      guide = guide_colorbar(
        frame.colour = "black",
        ticks.colour = "black"
      )
    ) +
    facet_wrap(~Digit) +
    # facet_wrap(~ Digit, scales = "free_y") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
          axis.text=element_text(size=5)) +
    labs(
      # title = "Importance of Each Encoded Dimension",
      x = "Encoded Dimension",
      y = "Importance"
    )
  return(p)
}


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
#library(ggplot2)
# perm imp for encoded
edim = encoding_dim
set.seed(1234)
ae_vimp_enc <- vimpEncodedtt(encoder = encoder, testData = x_test, num_permutations = 4)
# perm imp for decoded
set.seed(1234)
ae_vimp_dec <- vimpDecodedNewtt(encoder = encoder,
                                x_test =  x_test,
                                y_test = y_test,
                                autoencoder = autoencoder,
                                digits =  c(0:9),
                                num_permutations = 4)
# LM
set.seed(1234)
lm_sum_2<- autoLmtt(encoder = encoder,
                    x_test =  x_test)


save.image(file = 'fmnist.RData')
load("/Users/alaninglis/Desktop/Autoencoder Paper/saved_objects/fmnist.RData")


plotDecodedtt(data = ae_vimp_dec) +  theme(
  axis.title = element_text(size = 10),
  axis.text.x = element_text(size = 10,
                             angle = 90,
                             hjust = 1,
                             vjust = 0.8),
  axis.text.y = element_text(size = 8)
)



shinyDecodedLm_test(decodedVimp = ae_vimp_dec, lmSummary = lm_sum_2, encodedVimp = ae_vimp_enc)

plotEncoded(ae_vimp_enc, legendName = 'Vimp')


fashion_names <- c('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                   'Shirt', 'Sneaker', 'Bag', 'Ankle_boot')

decodedLmt(encodedVimp = ae_vimp_enc, decodedVimp = ae_vimp_dec, lmSummary = lm_sum_2,
           digit = "Trouser",
           sort = T, topX = 10) + ggtitle('')
