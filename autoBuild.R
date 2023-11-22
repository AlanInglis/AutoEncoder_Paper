#' autoBuild
#'
#' @description Build an autoencoder model using the MNIST number data.
#'
#' @param numSel Which digit from the MNIST data to filter.
#'
#' @return an encoder model, and both the training and test data.
#'



autoBuild <- function(numSel = NULL){


# Get the Data ------------------------------------------------------------

  # Load Data
  mnist <- dataset_mnist()
  x_train <- mnist$train$x
  y_train <- mnist$train$y
  x_test <- mnist$test$x
  y_test <- mnist$test$y


  if(!is.null(numSel)){
    # Filter for selected number
    train_indices <- which(y_train == numSel)
    test_indices <- which(y_test == numSel)

    x_train <- x_train[train_indices, , ]
    y_train <- y_train[train_indices]
    x_test <- x_test[test_indices, , ]
    y_test <- y_test[test_indices]
  }


  # Preprocess Data
  x_train <- array_reshape(x_train / 255, c(nrow(x_train), 784))
  x_test <- array_reshape(x_test / 255, c(nrow(x_test), 784))



# Autoencoder Set-up ------------------------------------------------------

  # Encoder Code ------------------------------------------------------------
  # Set the number of dimensions in the encoded (compressed) representation
  encoding_dim <- 32

  # Define the input layer of the neural network with shape 784 (28x28)
  input_img <- layer_input(shape = c(784))


  # reduce dimensionality to size of 'encoding_dim', using ReLU activation
  encoded <- input_img %>%
    layer_dense(units = encoding_dim, activation = "relu")

  # bring dimensionality back up to 784, using sigmoid activation
  decoded <- encoded %>%
    layer_dense(units = 784, activation = "sigmoid")

  # Create the autoencoder model by specifying input and output layers
  autoencoder <- keras_model(inputs = input_img, outputs = decoded)

  # Compile the autoencoder model
  autoencoder %>% compile(
    optimizer = keras$optimizers$legacy$Adam(learning_rate = 0.01),
    loss = "binary_crossentropy"
  )

  # Train the autoencoder using:
  # - Training data (x_train) for both input and output (since it's an autoencoder)
  # - Validation data, which also uses the same input and output format as training data
  history <- autoencoder %>% fit(
    x_train, x_train,
    epochs = 10,
    batch_size = 256,
    validation_data = list(x_test, x_test)
  )

  # Create an encoder model by specifying input of the autoencoder and encoded output
  encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)

  # Generate encoded (compressed) representations of test images (Dim = 32)
  #encoded_imgs <- predict(encoder, x_test)

  myList <- list('encoder' = encoder, 'td' = x_test, 'trd' = x_train)
  return(myList)
}


