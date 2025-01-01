# Install devtools if you haven't already
#install.packages("devtools")

# Install the aim package
#devtools::install_github("AlanInglis/AIM")

library(keras)
library(aim)



##############################
#### Section 3 MNIST Data ####
##############################
rm(list = ls())

# This script builds the autoencoder and visualises the results for the MNIST data shown in Section 3 of our paper.
# You can bypass the fitting of the AE model and permutation importance process by
# loading the saved data objects directly, found at https://github.com/AlanInglis/AutoEncoder_Paper/Saved_Objects/MNIST
# These saved data objects contains the AE and importance results.
# Lines 27-37 below will load the saved objects and display the results

# If you wish to fit the AE and recalculate the results, begin at the "Data and Autoencoder"
# section (line 39).


# Load results, model, and visualise --------------------------------------------------

# load data (replace with your path to data)
load("~/mnist_model.RData")
autoencoder <- load_model_tf("~/autoencoder_model_mnist.keras")
encoder <- load_model_tf("~/encoder_model_mnist.keras")

# Figure 3 (NOTE: select sort by importance and flip horizontal)
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)

# END OF LOAD RESULTS AND VISUALISE

# Data and Autoencoder ----------------------------------------------------

# set seed
set.seed(101)
np <- reticulate::import("numpy")
np$random$seed(101L)
reticulate::py_run_string("import random; random.seed(123)")
tensorflow::tf$random$set_seed(101)

# Enable deterministic operations in TensorFlow to avoid non-deterministic behavior.
tensorflow::tf$config$experimental$enable_op_determinism()

# Load Data
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# select only digit 0 and 1
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

# Define Autoencoder Model
encoding_dim <- 12
input_img <- layer_input(shape = c(784))

encoded <- input_img %>%
  layer_dense(units = encoding_dim, activation = 'relu')

decoded <- encoded %>%
  layer_dense(units = 784, activation = 'sigmoid')

autoencoder <- keras_model(inputs = input_img, outputs = decoded)

# Compile Model
autoencoder %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy'
)

# fit autoencoder
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


# END OF MODEL BUILDING

# Plot reconstructed images -----------------------------------------------

# get decoced images
decoded_imgs <- predict(autoencoder, test_images)
# Adjusting plot layout and improving image quality
par(mfrow = c(2, 10), mar = c(0.5, 0.5, 2, 0.5), oma = c(2, 2, 2, 2))

# First row: Original images
for (i in 1:10) {
  img <- matrix(x_test[i,], ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Input", i), cex.main = 0.8)
}

# Second row: Reconstructed images
for (i in 1:10) {
  img <- matrix(decoded_imgs[i,], ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Output", i), cex.main = 0.8)
}

# END OF PLOT RECONSTRUCTED



# Run AIM functions -------------------------------------------------------

# input pixel importance
ae_vimp_enc <- vimp_input(encoder = encoder, test_data = x_test, num_permutations = 4)

# encoded dimension importance
ae_vimp_dec <- vimp_encoded(encoder = encoder,
                            test_data =  x_test,
                            test_labels = y_test,
                            autoencoder = autoencoder,
                            classes = considered_digits,
                            num_permutations = 4)

# directional importance
lm_sum_2  <- vimp_direction(encoder = encoder,
                            test_data =  x_test)



# plot
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lm_sum_2,
                     class = 1,
                     sort = T,
                     topX = 10,
                     flip_horizontal = T,
                     flip_vertical = F,
                     rotate = 0,
                     filter_zero = F)

# Figure 4 (NOTE: select sort by importance and flip horizontal)
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)


# END OF SCRIPT


# Save Objects ------------------------------------------------------------

# save_model_tf(autoencoder, "autoencoder_model_mnist.keras")
# save_model_tf(encoder, "encoder_model_mnist.keras")
# save.image(file = 'mnist_model.RData')





