# Install devtools if you haven't already
#install.packages("devtools")

# Install the aim package
#devtools::install_github("AlanInglis/AIM")

library(keras)
library(aim)

###############################
#### Section 4 FMNIST Data ####
###############################
rm(list = ls())

# This script builds the autoencoder and visualises the results for the MNIST data shown in Section 4 of our paper.
# You can bypass the fitting of the AE model and permutation importance process by
# loading the saved data objects directly, found at https://github.com/AlanInglis/AutoEncoder_Paper/Saved_Objects/FMNIST
# These saved data objects contains the AE and importance results.
# Lines 24-34 below will load the saved objects and display the results

# If you wish to fit the AE and recalculate the results, begin at the "Data and Autoencoder"
# section (line 36).

# Load results, model, and visualise --------------------------------------------------

# load data (replace with your path to data)
load("~/fmnist_model.RData")
autoencoder <- load_model_tf("~/autoencoder_model_fmnist.keras")
encoder <- load_model_tf("~/encoder_model_fmnist.keras")

# Figure 7  (NOTE: select sort by importance and flip vertical)
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)

# END OF LOAD RESULTS AND VISUALISE

# Data and Autoencoder ----------------------------------------------------

# set seed
set.seed(1701)
np <- reticulate::import("numpy")
np$random$seed(1701L)
reticulate::py_run_string("import random; random.seed(1701)")
tensorflow::tf$random$set_seed(1701)

# Enable deterministic operations in TensorFlow to avoid non-deterministic behavior.
tensorflow::tf$config$experimental$enable_op_determinism()


# Load MNIST fashion data (FMNIST) from keras
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

# autoencoder
encoded <- input_img %>%
  layer_dense(units = encoding_dim, activation = 'relu')

decoded <- encoded %>%
  layer_dense(units = 784, activation = 'sigmoid')

# autoencoder model
autoencoder <- keras_model(inputs = input_img, outputs = decoded)


# Compile Model
autoencoder %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy'
)

# fit autoencoder
autoencoder %>% fit(
  x_train, x_train,
  epochs = 100,
  batch_size = 256,
  validation_data = list(x_test, x_test)
)


# Create Encoder Model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)
encoded_imgs <- predict(encoder, x_test)
decoded_imgs <- predict(autoencoder, x_test)

# END OF MODEL BUILDING


# Plot reconstructed images -----------------------------------------------
# Figure 6:

# Get unique classes and initialise plotting
unique_classes <- sort(unique(y_test))
num_classes <- length(unique_classes)

# Select one image per class
selected_indices <- sapply(unique_classes, function(class) {
  which(y_test == class)[4]  # Select the first occurrence of each class
})
selected_images <- x_test[selected_indices, ]  # Input images
selected_reconstructions <- decoded_imgs[selected_indices, ]  # Reconstructed images

# Set up the plotting area
par(mfrow = c(2, num_classes),    # 2 rows, number of classes columns
    mar = c(0.5, 0.5, 2, 0.5),    # Margins around each plot
    oma = c(1, 0, 0, 0),          # Outer margins for the whole plot area
    mgp = c(1, 0.5, 0),           # Margin line for title, labels, and axis
    cex.main = 1.2)               # Main title size

# Plot original images on the top row
for (i in 1:num_classes) {
  img <- selected_images[i, ]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev))  # Flip the image
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Input:", unique_classes[i]))
}

# Plot reconstructed images on the bottom row
for (i in 1:num_classes) {
  img <- selected_reconstructions[i, ]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev))  # Flip the image
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Output:", unique_classes[i]))
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
                            classes = c(0:9),
                            num_permutations = 4)

# directional importance
lm_sum_2  <- vimp_direction(encoder = encoder,
                            test_data =  x_test)


# Figure 7:
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lm_sum_2,
                     class = "8",
                     sort = T,
                     topX = 10,
                     flip_horizontal = F,
                     flip_vertical = T,
                     rotate = 0,
                     filter_zero = F)

# shiny plot (NOTE: select flip vertical)
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)

# SAVE OBJECTS ------------------------------------------------------------
# save_model_tf(autoencoder, "autoencoder_model_fmnist.keras")
# save_model_tf(encoder, "encoder_model_fmnist.keras")
# save.image(file = 'fmnist_model.RData')

# END OF SCRIPT
















# set seed
set.seed(1701)
np <- reticulate::import("numpy")
np$random$seed(1701L)
reticulate::py_run_string("import random; random.seed(1701)")
tensorflow::tf$random$set_seed(1701)

# Enable deterministic operations in TensorFlow to avoid non-deterministic behavior.
tensorflow::tf$config$experimental$enable_op_determinism()

# Data and Autoencoder ----------------------------------------------------

# Load MNIST fashion data (FMNIST)
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

# autoencoder
encoded <- input_img %>%
  layer_dense(units = encoding_dim, activation = 'relu')

decoded <- encoded %>%
  layer_dense(units = 784, activation = 'sigmoid')

# autoencoder model
autoencoder <- keras_model(inputs = input_img, outputs = decoded)


# Compile Model
autoencoder %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy'
)

# fit autoencoder
autoencoder %>% fit(
  x_train, x_train,
  epochs = 100,
  batch_size = 256,
  validation_data = list(x_test, x_test)
)




# Create Encoder Model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)
encoded_imgs <- predict(encoder, x_test)

# END OF MODEL BUILDING


# Plot reconstructed images -----------------------------------------------

# Get unique classes and initialise plotting
unique_classes <- sort(unique(y_test))
num_classes <- length(unique_classes)

# Select one image per class
selected_indices <- sapply(unique_classes, function(class) {
  which(y_test == class)[4]  # Select the first occurrence of each class
})
selected_images <- x_test[selected_indices, ]  # Input images
selected_reconstructions <- decoded_imgs[selected_indices, ]  # Reconstructed images

# Set up the plotting area
par(mfrow = c(2, num_classes),    # 2 rows, number of classes columns
    mar = c(0.5, 0.5, 2, 0.5),    # Margins around each plot
    oma = c(1, 0, 0, 0),          # Outer margins for the whole plot area
    mgp = c(1, 0.5, 0),           # Margin line for title, labels, and axis
    cex.main = 1.2)               # Main title size

# Plot original images on the top row
for (i in 1:num_classes) {
  img <- selected_images[i, ]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev))  # Flip the image
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Input:", unique_classes[i]))
}

# Plot reconstructed images on the bottom row
for (i in 1:num_classes) {
  img <- selected_reconstructions[i, ]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- t(apply(img, 2, rev))  # Flip the image
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Output:", unique_classes[i]))
}

# END OF PLOT RECONSTRUCTED

# Save -----------------------------------------------------------

# save
# save.image(file = 'fmnist.RData')

# Run AIM functions -------------------------------------------------------

# input pixel importance
ae_vimp_enc <- vimp_input(encoder = encoder, test_data = x_test, num_permutations = 4)

# encoded dimension importance
ae_vimp_dec <- vimp_encoded(encoder = encoder,
                            test_data =  x_test,
                            test_labels = y_test,
                            autoencoder = autoencoder,
                            classes = c(0:9),
                            num_permutations = 4)

# directional importance
lm_sum_2  <- vimp_direction(encoder = encoder,
                            test_data =  x_test)



toneR::tone(2)

# Figure 7:
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lm_sum_2,
                     class = "8",
                     sort = T,
                     topX = 10,
                     flip_horizontal = F,
                     flip_vertical = T,
                     rotate = 0,
                     filter_zero = F)

# shiny plot  (NOTE: select sort by importance and flip vertical)
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)

# SAVE OBJECTS ------------------------------------------------------------
save_model_tf(autoencoder, "autoencoder_model_fmnist.keras")
save_model_tf(encoder, "encoder_model_fmnist.keras")
save.image(file = 'fmnist_model.RData')

# END OF SCRIPT



