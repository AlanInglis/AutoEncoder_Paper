# Install devtools if you haven't already
#install.packages("devtools")

# Install the aim package
#devtools::install_github("AlanInglis/AIM")


library(keras)
library(aim)
library(umap)
library(ggplot2)

####################################
#### Section 5 Audio MNIST Data ####
####################################
rm(list = ls())

# This script builds the autoencoder and visualises the results for the Audio MNIST data shown in Section 5 of our paper.
# You can bypass the fitting of the AE model and permutation importance process by
# loading the saved data objects directly, found at https://github.com/AlanInglis/AutoEncoder_Paper/Saved_Objects/AUDIO_MNIST
# These saved data objects contains the AE and importance results.
# Lines 22-32 below will load the saved objects and display the results

# If you wish to fit the AE and recalculate the results, begin at the "Data and Autoencoder"
# section (line 34).

# Load results, model, and visualise --------------------------------------------------

# load data (replace with your path to data)
load("~/audio_mmnist_model.RData")
autoencoder <- load_model_tf("~/autoencoder_model_audio.keras")
encoder <- load_model_tf("~/encoder_model_audio.keras")

# Figure 11 (NOTE: select sort by importance, rotation = 1)
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)

# END OF LOAD RESULTS AND VISUALISE

# Data and Autoencoder ----------------------------------------------------

# set seed
set.seed(2001)
np <- reticulate::import("numpy")
np$random$seed(2001L)
reticulate::py_run_string("import random; random.seed(2001)")
tensorflow::tf$random$set_seed(2001)

# Enable deterministic operations in TensorFlow to avoid non-deterministic behavior.
tensorflow::tf$config$experimental$enable_op_determinism()

# Load images -------------------------------------------------------------

# load data (data can be found at https://github.com/AlanInglis/AutoEncoder_Paper/Data/Audio_MNIST_DATA.zip)
train_images <- read.csv('~/train_images.csv', sep = ',')
test_images <- read.csv('~/test_images.csv', sep = ',')
load("~/test_indices.RData")
load("~/y_test.RData")
train_images <- as.matrix(train_images)
test_images <- as.matrix(test_images)
x_test <- test_images


#  confirm the input shape:
cat("Shape of train_images: ", dim(train_images), "\n")
cat("Shape of test_images: ", dim(test_images), "\n")

# END OF LOADING


# Autoencoder -------------------------------------------------------------

# input size
input_size <- 32 * 32
input_img <- layer_input(shape = c(input_size))

# Encoder
encoded <- input_img %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 450, activation = "relu") %>%
  layer_dense(units = 384, activation = "relu") %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 192, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 96, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu")

# Decoder
decoded <- encoded %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 96, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 192, activation = "relu") %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 384, activation = "relu") %>%
  layer_dense(units = 450, activation = "relu") %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = input_size, activation = "sigmoid")


# Autoencoder model
autoencoder <- keras_model(inputs = input_img, outputs = decoded)

autoencoder %>% compile(
  loss =  'binary_crossentropy',
  optimizer = 'adam'
)


# Train the model
autoencoder %>% fit(
  train_images, train_images,
  epochs = 100,
  batch_size = 256,
  shuffle = F,
  validation_data = list(test_images, test_images)
)

# encoder model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)

# END OF AUTOENCODER

# Plot reconstructed images -----------------------------------------------
# Figure 9:

# reconstructed images from the test set
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

# Plot original images on the top row
for (i in 1:10) {
  image(1:32, 1:32, flip_matrix_vertically(original_reshaped[i,,]),
        col = viridis::viridis(255), main = paste("Input", i-1), axes = FALSE)
}

# Plot reconstructed images on the bottom row
for (i in 1:10) {
  image(1:32, 1:32, flip_matrix_vertically(reconstructed_reshaped[i,,]),
        col = viridis::viridis(255), main = paste("Output", i-1), axes = FALSE)
}
# END OF PLOT RECONSTRUCTED


# UMAP --------------------------------------------------------------------

# get encoded images
encoded_imgs <- predict(encoder, x_test)
df <- as.data.frame(encoded_imgs)

# Perform UMAP dimensionality reduction
umap_results <- umap(df, n_neighbors = 15, min_dist = 0.1, n_components = 2)

# Convert UMAP results to a data frame
umap_data <- as.data.frame(umap_results$layout)
umap_data$digit <- y_test

# Figure 10:
ggplot(umap_data, aes(x = V1, y = V2, label = as.factor(digit), color = as.factor(digit))) +
  geom_text(alpha = 0.5) +
  labs(title = "",
       x = "UMAP 1",
       y = "UMAP 2",
       color = "Digit") +
  theme_bw() +
  theme(legend.position = 'none')

# END OF UMAP


# Run AIM functions -------------------------------------------------------

# input pixel importance
ae_vimp_enc <- vimp_input(encoder = encoder, test_data = x_test, num_permutations = 4)

# encoded dimension importance
ae_vimp_dec <- vimp_encoded(
  encoder = encoder,
  test_data = x_test,
  test_labels = y_test,
  autoencoder = autoencoder,
  classes =  c(0:9),
  num_permutations = 4
)

# directional importance
lm_sum_2 <- vimp_direction(
  encoder = encoder,
  test_data = x_test
)


# Figure 11:
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lm_sum_2,
                     class = 9,
                     sort = T,
                     topX = 32,
                     flip_horizontal = F,
                     flip_vertical = F,
                     rotate = 1,
                     filter_zero = T,
                     show_legend = F)


# NOTE: select sort by importance, rotation = 1
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)


# END OF SCRIPT


# Save Objects ------------------------------------------------------------
save_model_tf(autoencoder, "autoencoder_model_audio.keras")
save_model_tf(encoder, "encoder_model_audio.keras")
save.image(file = 'audio_mmnist_model.RData')





