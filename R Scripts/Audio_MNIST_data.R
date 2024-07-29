library(keras)
library(aim)
library(umap)



####################################
#### Section 4 Audio MNIST Data ####
####################################

# This script builds the autoencoder and visualises the results for the audio MNIST data.
# You can bypass this script by loading the saved object directly, found at https://github.com/AlanInglis/AutoEncoder_Paper
# This object contains the AE and importance results. If loading saved object, skip to "Run AIM functions" section.
# NOTE: If loading object, the data must first be converted to the new format
# via the code provided.

# Load results and model --------------------------------------------------


# load
load("/Users/alaninglis/Desktop/Autoencoder Paper/saved_objects/mnist_speech.RData")


# convert to new format:
# ae_vimp_enc is the list of input pixel importance, convert column types
for (i in 1:length(ae_vimp_enc$Vimp)) {
  ae_vimp_enc$Vimp[[i]]$Var1 <- as.integer(ae_vimp_enc$Vimp[[i]]$Var1)
  ae_vimp_enc$Vimp[[i]]$Var2 <- as.integer(ae_vimp_enc$Vimp[[i]]$Var2)
}


# rename columns
for (i in 1:length(ae_vimp_enc$Vimp)) {
  names(ae_vimp_enc$Vimp[[i]]) <- c("Row", "Col", "Value")
}

# ae_vimp_dec is the object of encoded dimension importance, rename column
colnames(ae_vimp_dec)[1] <- "Class"

# END OF LOAD RESULTS AND CONVERSION

# Load images -------------------------------------------------------------

# load and preprocess images function
load_and_preprocess_image <- function(image_path) {
  img <- png::readPNG(image_path)
  img <- img[, , 1:3] # Keep only the RGB channels, discard alpha
  return(img)
}

image_directory <- "/Users/alaninglis/Desktop/autoencoder_stuff/MNIST_speech/col_32"
image_files <- list.files(path = image_directory, pattern = "\\.png$", full.names = TRUE)


# load images
pb <- txtProgressBar(min = 0, max = length(image_files), style = 3)
images <- lapply(image_files, function(file) {
  img <- load_and_preprocess_image(file)
  setTxtProgressBar(pb, which(image_files == file))
  return(img)
})
close(pb)

# END OF LOAD IMAGES

# Smooth images -----------------------------------------------------------

# Function to smooth and average the channels of a single image
smooth_and_average_channels <- function(image_data) {
  # Initialise an empty matrix to store the averaged smoothed channels
  smoothed_image <- matrix(0, nrow = 32, ncol = 32)

  # Define the grid coordinates
  x_grid <- rep(1:32, each = 32)
  y_grid <- rep(1:32, times = 32)

  # Convert the 3D image data array to a list of vectors, one for each channel
  channel_vectors <- lapply(1:3, function(channel) as.vector(image_data[, , channel]))

  # Apply smooth.2d to each color channel and then average the results
  for (i in 1:3) {
    # Apply smooth.2d to the vector with coordinates
    smoothed_vector <- smooth.2d(Y = channel_vectors[[i]],
                                 x = cbind(x_grid, y_grid),
                                 nrow = 32,
                                 ncol = 32,
                                 theta = 0.1,
                                 surface = TRUE)$z

    # Accumulate the smoothed results
    smoothed_image <- smoothed_image + smoothed_vector / 3
  }

  return(smoothed_image)
}

#  Apply function to each image

# progress bar
pb <- txtProgressBar(min = 0, max = length(images), style = 3)

# Initialise counter to track the progress
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

# Apply the function to all images
smoothed_images <- lapply(images, process_image_with_progress)


# Close the progress bar
close(pb)


# display example of smoothed images
par(mfrow = c(1, 5))
for (i in 1:5) {
  image(1:32, 1:32, smoothed_images[[i]], col = viridis(256), main = paste("Image", i))
}

# END OF SMOOTH IMAGES

# Preprocess data ---------------------------------------------------------

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

# END OF PREPROCESSING


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
  shuffle = T,
  validation_data = list(test_images, test_images)
)



# encoder model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)

# END OF AUTOENCODER

# Plot reconstructed images -----------------------------------------------

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

# Figure 10:
for (i in 1:10) {
  # Plot original image
  image(1:32, 1:32, flip_matrix_vertically(original_reshaped[i,,]), col = viridis(255), main = paste("Original", i), axes = FALSE)

  # Plot reconstructed image
  image(1:32, 1:32, flip_matrix_vertically(reconstructed_reshaped[i,,]), col = viridis(255), main = paste("Reconstructed", i), axes = FALSE)
}

# END OF PLOT RECONSTRUCTED


# UMAP --------------------------------------------------------------------

# get encoded images
encoded_imgs <- predict(encoder, x_test)

# no idea why 2 rows are identical
df <- as.data.frame(encoded_imgs)

# Perform UMAP dimensionality reduction
umap_results <- umap(df, n_neighbors = 15, min_dist = 0.1, n_components = 2)

# Convert UMAP results to a data frame
umap_data <- as.data.frame(umap_results$layout)
umap_data$digit <- y_test

# Figure 11:
ggplot(umap_data, aes(x = V1, y = V2, label = as.factor(digit), color = as.factor(digit))) +
  geom_text(alpha = 0.5) +
  labs(title = "",
       x = "UMAP 1",
       y = "UMAP 2",
       color = "Digit") +
  theme_bw() +
  theme(legend.position = 'none')

# END OF UMAP

# Save -----------------------------------------------------------

# save
# save.image(file = 'mnist_speech.RData')


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



# Figure 12:
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lm_sum_2,
                     class = 0,
                     sort = T,
                     topX = 32,
                     flip_horizontal = T,
                     flip_vertical = F,
                     rotate = 1,
                     filter_zero = F,
                     show_legend = F)

# Figure 4
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)


# END OF SCRIPT
