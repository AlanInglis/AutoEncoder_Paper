# This script is preprocessing for the Audio_Mnist data

# Load images -------------------------------------------------------------

# load and preprocess images function
load_and_preprocess_image <- function(image_path) {
  img <- png::readPNG(image_path)
  img <- img[, , 1:3] # Keep only the RGB channels, discard alpha
  return(img)
}


image_directory <- "/Users/alaninglis/Desktop/Autoencoder Paper/test_Dat/col_images" # replace with own path to directory
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
    smoothed_vector <- fields::smooth.2d(Y = channel_vectors[[i]],
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
  image(1:32, 1:32, smoothed_images[[i]], col = viridis::viridis(256), main = paste("Image", i))
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
  reticulate::array_reshape(img, c(1, prod(dim(img))))  # Reshape to include batch dimension
})

# Convert to matrix, ensuring it maintains a 2D shape
image_matrix <- do.call(rbind, images_flattened)

# Check the shape of the inputs
cat("Shape of the image matrix: ", dim(image_matrix), "\n")

#  split
set.seed(1357)
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

y_train <- labels[train_indices]


# -------------------------------------------------------------------------
saveRDS(train_images, "train_images.rds")
saveRDS(test_images, "test_images.rds")
saveRDS(image_files, "image_files.rds")
saveRDS(labels, "labels.rds")
saveRDS(y_train, "y_train.rds")
saveRDS(y_test, "y_test.rds")
saveRDS(train_indices, 'train_indices.rds')
saveRDS(test_indices, 'test_indices.rds')

train_images <- readRDS("train_images.rds")
test_images <- readRDS("test_images.rds")
image_files <- readRDS("image_files.rds")
labels <- readRDS("labels.rds")
y_train <- readRDS("y_train.rds")
y_test <- readRDS("y_test.rds")
train_indices <- readRDS("train_indices.rds")
test_indices <- readRDS("test_indices.rds")
x_test <- test_images
