# Install devtools if you haven't already
#install.packages("devtools")

# Install the aim package
#devtools::install_github("AlanInglis/AIM")

library(keras)
library(aim)



################################
#### Section 2 MNIST Data ####
################################

# This script builds the autoencoder and visualises the results for the MNIST data.
# You can bypass the fitting of the AE model and permutation importance process by
# loading the saved data object directly, found at https://github.com/AlanInglis/AutoEncoder_Paper
# This saved data object contains the AE and importance results. If loading saved object,
# the data must first be converted to the new format via the code in the "Load results and model"
# section provided directly below. Following this, skip to "Run AIM functions" section and plot
# the results. If you wish to fit the AE and recalculate the results, begin at the "Data and Autoencoder"
# section.


# Load results and model --------------------------------------------------


# load data (replace with your path to data)
load("/Users/alaninglis/Desktop/AutoEncoder/Saved_Objects/mnist_ex.RData")


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
colnames(ae_vimp_dec)[1] <- 'Class'

# END OF LOAD RESULTS AND CONVERSION

# Data and Autoencoder ----------------------------------------------------

# set seed
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

# END OF PLOT RECONSTRUCTED

# Save -----------------------------------------------------------

# save
# save.image(file = 'mnist_ex.RData')


# Run AIM functions -------------------------------------------------------

# input pixel importance
ae_vimp_enc <- vimp_input(encoder = encoder, test_data = x_test, num_permutations = 4)
ae_vimp_enc
# encoded dimension importance
ae_vimp_dec <- vimp_encoded(encoder = encoder,
                            test_data =  x_test,
                            test_labels = y_test,
                            autoencoder = autoencoder,
                            classes = considered_digits,
                            num_permutations = 4)
ae_vimp_dec
# directional importance
lm_sum_2  <- vimp_direction(encoder = encoder,
                            test_data =  x_test)
lm_sum_2


# plot
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lm_sum_2,
                     class = 0,
                     sort = T,
                     topX = 10,
                     flip_horizontal = T,
                     flip_vertical = F,
                     rotate = 0,
                     filter_zero = F)

# Figure 4
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)


# END OF SCRIPT

