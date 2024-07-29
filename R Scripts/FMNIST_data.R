library(keras)
library(aim)




#################################
#### Section 3.B FMNIST Data ####
#################################

# This script builds the autoencoder and visualises the results for the FMNIST data.
# You can bypass this script by loading the saved object directly, found at https://github.com/AlanInglis/AutoEncoder_Paper
# This object contains the AE and importance results. If loading saved object, skip to "Run AIM functions" section.
# NOTE: If loading object, the data must first be converted to the new format
# via the code provided.

# Load results and model --------------------------------------------------


# load data (replace with your path to data)
load("~/fmnist.RData")

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
set.seed(1701)
tensorflow::tf$random$set_seed(1701)


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
  epochs = 50,
  batch_size = 256,
  validation_data = list(x_test, x_test)
)




# Create Encoder Model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)
encoded_imgs <- predict(encoder, x_test)

# END OF MODEL BUILDING


# Plot reconstructed images -----------------------------------------------

# get decoced images
decoded_imgs <- predict(autoencoder, test_images)

# Set up the plotting area
par(mfrow = c(2, 8),    # 2 rows, 10 columns layout
    mar = c(0.5, 0.5, 2, 0.5),  # Margins around each plot
    oma = c(1, 0, 0, 0),  # Outer margins for the whole plot area
    mgp = c(1, 0.5, 0),  # Margin line for title, labels, and axis
    cex.main = 1.2)  # Main title size

# Figure 7:
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



# Figure 8:
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lm_sum_2,
                     class = "Trouser",
                     sort = T,
                     topX = 10,
                     flip_horizontal = F,
                     flip_vertical = F,
                     rotate = 0,
                     filter_zero = F)

# shiny plot
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)


# END OF SCRIPT

