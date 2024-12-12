# Install devtools if you haven't already
#install.packages("devtools")

# Install the aim package
#devtools::install_github("AlanInglis/AIM")

library(keras)
library(aim)

###############################
#### Section 4 EMNIST Data ####
###############################
rm(list = ls())

# This script builds the autoencoder and visualises the results for the EMNIST data shown in Section 4 of our paper.
# You can bypass the fitting of the AE model and permutation importance process by
# loading the saved data objects directly, found at https://github.com/AlanInglis/AutoEncoder_Paper/Saved_Objects/EMNIST
# These saved data objects contains the AE and importance results.
# Lines 25-35 below will load the saved objects and display the results

# If you wish to fit the AE and recalculate the results, begin at the "Data and Autoencoder"
# section (line 38).


# Load results, model, and visualise --------------------------------------------------

# load data (replace with your path to data)
load("~/emnist_model.RData")
autoencoder <- load_model_tf("~/autoencoder_model_emnist.keras")
encoder <- load_model_tf("~/encoder_model_emnist.keras")

# Figure 5 (NOTE: sort output by importance, rotation = 1)
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

# load data (data can be found at https://github.com/AlanInglis/AutoEncoder_Paper/Data/EMNIST_DATA.zip)
df_train <- read.csv('~/Saved_Objects/EMNIST/ltr_train.csv', sep = ',')
df_test <- read.csv('~/Saved_Objects/EMNIST/ltr_test.csv', sep = ',')

# first column as labels
train_images <- as.matrix(df_train[, -1])  # Remove labels
test_images <- as.matrix(df_test[, -1])    # Remove labels

# Normalise pixel values
train_images <- train_images / 255
test_images <- test_images / 255

input_size <- ncol(train_images)  # 784 for 28x28 images
encoding_dim <- 32  # Size of the encoded representation
edim <- encoding_dim

# autoencoder
input_img <- layer_input(shape = c(input_size))
encoded <- input_img %>%
  layer_dense(units = encoding_dim, activation = 'relu')

decoded <- encoded %>%
  layer_dense(units = input_size, activation = 'sigmoid')

# autoencoder model
autoencoder <- keras_model(inputs = input_img, outputs = decoded)


# Compile model
autoencoder %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy'
)


# fit autoencoder
autoencoder %>% fit(
  train_images, train_images,
  epochs = 100,
  batch_size = 256,
  shuffle = F,
  validation_data = list(test_images, test_images)
)

# Create Encoder Model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)
decoded_imgs <- predict(autoencoder, test_images)

# END OF MODEL BUILDING


# Plot reconstructed images -----------------------------------------------

# Figure 4 in paper:

# get decoded images
decoded_imgs <- predict(autoencoder, test_images)

# Set up the plotting area
par(mfrow = c(2, 8),    # 2 rows, 8 columns layout
    mar = c(0.5, 0.5, 2, 0.5),  # Margins around each plot
    oma = c(1, 0, 0, 0),  # Outer margins for the whole plot area
    mgp = c(1, 0.5, 0),  # Margin line for title, labels, and axis
    cex.main = 1.2)  # Main title size

# Plot original images on the top row
for (i in 17:24) {
  img <- test_images[i,]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- img[, ncol(img):1]  # Flip horizontally
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Input", i-16))
}

# Plot reconstructed images on the bottom row
for (i in 17:24) {
  img <- decoded_imgs[i,]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- img[, ncol(img):1]
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Output", i-16))
}



# END OF PLOT RECONSTRUCTED


# Labels ------------------------------------------------------------------

# Get labels
labels_test <- df_test$X41

# Convert numerical labels to corresponding vowel
number_to_letter <- c("10" = "A", "14" = "E", "18" = "I", "24" = "O", "30" = "U")

# Replace numbers with corresponding letters
labels_test_letters <- factor(labels_test, levels = names(number_to_letter), labels = number_to_letter)

# Convert to character
labels_test_letters <- as.character(labels_test_letters)

# END OF LABELS

# Run AIM functions -------------------------------------------------------

# input pixel importance
ae_vimp_enc <- vimp_input(encoder = encoder, test_data = test_images, num_permutations = 4)

# encoded dimension importance
ae_vimp_dec <- vimp_encoded(encoder = encoder,
                            test_data =  test_images,
                            test_labels = labels_test_letters,
                            autoencoder = autoencoder,
                            classes = c("A", "E", "I", "O", "U"),
                            num_permutations = 4)

# directional importance
lm_sum_2  <- vimp_direction(encoder = encoder,
                            test_data =  test_images)



# Figure 5:
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lm_sum_2,
                     class = "I",
                     sort = T,
                     topX = 15,
                     flip_horizontal = F,
                     flip_vertical = F,
                     rotate = 1,
                     filter_zero = F)

# shiny plot (NOTE: sort output by importance, rotation = 1)
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)





# SAVE OBJECTS ------------------------------------------------------------
# save_model_tf(autoencoder, "autoencoder_model_emnist.keras")
# save_model_tf(encoder, "encoder_model_emnist.keras")
# save.image(file = 'emnist_model.RData')







