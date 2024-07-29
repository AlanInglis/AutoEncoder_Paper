library(keras)
library(aim)
library(dplyr)



#################################
#### Section 3.A EMNIST Data ####
#################################

# This script builds the autoencoder and visualises the results for the EMNIST data.
# You can bypass this script by loading the saved object directly, found at https://github.com/AlanInglis/AutoEncoder_Paper
# This object contains the AE and importance results. If loading saved object, skip to "Run AIM functions" section.
# NOTE: If loading object, the data must first be converted to the new format
# via the code provided.

# Load results and model --------------------------------------------------


# load
load("/Users/alaninglis/Desktop/Autoencoder Paper/saved_objects/emnist.RData")

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

# load data
df_test <- read.csv("/Users/alaninglis/Desktop/autoencoder_stuff/letters/ltr_b_test.csv", sep = ',')
df_train <- read.csv("/Users/alaninglis/Desktop//autoencoder_stuff/letters/ltr_b_train.csv", sep = ',')


# Filter df_train for letters only (capitals only)
# letter mapping: A = 10, B= 11, ..., Z = 35
df_train <- df_train[df_train$X45 %in%  c(10,14,18,24,30),]

# Filter df_test for letters only
df_test <- df_test[df_test$X41 %in%  c(10,14,18,24,30),]

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
  shuffle = T,
  validation_data = list(test_images, test_images)
)



# Create Encoder Model
encoder <- keras_model(inputs = autoencoder$input, outputs = encoded)
decoded_imgs <- predict(autoencoder, test_images)

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

# Figure 5:
for (i in 1:8) {
  # Original image
  img <- test_images[i,]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- img[, ncol(img):1]  # Flip horizontally
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Input", i))

  # Reconstructed image
  img <- decoded_imgs[i,]
  img <- matrix(img, ncol = 28, byrow = TRUE)
  img <- img[, ncol(img):1]
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste("Output", i))
}


# END OF PLOT RECONSTRUCTED


# Labels ------------------------------------------------------------------

# Get labels
labels_test <- df_test$X41

# Convert numberical labels to corresponding vowel
number_to_letter <- c("10" = "A", "14" = "E", "18" = "I", "24" = "O", "30" = "U")

# Replace numbers with corresponding letters
labels_test_letters <- factor(labels_test, levels = names(number_to_letter), labels = number_to_letter)

# Convert to character
labels_test_letters <- as.character(labels_test_letters)

# END OF LABELS


# Save -----------------------------------------------------------

# save
# save.image(file = 'emnist.RData')

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



# Figure 6:
plot_input_direction(input_vimp = ae_vimp_enc,
                     encoded_vimp = ae_vimp_dec,
                     direction_vimp = lmdef,
                     class = "A",
                     sort = T,
                     topX = 10,
                     flip_horizontal = F,
                     flip_vertical = F,
                     rotate = 1,
                     filter_zero = F)

# shiny plot
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)


# END OF SCRIPT

