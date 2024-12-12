

# This script selects the EMNIST data containing all the letters and selects only the Vowels

# load data (data can be found at https://github.com/AlanInglis/AutoEncoder_Paper)
df_test <- read.csv("/Users/alaninglis/Desktop/R Code/autoencoder_stuff/letters/EMNIST_Data/ltr_b_test.csv", sep = ',')
df_train <- read.csv("/Users/alaninglis/Desktop/R Code/autoencoder_stuff/letters/EMNIST_Data/ltr_b_train.csv", sep = ',')


# Define the labels
vowel_labels <- c(10, 14, 18, 24, 30, 36, 39)

# Replace labels in df_train
df_train[df_train[, 1] == 36, 1] <- 10
df_train[df_train[, 1] == 39, 1] <- 14

# Replace labels in df_test
df_test[df_test[, 1] == 36, 1] <- 10
df_test[df_test[, 1] == 39, 1] <- 14

# Filter df_train for the specified labels
df_train <- df_train[df_train[, 1] %in% vowel_labels, ]
df_test <- df_test[df_test[, 1] %in% vowel_labels, ]


write.csv(df_train, file = "ltr_train.csv", row.names = FALSE)
write.csv(df_test, file = "ltr_test.csv", row.names = FALSE)
