This repository contains the `R` scripts that accompany the paper titled *"Permutation-Based Visualisation of Input and Encoded Space in Autoencoders"*. 

A description of the contents of each files is as follows: 

* **Data:** This folder contains three zip files. 

The zip file `AUDIO_MNIST_DATA.zip` contains the processed spectrograms of the audio MNIST data. The data was originally sourced from [here](https://github.com/soerenab/AudioMNIST). 
The zip file `col_images.zip`, contains the processed spectrograms images of the audio MNIST data.
The zip file `EMNIST_DATA.zip`, contains the csv files for the letters data, filtered for just the vowels (a,e,i,o,u)

* **Paper:** This folder contains a `.pdf` of the finalised article.

* **R_Scripts:** This folder contains the R scripts used to generate the results in the article. 

* **Saved_Objects:** This folder contains data and model objects that correspond to the results obtained from running the R scripts in the
**R_Scripts** folder and which are seen in the finalised article. 

* **Preprocessing_data_scripts** These scripts were used to generally preprocess the data (e.g., covert spectrogram images into csv files etc.) and 
are not needed to run the scripts in **R_Scripts**.



# Loading saved data and model objects
To load the saved EMNIST data and model, navigate to the `Saved_Objects/EMNIST` folder/file and download the contents. 
Then in an R script run: 

```r
# Install devtools if you haven't already
#install.packages("devtools")

# Install the aim package
#devtools::install_github("AlanInglis/AIM")

library(keras)
library(aim)

# load data and model (replace with your path to data)
load("~/emnist_model.RData")
autoencoder <- load_model_tf("~/autoencoder_model_emnist.keras")
encoder <- load_model_tf("~/encoder_model_emnist.keras")

# Run shiny app (NOTE: sort output by importance, rotation = 1 for correct orientation)
shiny_vimp(input_vimp = ae_vimp_enc, encoded_vimp = ae_vimp_dec, direction_vimp = lm_sum_2)
```

# How to load data
To load a specific preprocessed dataset (e.g., the training and test data for the EMNIST example), navigate to the `DATA/EMNIST.zip` folder/file 
and download and unzip. Then in R do:

```r
# load data (replace with your path to data)
df_train <- read.csv('~/Saved_Objects/EMNIST/ltr_train.csv', sep = ',')
df_test <- read.csv('~/Saved_Objects/EMNIST/ltr_test.csv', sep = ',')
```
Now the data can be used to build an autoencoder model, like those outlined in the scripts in **R_Scripts**.











