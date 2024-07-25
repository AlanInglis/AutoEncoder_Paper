


library(tuneR)
library(seewave)


# plot palette function ---------------------------------------------------


# Define a function to create a custom color palette in R
create_custom_palette <- function(colors) {
  # Validate that the input is a named vector
  if (!is.character(colors) || is.null(names(colors))) {
    stop("Please provide a named character vector with color names and hex codes.")
  }
  
  # Create the color palette
  palette(colors)
  
  # Function to return the palette when called
  return(function(n) {
    if (n > length(colors)) {
      warning("Requested number of colors exceeds the number in the palette, recycling colors.")
    }
    return(colors[1:n])
  })
}

# Example usage of the function
my_palette <- create_custom_palette(c("deep white" = "black", "deep white2" = "black"))

# Set the parent directory path
parent_directory <- "/Users/alaninglis/Desktop/autoencoder_stuff/MNIST_speech/soerenab AudioMNIST master data"

# Assuming folders are named as "01", "02", ..., "60"
folder_names <- sprintf("%02d", 1:60) # Generates folder names from 01 to 60

# Loop through each folder
for(folder_name in folder_names) {
  
  # Construct the path to the current folder
  current_folder_path <- file.path(parent_directory, folder_name)
  setwd(current_folder_path)
  
  wav_files <- list.files(pattern = "\\.wav$")
  
  # Loop through each WAV file in the current folder
  for(file_name in wav_files) {
    
    print(paste0("File ", file_name, " Converted"))
    
    # Assuming 'readWave', 'fir', 'spectro', and other necessary libraries are loaded
    
    bcch <- readWave(file_name)
    # filter high and low freqs
    bcch <- fir(wave = bcch, from = 1000, to = 8000, bandpass = TRUE, output = "Wave")
    
    # Extract details from the file name
    parts <- strsplit(file_name, "_")[[1]]
    number_spoken <- parts[1]
    person <- parts[2]
    sample_number <- sub("\\.wav$", "", parts[3])
    
    # Define the output path for the plots
    output_path <- "/Users/alaninglis/Desktop/autoencoder_stuff/test_images"
    plot_file_name <- sprintf("%s_%s_%s.png", number_spoken, person, sample_number)
    
    # Create and save the plot
    png(file.path(output_path, plot_file_name),  width = 32, height = 32)
    par(mar = c(0, 0, 0, 0), oma = c(0, 0, 0, 0)) 
    #par(mar = c(1, 1, 1, 1), bty = "n", xaxs = "i", yaxs = "i", xaxt = "n", yaxt = "n")
    spectro(wave = bcch, osc = FALSE,
            flim = c(0.1, 8), tlab = '',
            flab = '', flog = FALSE, scale = FALSE,
            cont = FALSE, axisX = FALSE, axisY = FALSE,
            grid = FALSE,
            collevels = seq(-30, 0, 15),
            palette = my_palette,
            colaxis="white",
            collab="white")
    dev.off()
  }
}
