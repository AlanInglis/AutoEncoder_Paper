library(ggplot2)

plotAverageImportance <- function(digit_results, digit) {
  # Ensure that each element in digit_results is a data frame
  if (!all(sapply(digit_results, is.data.frame))) {
    stop("All elements in digit_results must be data frames")
  }

  # Calculate the average importance across all dimensions
  suppressWarnings(
    combined_df <- Reduce("+", digit_results) / length(digit_results)
  )
  combined_df$Var1 <- digit_results[[1]]$Var1
  combined_df$Var2 <- digit_results[[1]]$Var2

  # Create the plot
  p <- ggplot(combined_df, aes(Var1, Var2, fill = Freq)) +
    geom_tile() +
    scale_y_discrete(limits = rev(levels(combined_df$Var2))) +
    scale_fill_gradient(low = "white",
                        high = "red",
                        name = "Average MSE",
                        guide = guide_colorbar(
                          order = 1,
                          frame.colour = "black",
                          ticks.colour = "black"
                        ),
                        oob = scales::squish) +
    theme_void() +
    labs(title = paste0("Average Permutation Importance for Digit ", digit),
         x = "Pixel Row", y = "Pixel Column") +
    coord_fixed()

  return(p)
}

