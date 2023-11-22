library(keras)
library(ggplot2)


allDigits_allDims <- function() {
  all_results <- list()

  for (digit in 0:9) {
    print(paste("Processing digit:", digit))
    model_results <- autoBuild(numSel = digit)
    importance_results <- permImp(encoderModel =model_results,
                                  num_permutations = 4,
                                  MSE = FALSE)
    all_results[[as.character(digit)]] <- importance_results
  }

  return(all_results)
}


list_of_dfs <- allDigits_allDims()

dim_plots <- list()
for(digit in 0:9){
  print(paste0('Number ', digit, ' Converted'))
  digitResults <- list_of_dfs[[as.character(digit)]]
  dim_plots[[as.character(digit)]] <- plotDimensions(digitResults, digit)
}


plot_object <- dim_plots[[4]]  # Retrieve the first plot object
grid::grid.draw(plot_object)


for(k in 0:9){
  avg_plot <- plotAverageImportance(list_of_dfs[[as.character(k)]], k)  # select digit
  print(avg_plot)
}
avg_plot <- plotAverageImportance(list_of_dfs[['9']], 9)  # select digit
avg_plot



# -------------------------------------------------------------------------
# using full dataset (ie, not filtering digits)
model_results <- autoBuild(numSel = NULL)
importance_results <- permImp(encoderModel = model_results,
                              num_permutations = 4,
                              MSE = TRUE)


importance_results




