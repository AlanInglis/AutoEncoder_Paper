allDigits_allDims <- function() {
  all_results <- list()

  for (digit in 0:9) {
    print(paste("Processing digit:", digit))
    model_results <- autoBuild(numSel = digit)
    importance_results <- permImp(model_results)
    all_results[[as.character(digit)]] <- importance_results
  }

  return(all_results)
}


list_of_dfs <- allDigits_allDims()

dim_plots <- list()
for(digit in 0:9){
  digitResults <- list_of_dfs[[as.character(digit)]]
  dim_plots[[as.character(digit)]] <- plotDimensions(digitResults, digit)
}


plot_object <- dim_plots[[1]]  # Retrieve the first plot object
grid::grid.draw(plot_object)



