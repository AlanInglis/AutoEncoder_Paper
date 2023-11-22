# AutoEncoder
Testing pixel importance and interactions in autoencoders


In the following, a brief tutorial on how to create the plots is given.

```
allDigits_allDims <- function() {
  all_results <- list()

  for (digit in 0:9) {
    print(paste("Processing digit:", digit))
    model_results <- autoBuild(numSel = digit)
    importance_results <-permImp(encoderModel =model_results,
                                  num_permutations = 4,
                                  MSE = TRUE)
    all_results[[as.character(digit)]] <- importance_results
  }

  return(all_results)
}
```

In the above function, we are creating a list of data frames, where each list element corresponds to a single digit. Inside each element, is a list of 32 data frames, with each data frame corresponding to a particular dimension. The break down of the above function is as follows:

1. The `autoBuild` function is used to build the autoencoder. The `numSel` argument selects a specific digit to analyse. 
2. The `permImp` function then calculates the permutation importance for all the built encoders. 
3. The list of data frames is stored in the  `all_results` object, which is returned to the user. 

Running that function as displayed above, gives us the list of data frames for every digit:

```
list_of_dfs <- allDigits_allDims()
```

Then to plot each digit, we create another list. This time, a list of plots where each plot shows all dimensions of a particular digit:


```
dim_plots <- list()
for(digit in 0:9){
  digitResults <- list_of_dfs[[as.character(digit)]]
  dim_plots[[as.character(digit)]] <- plotDimensions(digitResults, digit)
}
```
The `dim_plots` object is now a list of `gtable` objects. To select and display a specific digit, we can use:


```
plot_object <- dim_plots[[1]]  # Display digit zero
grid::grid.draw(plot_object)
```


Additionally, we can select a specific digit and average across all the dimensions of that digit to show an average plot:

```
avg_plot <- plotAverageImportance(list_of_dfs[['0']], 0)  # For digit zero
print(avg_plot)
```








