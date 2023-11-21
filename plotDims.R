plotDimensions <- function(digitResults, digit, legendName = 'Error') {
  all_plots <- list()

  # get range for legend
  max_val <- max(sapply(digitResults, function(df) max(df$Freq)))
  min_val <- 0


  for (j in 1:length(digitResults)) {
    feature_importance_df <- digitResults[[j]]

    # create all plots
    p <- ggplot(feature_importance_df, aes(Var1, Var2, fill = Freq)) +
    geom_tile() +
    scale_y_discrete(limits = rev(levels(feature_importance_df$Var2))) +
    scale_fill_gradient(low = "white", high = "red",
                        limits = c(min_val, max_val),
                        name = legendName,
                        guide = guide_colorbar(
                          order = 1,
                          frame.colour = "black",
                          ticks.colour = "black"
                        ),
                        oob = scales::squish) +
    theme_void() +
    theme(panel.border = element_rect(colour = "black", fill=NA, linewidth=1)) +
    labs(title = paste0('No ', digit, ', Dim ', j)) +
    coord_fixed()

    all_plots[[j]] <- p
  }


  # Get legend
  legend <- cowplot::get_legend(all_plots[[1]])

  # remoe legend from all plots
  all_plots_no_leg <- lapply(all_plots, function(x) x + theme(legend.position = "none"))


  # plot on a grid
  n <- length(all_plots)
  nRow <- floor(sqrt(n))
  all_plot_grid <- gridExtra::arrangeGrob(grobs=all_plots_no_leg, nrow=nRow)
  suppressMessages(
    gridExtra::grid.arrange(all_plot_grid,
                            legend,
                            ncol = 2,
                            widths = c(10, 1))
  )
}


plotDimensions(digitResults = list_of_dfs[['2']], digit = 2, legendName = "MSE")
