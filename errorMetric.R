#' errorMetric
#'
#' @description Which error metric to apply
#'
#' @param actual Actual data (in this case, the original encoded image)
#' @param predicted Predicted data.
#' @param MSE LOGICAL. If TRUE, then MSE. If FALSE then RMSE
#'



# Error metric ------------------------------------------------------------

errorMetric <- function(actual, predicted, MSE = TRUE) {
  err <-  mean((actual - predicted)^2)
  if(MSE){
    return(err)
  }else{
    err <- sqrt(err)
    return(err)
  }
}

