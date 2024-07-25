


# Assuming ae_vimp_enc is the list of input pixel importance, convert column types
for (i in 1:length(ae_vimp_enc$Vimp)) {
  ae_vimp_enc$Vimp[[i]]$Var1 <- as.integer(ae_vimp_enc$Vimp[[i]]$Var1)
  ae_vimp_enc$Vimp[[i]]$Var2 <- as.integer(ae_vimp_enc$Vimp[[i]]$Var2)
}


# rename columns
for (i in 1:length(ae_vimp_enc$Vimp)) {
  names(ae_vimp_enc$Vimp[[i]]) <- c("Row", "Col", "Value")
}

# Assuming ae_vimp_decis the object of encoded dimension importance, rename column
colnames(ae_vimp_dec)[1] <- 'Class'
