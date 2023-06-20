## Making some nice looking plots

require(plotrix) # For CI plots

rm(list = ls())
setwd("/Users/daniel/Documents/Arbeit/Uni/Thesis/spikingdentategyrus/")

path_to_data <- "scripts/final_data/"

conditions <- c("base", 
                "sparse_015", 
                "sparse_015_015_neurogenesis",
                "sparse_015_015_neurogenesis_threshold")

legend_labels <- c("Base", 
                   "Sparse", 
                   "+ Neurogenesis",
                   "+ Neurogenesis + Threshold")

# dev.off()
# pdf("sparse_nosparse_threshold.pdf", width = 8, height = 6) # turn on to save plot

par(mfrow = c(1, 2))
par(mgp=c(1.7,0.6,0), mar = c(5,3,3,1))

lw <- 2
cex_axis <- 0.85
width_ci <- 0.008
xlim <- c(0.5, 10.3)
ylim <- c(0.8, 1)

cex_main <- 1
cex_lab <- 0.9
  
points <- c(1, 4, 5, 2)
cex_points <- 1
colors <- c("firebrick2", "steelblue3", "darkolivegreen3", "lightgoldenrod2")



#### RETROACTIVE INTERFERENCE ####

plot(NA, xlim = xlim, ylim = ylim, frame = F, 
     axes = F, xlab = "Group", ylab = "Percent match", main = "Retroactive Interference", 
     cex.lab = cex_lab, cex.main = cex_main)
axis(2, at = seq(0.75, 1, 0.05), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)
axis(1, at = seq(1, 10, 1), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)


counter <- 1
space <- -0.2
space_increment <- 0.2

for (condition in conditions){

  path <- paste0(path_to_data, condition)
  
  df_percent_match_within <- t(read.csv(paste(path, "_pm_table_within.csv", sep = ""), header = TRUE)[-1])      # proactive interference # nolint: line_length_linter.

  means_within <- apply(df_percent_match_within, 2, mean)
  sd_within <- apply(df_percent_match_within, 2, sd)
  n_within <- nrow(df_percent_match_within)
  
  #calculate margin of error
  margin_within  <- qt(0.975,df=n_within-1)*sd_within/sqrt(n_within)
  
  #calculate lower and upper bounds of confidence interval
  low_within <- means_within - margin_within 
  
  high_within <- means_within + margin_within 
  
  plotCI((1:10) + space , means_within, ui=high_within, li=low_within, add = TRUE, lwd = lw/1.5, 
         pch = NA, lend = 1, sfrac = width_ci)
  lines((1:10) + space, means_within, col = colors[counter], lty = 4, lend = 1, lw = lw/1.5)
  points((1:10) + space, means_within, lw = lw, pch = points[counter], col = colors[counter], 
         cex = cex_points, lend = 1)
  
  space <- space + space_increment
  counter <- counter + 1
}

legend("bottomleft", legend_labels, 
       col = colors, pch = points, 
       bty = "n", bg = F, cex = 0.8)


### PROACTIVE INTERFERENCE ###

plot(NA, xlim = xlim, ylim = ylim, frame = F, 
     axes = F, xlab = "Group", ylab = "Percent match", main = "Proactive Interference", 
     cex.lab = cex_lab, cex.main = cex_main)
axis(2, at = seq(0.75, 1, 0.05), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)
axis(1, at = seq(1, 10, 1), lwd = lw, lend = 1, font = 3, cex.axis = cex_axis)


counter <- 1
space <- -0.2
space_increment <- 0.2

for (condition in conditions){
  
  path <- paste0(path_to_data, condition)
  
  df_percent_match_between <- t(read.csv(paste(path, "_pm_table_between.csv", sep = ""), header = TRUE)[-1])      # retroactive interference # nolint: line_length_linter.
  
  means_between <- apply(df_percent_match_between, 2, mean)
  sd_between <- apply(df_percent_match_between, 2, sd)
  n_between <- nrow(df_percent_match_between)
  
  #calculate margin of error
  margin_between  <- qt(0.975,df=n_between-1)*sd_between/sqrt(n_between)
  
  #calculate lower and upper bounds of confidence interval
  low_between <- means_between - margin_between
  
  high_between <- means_between + margin_between 
  
  plotCI((1:10) + space , means_between, ui=high_between, li=low_between, add = TRUE, lwd = lw/1.5, 
         pch = NA, lend = 1, sfrac = width_ci)
  lines((1:10) + space, means_between, col = colors[counter], lty = 4, lend = 1, lw = lw/1.5)
  points((1:10) + space, means_between, lw = lw, pch = points[counter], col = colors[counter], 
         cex = lw/2, lend = 1)
  
  space <- space + space_increment
  counter <- counter + 1
}

legend("bottomleft", legend_labels, 
       col = colors, pch = points, 
       bty = "n", bg = F, cex = 0.8)


#dev.off()







