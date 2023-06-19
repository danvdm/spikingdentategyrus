## Making some nice looking plots

rm(list = ls())
setwd("/Users/daniel/Documents/Arbeit/Uni/Thesis/spikingdentategyrus/")

require(plotrix) # For CI plots

path_to_data <- "scripts/final_data/"

conditions <- c("base" 
                ,"sparse_015"
                ,"sparse_015_015_neurogenesis" 
                )

legend_labels <- c("Base" 
                   ,"Sparse" 
                   ,"+Neurogenesis"
                   )

#dev.off()
#pdf("base_vs_ng.pdf", width = 8, height = 6) # turn on to save plot

par(mfrow = c(1, 1))
par(mgp=c(1.7,0.6,0), mar = c(5,3,3,1))

lw <- 2
cex_axis <- 0.85
width_ci <- 0.01
xlim <- c(0, 55)
ylim <- c(0, 70)

cex_main <- 1
cex_lab <- 0.9
  
points <- c(21, 22, 25, 24)
cex_points <- 0.5
opacity <- 1
colors <- c(rgb(0.93, 0.17, 0.17, opacity), #firebrick2
            rgb(0.31, 0.58, 0.8, opacity),  # steelblue3
            rgb(0.64, 0.8, 0.35, opacity),  # darkolivegreen3
            rgb(0.93, 0.86, 0.51, opacity)) # lightgoldenrod2

colors_legend <- c("firebrick2", "steelblue3", "darkolivegreen3", "lightgoldenrod2")
pos_legend <- c(30, 15)

#### PATTERN SEPARATION ####

plot(NA, xlim = xlim, ylim = ylim, frame = F, 
     axes = F, xlab = "Distance input", ylab = "Distance hidden representation", main = "Pattern separation", 
     cex.lab = cex_lab, cex.main = cex_main)
axis(2, at = seq(0, ylim[2], 10), labels = seq(0, ylim[2], 10), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)
axis(1, at = seq(0, xlim[2], 10), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)

counter <- 1
space <- -0.2
space_increment <- 0.2

for (condition in conditions){
  
  path <- paste0(path_to_data, condition)
  
  df_x <- t(read.csv(paste(path, "_unique_x.csv", sep = ""), header = TRUE)[-1])  
  df_y <- read.csv(paste(path, "_pattern_separation.csv", sep = ""), header = TRUE)[-1]
  
  mean_y <- apply(df_y, 2, mean, na.rm = TRUE)
  sd_y <- apply(df_y, 2, sd, na.rm = TRUE)
  n_y <- colSums(!is.na(df_y))
  
  margin  <- qt(0.975,df=n_y-1)*sd_y/sqrt(n_y)
  
  #calculate lower and upper bounds of confidence interval
  low <- mean_y - margin
  
  high <- mean_y + margin 
  
  plotCI(df_x , mean_y, ui=high, li=low, add = TRUE, lwd = lw/1.5, 
         pch = NA, lend = 1, sfrac = width_ci)
  points(df_x, mean_y, lw = lw, pch = points[counter], col = colors[counter], 
         cex = cex_points, lend = 1, bg = colors[counter])
  counter <- counter + 1
}
abline(0, 1, col = "gray", lty = 4, lw = lw, lend = 1)
legend(pos_legend[1], pos_legend[2], legend_labels, 
       col = colors_legend, pch = points,
       bty = "n", bg = colors_legend, cex = 1)

#dev.off()


