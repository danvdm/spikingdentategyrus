## Making some nice looking plots

rm(list = ls())
setwd("/Users/daniel/Documents/Arbeit/Uni/Thesis/spikingdentategyrus/")

conditions <- c("base", 
                "sparse_015", 
                "sparse_015_015_neurogenesis"
                )

legend_labels <- c("Base", 
                   "Sparse" 
                   ,"+Neurogenesis"
                   )

#dev.off()
#pdf("base_vs_ng.pdf", width = 8, height = 6) # turn on to save plot

par(mfrow = c(1, 1))
par(mgp=c(1.7,0.6,0), mar = c(5,3,3,1))

lw <- 2
cex_axis <- 0.85
width_ci <- 0.008
xlim <- c(0, 55)
ylim <- c(0, 150)

cex_main <- 1
cex_lab <- 0.9
  
points <- c(1, 4, 5, 2)
cex_points <- 1
opacity <- 0.2
colors <- c(rgb(0.93, 0.17, 0.17, opacity), #firebrick2
            rgb(0.31, 0.58, 0.8, opacity),  # steelblue3
            rgb(0.64, 0.8, 0.35, opacity),  # darkolivegreen3
            rgb(0.93, 0.86, 0.51, opacity)) # lightgoldenrod2

colors_legend <- c("firebrick2", "steelblue3", "darkolivegreen3", "lightgoldenrod2")
pos_legend <- c(0, 135)

#### PATTERN SEPARATION ####

plot(NA, xlim = xlim, ylim = ylim, frame = F, 
     axes = F, xlab = "Distance input", ylab = "Distance hidden representation", main = "Pattern separation", 
     cex.lab = cex_lab, cex.main = cex_main)
axis(2, at = seq(0, ylim[2], 20), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)
axis(1, at = seq(0, xlim[2], 10), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)


counter <- 1
space <- -0.2
space_increment <- 0.2

for (condition in conditions){

  path <- paste0("scripts/final_data_second_attempt/", condition)
  
  df_distances_in <- t(read.csv(paste(path, "_distances_in.csv", sep = ""), header = TRUE)[-1])      # proactive interference
  df_distances_out <- t(read.csv(paste(path, "_distances_out.csv", sep = ""), header = TRUE)[-1])      # proactive interference
  
  df_distances_in_flat <- do.call("c", list(df_distances_in[df_distances_in != 0]))
  df_distances_out_flat <- do.call("c", list(df_distances_out[df_distances_in != 0]))
  
  model <- lm(df_distances_out_flat ~ df_distances_in_flat)
  
  points(df_distances_in_flat, df_distances_out_flat, lw = lw, pch = points[counter], col = colors[counter], 
         cex = cex_points, lend = 1)
  abline(model, col = colors_legend[counter], lty = 4, lw = lw, lend = 1)
  
  counter <- counter + 1
}

legend(pos_legend[1], pos_legend[2], legend_labels, 
       col = colors_legend, pch = points, 
       bty = "n", bg = F, cex = 1)

#dev.off()