## Making some nice looking plots

rm(list = ls())
setwd("/Users/daniel/Documents/Arbeit/Uni/Thesis/spikingdentategyrus/")

require(plotrix) # For CI plots

path_to_data <- "scripts/final_data/"
file_name <- "test_separation"

# plot is only shown if save_file == F
save_file <- F

Base <- "#EE4000" 

neurogenesis <- "darkseagreen3"
neurogenesis_leak <- "#8B3626"
neurogenesis_turnover <- "lightsteelblue2"

sparse <- "#FF8C00"
sparse_neurogenesis_full_connectivity <- "#548B54"
sparse_neurogenesis <- "royalblue4"

all_sparse_neurogenesis <- "#CDB5CD"
sparse_neurogenesis <- "#00E5EE"
sparse_neurogenesis_turnover <- "#1E90FF"
sparse_neurogenesis_leak <- "#8B8989"
sparse_neurogenesis_leak_turnover <- "yellow1"

figure <- 4

if (figure == 1){
  file_name <- "no_sparse_separation"
  
  conditions <- c("base",
                  "015_neurogenesis",
                  "015_neurogenesis_leak_adj",
                  "015_neurogenesis_turnover"
  )
  
  legend_labels <- c("base",
                     "neurogenesis",
                     "ng. + hyper-excitable imGCs",
                     "ng. + apoptosis"
  )
  
  colors <- c(Base, neurogenesis, neurogenesis_leak, neurogenesis_turnover)
}

################

if (figure == 2){
  file_name <- "base_sparse_separation"
  
  conditions <- c("base",
                  "sparse_015",
                  "sparse_015_015_neurogenesis"
  )
  
  legend_labels <- c("base",
                     "all sparse firing",
                     "ng (mGCs sp. fir.) + imGCs sp. conn."
  )
  
  colors <- c(Base, sparse, sparse_neurogenesis, sparse_neurogenesis_leak, sparse_neurogenesis_turnover)
}

################

if (figure == 3){
  file_name <- "sparse_firing_connectivity_separation"
  conditions <- c("sparse_015_015_neurogenesis",
                  "sparse_015_015_neurogenesis_full_connectivity",
                  "all_sparse_015_015_neurogenesis"
                  
  )
  
  legend_labels <- c("ng (mGCs sp. fir.) + imGCs sp. conn.",
                     "ng (mGCs sp. fir.) + imGCs f. conn.",
                     "ng (mGCs + imGCs sparse firing)"
                     
  )
  
  colors <- c(sparse_neurogenesis, sparse_neurogenesis_full_connectivity, all_sparse_neurogenesis)
}

################

if (figure == 4){
  file_name <- "sparse_turnover_leak_separation"
  
  conditions <- c("sparse_015_015_neurogenesis",
                  "sparse_015_015_neurogenesis_leak_adj",
                  "sparse_015_015_neurogenesis_turnover", 
                  "sparse_015_015_neurogenesis_leak_adj_turnover"
  )
  
  legend_labels <- c("neurogenesis (mGCs sparse firing)",
                     "ng (mGCs sp. fir.) + hp.-exc. imGCs",
                     "ng (mGCs sp. fir.) + apoptosis", 
                     "all"
  )
  colors <- c(sparse_neurogenesis, sparse_neurogenesis_leak, sparse_neurogenesis_turnover, sparse_neurogenesis_leak_turnover)
}

################

if (save_file){
  pdf(paste("plots/", file_name, ".pdf", sep = ""), width = 5, height = 5) # turn on to save plot
}

par(mgp=c(2,0.6,0), mar = c(4,3.5,3,0.5), mfrow = c(1, 1),
    family = "serif")


confidence_intervals <- T

lw <- 2
cex_axis <- 1.3
width_ci <- 0.013
xlim <- c(0, 50)
ylim <- c(0, 95)

cex_main <- 1.2
cex_lab <- 1.2

points <- c(21, 22, 23, 24, 25)
cex_points <- 1
opacity <- 1
# colors <- c(rgb(0.93, 0.17, 0.17, opacity), #firebrick2
#             rgb(0.31, 0.58, 0.8, opacity),  # steelblue3
#             rgb(0.64, 0.8, 0.35, opacity),  # darkolivegreen3
#             rgb(0.93, 0.86, 0.51, opacity)) # lightgoldenrod2

cex_legend <- 0.8
pos_legend <- c(25, 30)

#### PATTERN SEPARATION ####

plot(NA, xlim = xlim, ylim = ylim, frame = F, 
     axes = F, xlab = "Distance input", ylab = "Distance hidden representation", main = "Pattern separation", 
     cex.lab = cex_lab, cex.main = cex_main)
axis(2, at = seq(0, ylim[2]+1, 10), labels = seq(0, ylim[2]+1, 10), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)
axis(1, at = seq(0, xlim[2]+1, 10), lwd = lw, lend = 1, font = 3, 
     cex.axis = cex_axis)

counter <- 1
space <- -0.1
space_increment <- 0.05

for (condition in conditions){
  
  path <- paste0(path_to_data, condition)
  
  df_x <- t(as.matrix(read.csv(paste(path, "_unique_x.csv", sep = ""), header = TRUE)[-1]))
  df_y <- as.matrix(read.csv(paste(path, "_pattern_separation.csv", sep = ""), header = TRUE)[-1])
  
  mean_y <- apply(df_y, 2, mean, na.rm = TRUE)
  sd_y <- apply(df_y, 2, sd, na.rm = TRUE)
  n_y <- colSums(!is.na(df_y))
  
  margin  <- qt(0.975,df=n_y-1)*sd_y/sqrt(n_y)
  
  #calculate lower and upper bounds of confidence interval
  low <- mean_y - margin
  
  high <- mean_y + margin 
  
  if (confidence_intervals){
    plotCI(df_x + space, mean_y, ui=high, li=low, add = TRUE, lwd = lw/1.5, 
           pch = NA, lend = 1, sfrac = width_ci)
  }
  segments(45, mean(mean_y[2:length(mean_y)]), 50, mean(mean_y[2:length(mean_y)]), col = colors[counter], lty = 1, lw = lw*3, lend = 1)
  segments(0, mean(mean_y[2:length(mean_y)]), 50, mean(mean_y[2:length(mean_y)]), col = colors[counter], lty = 3, lw = lw/2, lend = 1)
  points(df_x + space, mean_y, lw = lw, pch = points[counter], col = colors[counter],
         cex = cex_points, lend = 1, bg = colors[counter])
  #(mean_y, add = T, boxwex=10, col = colors[counter], axes = F)
  # lines(df_x, mean_y, lw = lw, pch = points[counter], col = colors[counter], 
  #       cex = cex_points, lend = 1)
  counter <- counter + 1
  space <- space + space_increment
}
#abline(0, 1, col = "black", lty = 3, lw = lw/2, lend = 1)
#segments(0, 0, 80, 80, col = "black", lty = 3, lw = lw/2, lend = 1)
legend("bottomright", 
       #pos_legend[1], pos_legend[2], 
       legend_labels, 
       col = colors, pch = points, 
       bty = "n", pt.bg = colors, cex = cex_legend)

if (save_file){
  dev.off()
}


