
setwd("/Users/daniel/Documents/Arbeit/Uni/Thesis/spikingdentategyrus/scripts/")
df <- read.csv("weights.csv", header = TRUE)

length(df[2, ])
length(df[, 1])

big_m <- as.matrix(df)



for (i in seq(1, 1000, 10)){
  m <- matrix(big_m[i, 2:length(big_m[1, ])], 50, 20, byrow = T)
  plot_net(list(m), no_bias_in_input = T, scale_factor = 0.5, do_no_scale = T)
  Sys.sleep(1)
}
  

matrix(big_m[1, 2:length(big_m[6, ])], 20, 50, byrow = T)[0:5, 0:4]

dir.create("examples")
setwd("examples") 

#Creating countdown .png files from 10 to "GO!"
png(file="example%02d.png", width=600, height=600)
par(mar = c(0, 0, 0, 0))
for (i in seq(1, 1000, 10)){
  m <- matrix(big_m[i, 2:length(big_m[1, ])], 50, 20, byrow = T)
  plot_net(list(m), no_bias_in_input = T, scale_factor = 0.5)
  Sys.sleep(1)
}
dev.off()


plot(dnorm, xlim = c(-3, 3))







