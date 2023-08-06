rm(list = ls())

setwd("/Users/daniel/Documents/Arbeit/Uni/Thesis/spikingdentategyrus/")

save = T

# Gompertz

g <- function(x, s = 1){
  exp(-exp(-s * x))
}
sg <- seq(-1, 1, length.out = 100)
xg <- g(sg)

g(3)

if (save){
  pdf("gompertz.pdf", width = 5, height = 5)
}
par(mfrow = c(1, 1), mar = c(5,5,4,3), family = "serif")
plot(sg, xg, type = "l", xlim = c(-1, 1), frame = F, cex.lab = 2, ylab = "y", xlab = "x", lwd = 2, col = "red", axes = F)
axis(side = 1, lwd = 1.5, cex.axis = 1.5)
axis(side = 2, lwd = 1.5, cex.axis = 1.5)
if (save){
  dev.off()
}

# STDP

tau_pre <- 20 
tau_post <- 20 
A_pre <- 0.01
A_post <- -A_pre * 1.05
delta_t <- seq(-50, 50, length.out = 100) 
W <- ifelse(delta_t > 0, A_pre * exp(-delta_t/tau_pre), A_post * exp(delta_t/tau_post))

if (save){
  pdf("stdp.pdf", width = 5, height = 5)
}
par(mfrow = c(1, 1), mar = c(5,5,4,2), family = "serif")
plot(delta_t, W, type = "l",  frame = F, cex.lab = 2, xlab = expression(Delta * t~'(ms)'), ylab = expression(Delta * w), lwd = 2, col = "red", axes = F)
segments(-50, 0, 50, 0, lty = 2, col = 'black', lwd = 2, lend = 1)
axis(side = 1, lwd = 1.5, cex.axis = 1.5)
axis(side = 2, lwd = 1.5, cex.axis = 1.5)
if (save){
  dev.off()
}






