library(ggplot2)
library(ggalt)
setwd("/Users/alison/Documents/DPhil/hybridmodels/data/csvs")
df <- read.csv("PCA_by_storm.csv")

continent = df[df$continent=="africa",]

gg <- ggplot(df,aes(PCA1,PCA2))
gg + geom_encircle(aes(group=continent, fill=continent), alpha=0.4) + 
  xlim(-8,8) + 
  ylim(-8,8) + 
  theme_bw()
  
gg + geom_encircle(aes(group=storm, fill=storm), alpha=0.4) + 
  xlim(-8,8) + 
  ylim(-8,8) + 
  theme_bw()
