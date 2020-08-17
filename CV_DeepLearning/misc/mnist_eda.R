mnist <- read.csv('mnist.csv')

library('tidyverse')
glimpse(mnist)

left <- mnist$Left
right <- mnist$Right

summary(left/right)
summary(log(right/left))

mnist$Class <- as.factor(mnist$Class)


mnist$x.1 = mnist$x + as.integer(sqrt(mnist$Left)-50)
mnist$y.1 = mnist$x + as.integer(sqrt(mnist$Right)-60)

ggplot(mnist, aes( x = Class, y =logL.R, group = Class, fill = Class)) + geom_boxplot()
ggplot(mnist, aes( x = Class, y =log(Left), group = Class, fill = Class)) + geom_boxplot()
ggplot(mnist, aes( x = Class, y =log(Right), group = Class, fill = Class)) + geom_boxplot()

ggplot(mnist, aes( x = Class, y =x.1, group = Class, fill = Class)) + geom_boxplot()
ggplot(mnist, aes( x = Class, y =y.1, group = Class, fill = Class)) + geom_boxplot()
