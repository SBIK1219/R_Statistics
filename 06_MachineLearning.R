library(dplyr)
library(ggplot2)
library(FNN)
library(rpart)
library(randomForest)
library(xgboost)

loan3000  <- read.csv('./data/loan3000.csv', stringsAsFactors=TRUE)
loan200  <- read.csv('./data/loan200.csv', stringsAsFactors=TRUE)
loan_data  <- read.csv('./data/loan_data.csv.gz', stringsAsFactors=TRUE)


loan200$outcome <- ordered(loan200$outcome, levels=c('paid off', 'default'))

loan3000$outcome <- ordered(loan3000$outcome, levels=c('paid off', 'default'))

loan_data <- select(loan_data, -X, -status)
loan_data$outcome <- ordered(loan_data$outcome, levels=c('paid off', 'default'))

Sys.setenv(KMP_DUPLICATE_LIB_OK = "TRUE")

newloan <- loan200[1, 2:3, drop=FALSE]
knn_pred <- knn(train=loan200[-1, 2:3], test=newloan, cl=loan200[-1, 1], k=20)
knn_pred == 'paid off'

nearest_points <- loan200[attr(knn_pred, 'nn.index') + 1, ] 
nearest_points
dist <- attr(knn_pred, 'nn.dist')

circleFun <- function(center=c(0, 0), r=1, npoints=100){
  tt <- seq(0, 2 * pi, length.out=npoints - 1)
  xx <- center[1] + r * cos(tt)
  yy <- center[2] + r * sin(tt)
  return(data.frame(x=c(xx, xx[1]), y=c(yy, yy[1])))
}

circle_df <- circleFun(center=unlist(newloan), r=max(dist), npoints=201)

loan200_df <- loan200 # bind_cols(loan200, circle_df)
levels(loan200_df$outcome)

levels(loan200_df$outcome) <- c(levels(loan200_df$outcome), "newloan")
loan200_df[1, 'outcome'] <- 'newloan'
head(loan200_df)
levels(nearest_points$outcome) <- levels(loan200_df$outcome)

graph <- ggplot(data=loan200_df, aes(x=payment_inc_ratio, y=dti, color=outcome)) + # , shape=outcome)) +
  geom_point(aes(shape=outcome), size=2, alpha=0.4) +
  geom_point(data=nearest_points, aes(shape=outcome), size=2) +
  geom_point(data=loan200_df[1,], aes(shape=outcome), size=2) +
  scale_shape_manual(values=c(15, 16, 4)) +
  scale_color_manual(values = c("paid off"="#1b9e77", "default"="#d95f02", "newloan"='black')) +
  geom_path(data=circle_df, aes(x=x, y=y), color='black') +
  coord_cartesian(xlim=c(3, 15), ylim=c(17, 29)) +
  theme_bw() 
graph

loan_df <- model.matrix(~ -1 + payment_inc_ratio + dti + revol_bal + 
                          revol_util, data=loan_data)
newloan <- loan_df[1, , drop=FALSE]
loan_df <- loan_df[-1,]
outcome <- loan_data[-1, 1]
knn_pred <- knn(train=loan_df, test=newloan, cl=outcome, k=5)
loan_df[attr(knn_pred, "nn.index"),]

loan_df <- model.matrix(~ -1 + payment_inc_ratio + dti + revol_bal + 
                          revol_util, data=loan_data)
loan_std <- scale(loan_df)
newloan_std <- loan_std[1, , drop=FALSE]
loan_std <- loan_std[-1,]
loan_df <- loan_df[-1,]
outcome <- loan_data[-1, 1]
knn_pred <- knn(train=loan_std, test=newloan_std, cl=outcome, k=5)
loan_df[attr(knn_pred, "nn.index"),]

borrow_df <- model.matrix(~ -1 + dti + revol_bal + revol_util + open_acc +
                            delinq_2yrs_zero + pub_rec_zero, data=loan_data)
borrow_knn <- knn(borrow_df, test=borrow_df, cl=loan_data[, 'outcome'], prob=TRUE, k=20)
prob <- attr(borrow_knn, "prob")
borrow_feature <- ifelse(borrow_knn == 'default', prob, 1 - prob)
summary(borrow_feature)

loan_data$borrower_score <- borrow_feature

plot(borrow_feature)

loan_tree <- rpart(outcome ~ borrower_score + payment_inc_ratio,
                   data=loan3000, control=rpart.control(cp=0.005))

plot(loan_tree, uniform=TRUE, margin=0.05)
text(loan_tree, cex=0.75)

loan_tree

r_tree <- tibble(x1 = c(0.575, 0.375, 0.375, 0.375, 0.475),
                 x2 = c(0.575, 0.375, 0.575, 0.575, 0.475),
                 y1 = c(0,         0, 10.42, 4.426, 4.426),
                 y2 = c(25,       25, 10.42, 4.426, 10.42),
                 rule_number = factor(c(1, 2, 3, 4, 5)))
r_tree <- as.data.frame(r_tree)

rules <- tibble(x=c(0.575, 0.375, 0.4, 0.4, 0.475),
                y=c(24, 24, 10.42, 4.426, 9.42),
                rule_number = factor(c(1, 2, 3, 4, 5))) # , 3, 4, 5)))

labs <- tibble(x=c(.575 + (1-.575)/2, 
                   .375/2, 
                   (.375 + .575)/2,
                   (.375 + .575)/2, 
                   (.475 + .575)/2, 
                   (.375 + .475)/2
),
y=c(12.5, 
    12.5,
    10.42 + (25-10.42)/2,
    4.426/2, 
    4.426 + (10.42-4.426)/2,
    4.426 + (10.42-4.426)/2
),
decision = factor(c('paid off', 'default', 'default', 'paid off', 'paid off', 'default')))

graph <- ggplot(data=loan3000, aes(x=borrower_score, y=payment_inc_ratio)) +
  geom_point( aes(color=outcome, shape=outcome), alpha=.5) +
  scale_color_manual(values=c('blue', 'red')) +
  scale_shape_manual(values = c(1, 46)) +
  geom_segment(data=r_tree, aes(x=x1, y=y1, xend=x2, yend=y2, linetype=rule_number), size=1.5, alpha=.7) +
  guides(color = guide_legend(override.aes = list(size=1.5)),
         linetype = guide_legend(keywidth=3, override.aes = list(size=1))) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0)) + 
  coord_cartesian(ylim=c(0, 25)) +
  geom_label(data=labs, aes(x=x, y=y, label=decision)) +
  #theme(legend.position='bottom') +
  theme_bw()
graph

graph <- ggplot(data=loan3000, aes(x=borrower_score, y=payment_inc_ratio)) +
  geom_point( aes(color=outcome, shape=outcome, size=outcome), alpha=.8) +
  scale_color_manual(values = c("paid off"="#7fbc41", "default"="#d95f02")) +
  scale_shape_manual(values = c('paid off'=0, 'default'=1)) +
  scale_size_manual(values = c('paid off'=0.5, 'default'=2)) +
  geom_segment(data=r_tree, aes(x=x1, y=y1, xend=x2, yend=y2), size=1.5) + #, linetype=rule_number), size=1.5, alpha=.7) +
  guides(color = guide_legend(override.aes = list(size=1.5)),
         linetype = guide_legend(keywidth=3, override.aes = list(size=1))) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0)) + 
  coord_cartesian(ylim=c(0, 25)) +
  geom_label(data=labs, aes(x=x, y=y, label=decision)) +
  geom_label(data=rules, aes(x=x, y=y, label=rule_number), 
             size=2.5,
             fill='#eeeeee', label.r=unit(0, "lines"), label.padding=unit(0.2, "lines")) +
  guides(color = guide_legend(override.aes = list(size=2))) +
  theme_bw()
graph

info <- function(x){
  info <- ifelse(x==0, 0, -x * log2(x) - (1-x) * log2(1-x))
  return(info)
}
x <- 0:50/100
plot(x, info(x) + info(1-x))

gini <- function(x){
  return(x * (1-x))
}
plot(x, gini(x))

impure <- data.frame(p = rep(x, 3),
                     impurity = c(2*x,
                                  gini(x)/gini(.5)*info(.5),
                                  info(x)),
                     type = rep(c('Accuracy', 'Gini', 'Entropy'), rep(51,3)))

graph <- ggplot(data=impure, aes(x=p, y=impurity, linetype=type, color=type)) + 
  geom_line(size=1.5) +
  guides( linetype = guide_legend( keywidth=3, override.aes = list(size=1))) +
  scale_x_continuous(expand=c(0,0.01)) + 
  scale_y_continuous(expand=c(0,0.01)) + 
  theme_bw() +
  theme(legend.title=element_blank()) 
graph