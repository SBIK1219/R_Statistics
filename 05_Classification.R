library(klaR)
library(MASS)
library(dplyr)
library(ggplot2)
library(FNN)
library(mgcv)
library(rpart)

loan3000  <- read.csv('./data/loan3000.csv', stringsAsFactors=TRUE)
loan_data <-  read.csv('./data/loan_data.csv.gz', stringsAsFactors=TRUE)
full_train_set <-  read.csv('./data/full_train_set.csv.gz', stringsAsFactors=TRUE)

loan3000$outcome <- ordered(loan3000$outcome, levels=c('paid off', 'default'))
loan_data$outcome <- ordered(loan_data$outcome, levels=c('paid off', 'default'))
full_train_set$outcome <- ordered(full_train_set$outcome, levels=c('paid off', 'default'))

naive_model <- NaiveBayes(outcome ~ purpose_ + home_ + emp_len_, 
                          data = na.omit(loan_data))
naive_model$table

new_loan <- loan_data[147, c('purpose_', 'home_', 'emp_len_')]
row.names(new_loan) <- NULL
new_loan

predict(naive_model, new_loan)

print(predict(naive_model, new_loan))

less_naive <- NaiveBayes(outcome ~ borrower_score + payment_inc_ratio + 
                           purpose_ + home_ + emp_len_, data = loan_data)
less_naive$table[1:2]

stats <- less_naive$table[[1]]
graph <- ggplot(data.frame(borrower_score=c(0,1)), aes(borrower_score)) +
  stat_function(fun = dnorm, color='blue', linetype=1, 
                args=list(mean=stats[1, 1], sd=stats[1, 2])) +
  stat_function(fun = dnorm, color='red', linetype=2, 
                args=list(mean=stats[2, 1], sd=stats[2, 2])) +
  labs(y='probability')
graph

loan_lda <- lda(outcome ~ borrower_score + payment_inc_ratio,
                data=loan3000)
loan_lda$scaling

pred <- predict(loan_lda)
print(head(pred$posterior))

pred <- predict(loan_lda)
lda_df <- cbind(loan3000, prob_default=pred$posterior[,'default'])

x <- seq(from=.33, to=.73, length=100)
y <- seq(from=0, to=20, length=100)
newdata <- data.frame(borrower_score=x, payment_inc_ratio=y)
pred <- predict(loan_lda, newdata=newdata)
lda_df0 <- cbind(newdata, outcome=pred$class)

graph <- ggplot(data=lda_df, aes(x=borrower_score, y=payment_inc_ratio, color=prob_default)) +
 geom_point(alpha=.6) +
 scale_color_gradient2(low='white', high='blue') +
 scale_x_continuous(expand=c(0,0)) +
 scale_y_continuous(expand=c(0,0), lim=c(0, 20)) +
 geom_line(data=lda_df0, col='darkgreen', size=2, alpha=.8) +
 theme_bw()
graph

graph
dev.off()

pred <- predict(loan_lda)
lda_df <- cbind(loan3000, prob_default=pred$posterior[,'default'])

center <- 0.5 * (loan_lda$mean[1, ] + loan_lda$mean[2, ])
slope <- -loan_lda$scaling[1] / loan_lda$scaling[2]
intercept = center[2] - center[1] * slope

graph <- ggplot(data=lda_df, aes(x=borrower_score, y=payment_inc_ratio, color=prob_default)) +
  geom_point(alpha=.6) +
  scale_color_gradientn(colors=c('#ca0020', '#f7f7f7', '#0571b0')) +
  scale_x_continuous(expand=c(0,0)) + 
  scale_y_continuous(expand=c(0,0), lim=c(0, 20)) + 
  geom_abline(slope=slope, intercept=intercept, color='darkgreen') +
  theme_bw()

graph

logistic_model <- glm(outcome ~ payment_inc_ratio + purpose_ + 
                        home_ + emp_len_ + borrower_score,
                      data=loan_data, family='binomial')
logistic_model
summary(logistic_model)

p <- seq(from=0.01, to=.99, by=.01)
df <- data.frame(p = p,
                 logit = log(p/(1-p)),
                 odds = p/(1-p))

graph <- ggplot(data=df, aes(x=p, y=logit)) +
  geom_line() +
  labs(x = 'p', y='logit(p)') +
  theme_bw()
graph

pred <- predict(logistic_model)
summary(pred)

prob <- 1/(1 + exp(-pred))
summary(prob)

graph <- ggplot(data=df, aes(x=logit, y=odds)) +
  geom_line() +
  labs(x='log(odds ratio)', y='odds ratio') +
  coord_cartesian(xlim=c(0, 5), ylim=c(1, 100)) +
  theme_bw()
graph

logistic_gam <- gam(outcome ~ s(payment_inc_ratio) + purpose_ + 
                      home_ + emp_len_ + s(borrower_score),
                    data=loan_data, family='binomial')
logistic_gam

terms <- predict(logistic_gam, type='terms')
partial_resid <- resid(logistic_gam) + terms
df <- data.frame(payment_inc_ratio = loan_data[, 'payment_inc_ratio'],
                 terms = terms[, 's(payment_inc_ratio)'],
                 partial_resid = partial_resid[, 's(payment_inc_ratio)'])

graph <- ggplot(df, aes(x=payment_inc_ratio, y=partial_resid, solid = FALSE)) +
  geom_point(shape=46, alpha=0.4) +
  geom_line(aes(x=payment_inc_ratio, y=terms), 
            color='red', alpha=0.5, size=1.5) +
  labs(y='Partial Residual') +
  coord_cartesian(xlim=c(0, 25)) +
  theme_bw()
graph

df <- data.frame(payment_inc_ratio = loan_data[, 'payment_inc_ratio'],
                 terms = 1/(1 + exp(-terms[, 's(payment_inc_ratio)'])),
                 partial_resid = 1/(1 + exp(-partial_resid[, 's(payment_inc_ratio)'])))

graph <- ggplot(df, aes(x=payment_inc_ratio, y=partial_resid, solid = FALSE)) +
  geom_point(shape=46, alpha=0.4) +
  geom_line(aes(x=payment_inc_ratio, y=terms), 
            color='red', alpha=0.5, size=1.5) +
  labs(y='Partial Residual') +
  coord_cartesian(xlim=c(0, 25)) +
  theme_bw()
graph
