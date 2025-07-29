library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(ggsignif)
library(pracma)
library(stringr)
library(Biostrings)

options(stringsAsFactors = F)

args <- commandArgs(trailingOnly=TRUE)
file_path <- args[1]
output_path <- args[2]

data <- read.table(file_path,
                  header = T)

calc_gc <- function(seq) {
    seq_length <- nchar(seq)
    noG <- gsub('G', '', seq)
    noGC <- gsub('C', '', noG)
    numGC <- seq_length - nchar(noGC)
    gc_content <- numGC / seq_length
    return(gc_content)
}

data$gc_content <- calc_gc(data$variant)

train <- filter(data, dataset == 'train')
test <- filter(data, dataset == 'test')

# simple linear model based on max -35 and -10 score
minus35and10_linear_fit <- lm(expn_med_fitted_scaled ~ minus35_max_score + minus10_max_score + 
                                  pwm_paired_max + gc_content, train)
summary(minus35and10_linear_fit)


test$predicted <- predict(minus35and10_linear_fit, test)
r.squared_minus35and10 <- summary(lm(expn_med_fitted_scaled ~ predicted, test))$r.squared


linear_accuracy <- rmserr(test$predicted, test$expn_med_fitted_scaled)
linear_accuracy

no_zeroes <- filter(train, expn_med_fitted_scaled != 0) %>% 
    mutate(logged = log(expn_med_fitted_scaled))
log_fit <- lm(logged ~ minus35_max_score + minus10_max_score +
                  pwm_paired_max + gc_content, no_zeroes)
summary(log_fit)

test$logged <- log(test$expn_med_fitted_scaled)
test$predicted_log <- predict(log_fit, test)
# no_zeroes$predicted_log <- predict(log_fit, no_zeroes)
r.squared_minus35and10_log <- summary(lm(log(expn_med_fitted_scaled) ~ predicted_log, test))$r.squared

log_accuracy <- rmserr(exp(test$predicted_log), exp(test$logged))
log_accuracy

write.table(
  data.frame(
    model = c("linear", "log"),
    rmse = c(linear_accuracy$rmse, log_accuracy$rmse),  # pick only RMSE
    r2 = c(r.squared_minus35and10, r.squared_minus35and10_log)
  ),
  file = output_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)