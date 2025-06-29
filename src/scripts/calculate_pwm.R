
pkgs <- c('dplyr', 'ggplot2', 'tidyr', 'cowplot', 'stringr')
library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)
library(stringr)
library(Biostrings)

options(stringsAsFactors = F)

args <- commandArgs(trailingOnly=TRUE)
train_name <- args[1]
test_name <- args[2]
output_name <- args[3]

train <- read.table(train_name, col.names = c('variant', 'expn_med_fitted_scaled')) %>% 
    mutate(dataset = 'train')

test <- read.table(test_name, col.names = c('variant', 'expn_med_fitted_scaled')) %>% 
    mutate(dataset = 'test')

data <- bind_rows(train, test)
#This code uses the pwms from bTSSFinder (which come from literature) to find the best -10 site
A = c(0.0097,	1.0000,	0.3363,	0.5335,	0.4963,	0.0781)
C = c(0.0618,	0.0000,	0.1190,	0.1094,	0.2299,	0.0268)
G = c(0.1042,	0.0000,	0.0856,	0.1317,	0.1399,	0.0000)
T = c(0.8244,	0.0000,	0.4591,	0.2254,	0.1339,	0.8951)
pwm_s70 <- data.frame(A,C,G,T)
pwm_s70 <- t(pwm_s70)

data$minus10_matches <- sapply(as.character(data$variant),
                               function(x) matchPWM(pwm_s70, str_sub(x,-150,-1)))

data$minus10_scores <- sapply(data$minus10_matches,
                              function(x) PWMscoreStartingAt(pwm_s70, 
                                                             subject(x),
                                                             start(x)))
# set empty lists to NA
data$minus10_scores[sapply(data$minus10_scores, length) == 0] <- NA
data$minus10_max_score <- sapply(data$minus10_scores, max)
data$minus10_max_entry <- sapply(data$minus10_scores, which.max)
data$minus10_scores <- sapply(data$minus10_matches,
                              function(x) PWMscoreStartingAt(pwm_s70, 
                                                             subject(x),
                                                             start(x)))

for (i in seq(1,length(data$minus10_matches))) {
    if(is.na(data$minus10_max_score[i])){
        data$minus10_max_score[i] <- 0
        data$minus10_start[i] <- -1
        data$minus10[i] <- "NNNNNN"
    }
    else{
        start_pos <- data$minus10_matches[i][[1]]@ranges@start[data$minus10_max_entry[[i]]]
        data$minus10_start[i] <- start_pos
        data$minus10[i] <- str_sub(data$variant[i], start_pos, start_pos+5)
    }
}

data <- select(data, -minus10_max_entry)

#This code uses the pwms from bTSSFinder (which come from literature) to find the best -35 site
A = c(0.0000,	0.0784,	0.0362,	0.4894,	0.3605,	0.4208)
C = c(0.1109,	0.0656,	0.0747,	0.2851,	0.3605,	0.0769)
G = c(0.1267,	0.0181,	0.6192,	0.1041,	0.0000,	0.2225)
T = c(0.7624,	0.8379,	0.2700,	0.1214,	0.2790,	0.2798)
pwm_s70_35 <- data.frame(A,C,G,T)
pwm_s70_35 <- t(pwm_s70_35)

data$minus35_matches <- sapply(as.character(data$variant),
                               function(x) matchPWM(pwm_s70_35, str_sub(x,-150,-1))) #Normally (x,-150,-1)
data$minus35_scores <- sapply(data$minus35_matches,
                              function(x) PWMscoreStartingAt(pwm_s70_35, 
                                                             subject(x),
                                                             start(x)))
data$minus35_scores[sapply(data$minus35_scores, length) == 0] <- NA
data$minus35_max_score <- sapply(data$minus35_scores, max)
data$minus35_max_entry <- sapply(data$minus35_scores,
                                 which.max)
data$minus35_scores <- sapply(data$minus35_matches,
                              function(x) PWMscoreStartingAt(pwm_s70_35, 
                                                             subject(x),
                                                             start(x)))

for (i in seq(1,length(data$minus35_matches))) {
    if(is.na(data$minus35_max_score[i])){
        data$minus35_max_score[i] <- 0
        data$minus35_start[i] <- -1
        data$minus35[i] <- "NNNNNN"
    }
    else{
        start_pos <- data$minus35_matches[i][[1]]@ranges@start[data$minus35_max_entry[[i]]]
        data$minus35_start[i] <- start_pos
        data$minus35[i] <- str_sub(data$variant[i],start_pos,start_pos+5)
    }
}

data <- select(data, -minus35_max_entry)

# paired scoring of PWMs with variable spacing
paired_pwm_scoring <- function(seq, minus10, minus35, spacer_length) {
    # grab scores at every position
    scores_minus10 <- matchPWM(minus10, seq, min.score = 0, with.score = TRUE)
    scores_minus35 <- matchPWM(minus35, seq, min.score = 0, with.score = TRUE)
    score_length <- length(mcols(scores_minus10)$score)
    # subtract six to account for length of minus 35 and spacer
    paired_scores <- mcols(scores_minus10)$score[1:(score_length - spacer_length - 6)] +
        mcols(scores_minus35)$score[(spacer_length + 6 + 1) : score_length]
    return(max(paired_scores))
}

data$pwm_paired_max_16bp <- sapply(as.character(data$variant),
                                   function(x) paired_pwm_scoring(x, 
                                                                  pwm_s70, 
                                                                  pwm_s70_35,
                                                                  16))
data$pwm_paired_max_17bp <- sapply(as.character(data$variant),
                                   function(x) paired_pwm_scoring(x, 
                                                                  pwm_s70, 
                                                                  pwm_s70_35,
                                                                  17))
data$pwm_paired_max_18bp <- sapply(as.character(data$variant),
                                   function(x) paired_pwm_scoring(x, 
                                                                  pwm_s70, 
                                                                  pwm_s70_35,
                                                                  18))

# create column for best score among three spacer lengths and which length
data <- data %>% 
    rowwise() %>% 
    mutate(pwm_paired_max = max(pwm_paired_max_16bp, pwm_paired_max_17bp, pwm_paired_max_18bp))

write.table(select(data, -minus35_scores, -minus35_matches, -minus10_matches, -minus10_scores),
            output_name, row.names = F, quote = F, sep = '\t')
