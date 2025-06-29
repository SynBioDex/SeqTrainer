# standardize expression across all datasets. Formatted expression files should be
# create first, e.g. ending with _expression_formatted.txt . Create a "standard curve" based
# on the wild-type sequences in both the TSS and scramble library. Use robust
# linear fit to mitigate effects of outliers on the standard linear fit


options(scipen = 999)
options(stringsAsFactors = F)

library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)

args <- commandArgs(trailingOnly=TRUE)
tss_path <- args[1]
scramble_path <- args[2]
peak_path <- args[3]
flp3_path <- args[4]
rlp6_path <- args[5]

tss <- read.table(tss_path, 
                  header = T)

scramble <- read.table(scramble_path, 
                       header = T)
peak_tile <- read.table(peak_path,
                        header = T)

flp3 <- read.table(flp3_path,
                   header = T)

rlp6 <- read.table(rlp6_path,
                   header = T)

# format scramble negative control category correctly
scramble$category[grep("_scrambled", scramble$name)] <- "scramble"

# wt_scramble <- scramble %>% 
#     filter(category == 'unscrambled') %>% 
#     left_join(select(tss, tss_name, RNA_exp_sum_ave, expn_med), by = 'tss_name',
#               suffix = c('_scramble', '_tss'))
# 
# wt_robust_fit <- MASS::rlm(expn_med_tss ~ expn_med_scramble, wt_scramble)
# 
# # fit expression using robust fit, rename column so it works with fit
# scramble$expn_med_fitted <- predict(wt_robust_fit, 
#                                     select(scramble, expn_med_scramble = expn_med))
# peak_tile$expn_med_fitted <- predict(wt_robust_fit,
#                                      select(peak_tile, expn_med_scramble = expn_med))
# flp3$expn_med_fitted <- predict(wt_robust_fit,
#                                      select(flp3, expn_med_scramble = expn_med))
# rlp6$expn_med_fitted <- predict(wt_robust_fit,
#                                      select(rlp6, expn_med_scramble = expn_med))

standardize <- function(df1, df2) {
    # subset positive controls
    pos_controls <- inner_join(filter(df1, category == 'pos_control') %>% 
                                   select(name, expn_med1 = expn_med),
                               filter(df2, category == 'pos_control') %>% 
                                   select(name, expn_med2 = expn_med),
                               by = 'name')
    control_fit <- MASS::rlm(expn_med1 ~ expn_med2, pos_controls)
    predicted <- predict(control_fit, select(df2, expn_med2 = expn_med))
    return(predicted)
}

scramble$expn_med_fitted <- standardize(tss, scramble)
peak_tile$expn_med_fitted <- standardize(tss, peak_tile)
flp3$expn_med_fitted <- standardize(tss, flp3)
rlp6$expn_med_fitted <- standardize(tss, rlp6)

# we do not need to scale TSS because it's the standard, but create a "fitted"
# column for consistency
tss$expn_med_fitted <- tss$expn_med

set_scale <- function(df) {
    neg <- filter(df, category == 'neg_control')
    neg_median <- median(neg$expn_med_fitted)
    return(df$expn_med_fitted - neg_median)
}


set_std_threshold <- function(df) {
    neg <- filter(df, category == 'neg_control')
    neg_sd <- sd(neg$expn_med_fitted_scaled)
    neg_median <- median(neg$expn_med_fitted_scaled)
    threshold <- neg_median + (2 * neg_sd)
    return(threshold)
}

set_std_threshold_peak <- function(df) {
    neg <- filter(df, category == 'neg_control')
    neg_mad <- mad(neg$expn_med_fitted_scaled)
    neg_median <- median(neg$expn_med_fitted_scaled)
    threshold <- neg_median + (3 * neg_mad)
    return(threshold)
}

tss$expn_med_fitted_scaled <- set_scale(tss)
scramble$expn_med_fitted_scaled <- set_scale(scramble)
peak_tile$expn_med_fitted_scaled <- set_scale(peak_tile)
flp3$expn_med_fitted_scaled <- set_scale(flp3)
rlp6$expn_med_fitted_scaled <- set_scale(rlp6)

# now that expression is standardized, calculate change in expression for scramble
# calculate change in expression: unscramble - scramble
scramble <- scramble %>% 
    group_by(tss_name) %>% 
    mutate(unscrambled_exp = ifelse(any(category == 'unscrambled'),
                                    RNA_exp_sum_ave[category == 'unscrambled'],
                                    NA),
           relative_exp = RNA_exp_sum_ave / unscrambled_exp)

scramble_threshold <- set_std_threshold(scramble)
scramble <- scramble %>% 
    mutate(active = ifelse(expn_med_fitted_scaled >= scramble_threshold, 'active', 'inactive'))
# create active and inactive columns for TSS and alternate landing pads
tss_threshold <- set_std_threshold(tss)
tss <- tss %>% 
    mutate(active = ifelse(expn_med_fitted_scaled >= tss_threshold, 'active', 'inactive'))
peak_threshold <- set_std_threshold(peak_tile)
peak_tile <- peak_tile %>% 
    mutate(active = ifelse(expn_med_fitted_scaled >= peak_threshold, 'active', 'inactive'))
flp3_threshold <- set_std_threshold(flp3)
flp3 <- flp3 %>% 
    mutate(active = ifelse(expn_med_fitted_scaled >= flp3_threshold, 'active', 'inactive'))
rlp6_threshold <- set_std_threshold(rlp6)
rlp6 <- rlp6 %>% 
    mutate(active = ifelse(expn_med_fitted_scaled >= rlp6_threshold, 'active', 'inactive'))

write.table(tss, file = '../data/processed_data/linear_model/rLP5_Endo2_lb_expression_formatted_std.txt',
            row.names = F, quote = F, sep = '\t')
write.table(scramble, file = '../data/processed_data/linear_model/endo_scramble_expression_formatted_std.txt',
            row.names = F, quote = F, sep = '\t')
write.table(peak_tile, file = '../data/processed_data/linear_model/peak_tile_expression_formatted_std.txt',
            row.names = F, quote = F, sep = '\t')
write.table(flp3, file = '../data/processed_data/linear_model/fLP3_Endo2_lb_expression_formatted_std.txt',
            row.names = F, quote = F, sep = '\t')
write.table(rlp6, file = '../data/processed_data/linear_model/rLP6_Endo2_lb_expression_formatted_std.txt',
            row.names = F, quote = F, sep = '\t')

# combine datasets for modeling
combined <- bind_rows(tss, 
                      select(scramble, variant, expn_med_fitted_scaled,
                             start=var_left, end=var_right, name, active),
                      peak_tile) %>% 
    select(variant, expn_med_fitted_scaled, start, end, name, active)

write.table(combined, '../data/processed_data/linear_model/tss_scramble_peak_expression_model_format.txt',
            row.names = F, col.names = F, quote = F, sep = '\t')
write.table(select(combined, variant, expn_med_fitted_scaled), 
            '../data/processed_data/linear_model/tss_scramble_peak_expression_model_format_values_only.txt',
            row.names = F, col.names = F, quote = F, sep = '\t')
# just TSS for gkmSVM
tss %>% 
    select(variant, expn_med_fitted_scaled, start, end, name, active) %>% 
    write.table('../data/processed_data/linear_model/tss_expression_model_format.txt',
                row.names = F, col.names = F, quote = F, sep = '\t')