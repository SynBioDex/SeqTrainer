# standardize expression across all datasets. Formatted expression files should be
# create first, e.g. ending with _expression_formatted.txt . Create a "standard curve" based
# on the wild-type sequences in both the TSS and scramble library. Use robust
# linear fit to mitigate effects of outliers on the standard linear fit

args <- commandArgs(trailingOnly = TRUE)
tss_path <- args[1]
scramble_path <- args[2]
peak_path <- args[3]
flp3_path <- args[4]
rlp6_path <- args[5]

options(scipen = 999)
options(stringsAsFactors = F)

library(dplyr)
library(tidyr)
library(ggplot2)
library(cowplot)

tss <- read.table(tss_path, header = T)

scramble <- read.table(scramble_path, header = T)

peak_tile <- read.table(peak_path, header = T)

flp3 <- read.table(flp3_path, header = T)

rlp6 <- read.table(rlp6_path, header = T)

# combine datasets for modeling
combined <- bind_rows(tss, 
                      select(scramble, variant, expn_med_fitted_scaled,
                             start=var_left, end=var_right, name, active),
                      peak_tile) %>% 
    select(variant, expn_med_fitted_scaled, start, end, name, active)

write.table(combined, './tss_scramble_peak_expression_model_format.txt',
            row.names = F, col.names = F, quote = F, sep = '\t')
write.table(select(combined, variant, expn_med_fitted_scaled), 
            './tss_scramble_peak_expression_model_format_values_only.txt',
            row.names = F, col.names = F, quote = F, sep = '\t')
# just TSS for gkmSVM
tss %>% 
    select(variant, expn_med_fitted_scaled, start, end, name, active) %>% 
    write.table('./tss_expression_model_format.txt',
                row.names = F, col.names = F, quote = F, sep = '\t')


# graph positive controls in scramble and TSS
pos_controls <- inner_join(filter(tss, category == 'pos_control') %>% 
                               select(name, expn_med1 = expn_med_fitted_scaled),
                           filter(scramble, category == 'pos_control') %>% 
                               select(name, expn_med2 = expn_med_fitted_scaled),
                           by = 'name')

controls <- inner_join(filter(tss, grepl("control", category)) %>% 
                           select(name, category, expn_med1 = expn_med_fitted),
                           filter(scramble, grepl("control", category)) %>% 
                               select(name, category, expn_med2 = expn_med_fitted),
                           by = 'name')

ggplot(controls, aes(expn_med1, expn_med2)) + 
    geom_point(aes(color = category.x)) +
    scale_x_log10() + scale_y_log10()

# graph unscrambled
wildtype <- inner_join(filter(tss, category == 'tss') %>% 
                           select(name, tss_name, expn_med1 = expn_med),
                       filter(scramble, category == 'unscrambled') %>% 
                           select(name, tss_name, expn_med2 = expn_med),
                       by = 'tss_name')


r2 <- summary(lm(expn_med1 ~ expn_med2, wildtype))$r.squared
ggplot(wildtype, aes(expn_med1, expn_med2)) + 
    geom_point() + 
    annotation_logticks(sides='bl') + scale_x_log10() + scale_y_log10() +
    geom_smooth(method = 'lm') +
    labs(x = 'TSS expression', y = 'scramble expression',
         title = 'Wild-type sequences') +
    annotate('text', x=50, y=0.1, parse=T, label=paste('R^2==', signif(r2, 3)))

