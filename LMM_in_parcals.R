library(ggplot2)
library(lme4)
library(reshape2)
library(data.table)
library(ggpubr)
library(emmeans)
library(lmerTest)
library(stringi)
library(stringr)
library(dplyr)
library(purrr)
#library(tidyverse)
library(scales)
#library(optimx)
options(scipen = 999)
path<-'/media/kristina/storage/probability/sources/alpha_8_12/df_1200_1600'
list_of_trained <- list.files(path = path,
                              recursive = TRUE,
                              pattern = "\\.csv$",
                              full.names = TRUE)

tbl_with_sources <- readr::read_csv(list_of_trained, id = "file_name")
tbl_with_sources<- as.data.table(tbl_with_sources)


emm_options(lmerTest.limit = 90000)
emm_options(pbkrtest.limit = 90000)

labels<- unique(tbl_with_sources$label)


p_vals <- data.table()
for (l in 1:length(labels)){
  print(labels[l])
  temp<- subset(lp_hp,label == labels[l])
  m <-  lmer(beta_power ~ group*trial_type*feedback_cur+(1|subject), data = temp)
  summary(m)
  an <- anova(m)
  print(an)
  an <- data.table(an,keep.rownames = TRUE)
  an_cols <- c('rn','Pr(>F)') 
  an <- an[, ..an_cols]
  an$`Pr(>F)` <- format(an$`Pr(>F)`, digits = 3)
  an$interval <- "beta_power"
  an$interval <- gsub('beta power','',an$interval)
  an <- dcast(an,formula = interval~rn,value.var = 'Pr(>F)')
  an$label <- unique(temp$label)
  p_vals <- rbind(p_vals,an)
}

setwd("/media/kristina/storage/probability/sources/df_lmems")
write.csv(p_vals, "lmem_alpha_1200_1600.csv")
