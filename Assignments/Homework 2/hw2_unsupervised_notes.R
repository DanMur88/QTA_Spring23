# Set working directory and read in data
setwd(getwd())
data_unsup <- read.csv("./data/breitbart_2016_sample.csv", 
                 stringsAsFactors=FALSE,
                 encoding = "utf-8")

# Load required libraries
library(stm)
library(lubridate)
library(tidyverse)
library(tidyr)

# Create corpus object
corp_unsup <- corpus(data_unsup,
                 text_field="content")

# Add original text as docvar
docvars(corp_unsup)$text <- texts(corp_unsup)

# Inspect corpus attributes
summary(corp_unsup,5)

# Tokenise text and remove punctuation/symbols/URLs, etc.
toks_unsup <- quanteda::tokens(corp_unsup, 
                         include_docvars = TRUE,
                         remove_numbers = TRUE,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_hyphens = TRUE,
                         remove_separators = TRUE,
                         remove_url = TRUE)

# Convert tokens to lowercase
toks_unsup <- tokens_tolower(toks_unsup)

# Create list of stopwords and remove these from tokens, retaining whitespaces in place of removed stopwords
stop_list <- stopwords("english")
toks_unsup <- tokens_remove(toks_unsup, stop_list, padding=TRUE)

# Identify bigram collocations
bigrams_unsup <- textstat_collocations(toks_unsup, 
                                 method = "lambda",
                                 size = 2,
                                 min_count = 10,
                                 smoothing = 0.5)

# Create list of bigrams of interest (z-score > 30)
bigrams_unsup <- bigrams_unsup[bigrams_unsup$z > 30,]

# Merge collocations with tokens and remove whitespaces
toks_col_unsup <- tokens_compound(toks_unsup, pattern = bigrams_unsup)
toks_col_unsup <- tokens_remove(toks_col_unsup, "")

# Generate dfm and trim
dfm_unsup <- dfm(toks_col_unsup)
dfm_unsup <- dfm_trim(dfm_unsup, min_docfreq = 20)

# Check top features of dfm
topfeatures(dfm_unsup, 50)

# Parse dates within dfm docvars
dfm_unsup@docvars$date <- dmy(dfm_unsup@docvars$date)
dfm_unsup@docvars$date_month <- floor_date(dfm_unsup@docvars$date, "month")

# Save document feature matrix
saveRDS(dfm_unsup, "data/dfm_unsup")

# Convert dfm to stm
stmdfm <- convert(dfm_unsup, to = "stm")

# Run stm algorithm
modelFit <- stm(documents = stmdfm$documents,
                vocab = stmdfm$vocab,
                K = 35,
                prevalence = ~ s(as.numeric(date_month)),
                data = stmdfm$meta,
                max.em.its = 500,
                init.type = "Spectral",
                seed = 1234,
                verbose = TRUE)

# Save model
saveRDS(modelFit, "data/STM_model")

# Read in model
stm_model <- readRDS("data/STM_model")

# Check top terms per label
labelTopics(stm_model)

# Plot topic prevalence and top terms per topic by FREX metric
plot.STM(modelFit, 
         type = "summary", 
         labeltype = "frex",
         text.cex = 0.7,
         main = "Topic prevalence and top terms (FREX)")

# Plot topic prevalence and top terms per topic by Probability metric
plot.STM(modelFit, 
         type = "summary", 
         labeltype = "prob",
         text.cex = 0.7,
         main = "Topic prevalence and top terms (Probability)")

findThoughts(stm_model,
             texts = dfm_unsup@docvars$title,
             topics = 1,
             n = 10)

## 4. Topic validation: predictive validity using time series data
# Convert metadata to correct format
stmdfm$meta$num_month <- month(stmdfm$meta$date)

# Aggregate topic probability by month
agg_theta <- setNames(aggregate(stm_model$theta,
                                by = list(month = stmdfm$meta$num_month),
                                FUN = mean),
                      c("month", paste("Topic",1:35)))
agg_theta <- pivot_longer(agg_theta, cols = starts_with("T")) 

# Aggregate topic probability by month (Topic 1-7)
agg_theta1 <- agg_theta[agg_theta$name %in% c("Topic 1", "Topic 2", "Topic 3",
                                               "Topic 4", "Topic 5", "Topic 6",
                                               "Topic 7"),]

# Aggregate topic probability by month (Topic 8-14)
agg_theta2 <- agg_theta[agg_theta$name %in% c("Topic 8", "Topic 9", "Topic 10",
                                              "Topic 11", "Topic 12", "Topic 13",
                                              "Topic 14"),]

# Aggregate topic probability by month (Topic 15-21)
agg_theta3 <- agg_theta[agg_theta$name %in% c("Topic 15", "Topic 16", "Topic 17",
                                              "Topic 18", "Topic 19", "Topic 20",
                                              "Topic 21"),]

# Aggregate topic probability by month (Topic 22-28)
agg_theta4 <- agg_theta[agg_theta$name %in% c("Topic 22", "Topic 23", "Topic 24",
                                              "Topic 25", "Topic 26", "Topic 27",
                                              "Topic 28"),]

# Aggregate topic probability by month (Topic 29-35)
agg_theta5 <- agg_theta[agg_theta$name %in% c("Topic 29", "Topic 30", "Topic 31",
                                              "Topic 32", "Topic 33", "Topic 34",
                                              "Topic 35"),]


# Plot aggregated theta over time (Topic 1-7)
ggplot(data = agg_theta1,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

# Plot aggregated theta over time (Topic 8-14)
ggplot(data = agg_theta2,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

# Plot aggregated theta over time (Topic 15-21)
ggplot(data = agg_theta3,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

# Plot aggregated theta over time (Topic 22-28)
ggplot(data = agg_theta4,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

# Plot aggregated theta over time (Topic 29-35)
ggplot(data = agg_theta5,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

## Check sample documents with a high probability of containing each topic ##

# Check headline of articles with high probability for Topic 1
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 1,
             n = 10)

# Check headline of articles with high probability for Topic 2
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 2,
             n = 10)

# Check headline of articles with high probability for Topic 3
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 3,
             n = 10)

# Check headline of articles with high probability for Topic 4
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 4,
             n = 10)

# Check headline of articles with high probability for Topic 5
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 5,
             n = 10)

# Check headline of articles with high probability for Topic 6
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 6,
             n = 10)

# Check headline of articles with high probability for Topic 7
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 7,
             n = 10)

# Check headline of articles with high probability for Topic 8
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 8,
             n = 10)

# Check headline of articles with high probability for Topic 9
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 9,
             n = 10)

# Check headline of articles with high probability for Topic 10
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 10,
             n = 10)

# Check headline of articles with high probability for Topic 11
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 11,
             n = 10)

# Check headline of articles with high probability for Topic 12
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 12,
             n = 10)

# Check headline of articles with high probability for Topic 13
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 13,
             n = 10)

# Check headline of articles with high probability for Topic 14
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 14,
             n = 10)

# Check headline of articles with high probability for Topic 15
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 15,
             n = 10)

# Check headline of articles with high probability for Topic 16
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 16,
             n = 10)

# Check headline of articles with high probability for Topic 17
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 17,
             n = 10)

# Check headline of articles with high probability for Topic 18
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 18,
             n = 10)

# Check headline of articles with high probability for Topic 19
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 19,
             n = 10)

# Check headline of articles with high probability for Topic 20
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 20,
             n = 10)

# Check headline of articles with high probability for Topic 21
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 21,
             n = 10)

# Check headline of articles with high probability for Topic 22
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 22,
             n = 10)

# Check headline of articles with high probability for Topic 23
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 23,
             n = 10)

# Check headline of articles with high probability for Topic 24
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 24,
             n = 10)

# Check headline of articles with high probability for Topic 25
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 25,
             n = 10)

# Check headline of articles with high probability for Topic 26
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 26,
             n = 10)

# Check headline of articles with high probability for Topic 27
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 27,
             n = 10)

# Check headline of articles with high probability for Topic 28
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 28,
             n = 10)

# Check headline of articles with high probability for Topic 29
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 29,
             n = 10)

# Check headline of articles with high probability for Topic 30
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 30,
             n = 10)


# Check headline of articles with high probability for Topic 31
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 31,
             n = 10)

# Check headline of articles with high probability for Topic 32
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 32,
             n = 10)

# Check headline of articles with high probability for Topic 33
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 33,
             n = 10)

# Check headline of articles with high probability for Topic 34
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 34,
             n = 10)

# Check headline of articles with high probability for Topic 35
findThoughts(modelFit,
             texts = dfm_unsup@docvars$title,
             topics = 35,
             n = 10)

findThoughts(modelFit,
             texts = dfm_unsup@docvars$text,
             topics = 32,
             n = 1)

findThoughts(stm_model,
             texts = dfm_unsup@docvars$text,
             topics = 20,
             n = 10)

## 5. Semantic validation (topic correlations)
topic_correlations <- topicCorr(stm_model)
plot.topicCorr(topic_correlations,
               vlabels = seq(1:ncol(stm_model$theta)),
               vertex.color = "white",
               main = "Topic correlations")

## 6. Topic quality (semantic coherence and exclusivity)
topicQuality(model = stm_model,
             documents = stmdfm$documents,
             xlab = "Semantic Coherence",
             ylab = "Exclusivity",
             labels = 1:ncol(stm_model$theta),
             M = 15)

