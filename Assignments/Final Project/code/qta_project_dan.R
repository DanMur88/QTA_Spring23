## Load required packages
library(tidyverse)
library(guardianapi)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(readtext)
library(stringi)
library(stm)
library(lubridate)

## Acquire Corpus Documents via Guardian API ##

# Run interactive function for inputting key
gu_api_key()

# Query API for articles about China from 6-months prior to CPC National Congress to today  
dat <- gu_content(query = "China", 
                  from_date = "2022-04-16")

# Save data
saveRDS(dat, "data/dat")

# Read in data
dat <- readRDS("data/dat")

# Create duplicate to work on
df <- dat

# Inspect first data entries
head(df)

# Check for duplicates and remove
which(duplicated(df$web_title) == TRUE)
df <- df[!duplicated(df$web_title),]


# Check unique section names
print(unique(dat['section_id']),n=50)

# Count number of articles for specific sections
nrow(df[df$section_id == "business",])

# Filter data to include only specific sections and document types of interest
df <- df[df$type == "article" & df$first_publication_date > "2022-04-16" & 
           df$section_id %in% c("business","world","technology",
                                                   "commentisfree","us-news",
                                                   "australia-news","uk-news",
                                                   "politics"),]

## Create Corpus and DFM ##

# Create corpus
corpus_guar <- corpus(df, 
                 docid_field = "web_title", 
                 text_field = "body_text")

# Check the corpus
summary(corpus_guar, 5)

# Read first five entries
as.character(corpus_guar)[1:3]

# Remove headline from body text
stri_replace_first(corpus_guar, 
                   replacement = "",
                   regex = "^.+?\"")

# Add original text as docvar
docvars(corpus_guar)$text <- texts(corpus_guar)

# Inspect corpus attributes
summary(corpus_guar,5)

# Tokenise text and remove punctuation/symbols/URLs, etc.
toks_guar <- quanteda::tokens(corpus_guar,
                         include_docvars = TRUE,
                         remove_numbers = TRUE,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_hyphens = TRUE,
                         remove_separators = TRUE,
                         remove_url = TRUE)

# Convert tokens to lowercase
toks_guar <- tokens_tolower(toks_guar)

# Create list of stopwords and remove these from tokens, retaining whitespaces in place of removed stopwords
stop_list <- stopwords("english")
toks_guar <- tokens_remove(toks_guar, stop_list, padding=TRUE)

# Identify trigram collocations
trigrams_guar <- textstat_collocations(toks_guar,
                                  method = "lambda",
                                  size = 3,
                                  min_count = 20,
                                  smoothing = 0.5)

# Create list of trigrams of interest (z-score > 30)
trigrams_guar <- trigrams_guar[trigrams_guar$z > 1.5,]

# Identify bigram collocations
bigrams_guar <- textstat_collocations(toks_guar,
                                 method = "lambda",
                                 size = 2,
                                 min_count = 20,
                                 smoothing = 0.5)

# Create list of bigrams of interest (z-score > 30)
bigrams_guar <- bigrams_guar[bigrams_guar$z > 30,]

# Merge collocations with tokens and remove whitespaces
toks_col_guar <- tokens_compound(toks_guar, pattern = trigrams_guar)
toks_col_guar <- tokens_compound(toks_col_guar, pattern = bigrams_guar)
toks_col_guar <- tokens_remove(toks_col_guar, "")

# Generate dfm and trim
dfm_guar <- dfm(toks_col_guar)
dfm_guar <- dfm_trim(dfm_guar, min_docfreq = 20)

# Check top features of dfm
topfeatures(dfm_guar, 50)

# Parse dates within dfm docvars
dfm_guar@docvars$date <- dfm_guar@docvars[["first_publication_date"]]
dfm_guar@docvars$date <- date(dfm_guar@docvars[["first_publication_date"]])
dfm_guar@docvars$date_month <- floor_date(dfm_guar@docvars$date, "month")

# Save document feature matrix
saveRDS(dfm_guar, "data/dfm_guar")

# Read in document feature matrix
dfm_guar <- readRDS("data/dfm_guar")

## Sentiment Analysis ##

# Create dfm object of LSD words matching corpus
dfm_guar_sent <- dfm_lookup(dfm_guar, dictionary =
                            data_dictionary_LSD2015[1:2]) %>%
  dfm_group(groups = date)

# Check first 5 entries
head(dfm_guar_sent,5)

# Calculate proportion of negative words to original dfm
docvars(dfm_guar_sent, "prop_negative") <- as.numeric(dfm_guar_sent[,1] / 
                                                      ntoken(dfm_guar_sent))

# Calculate proportion of positive words to original dfm
docvars(dfm_guar_sent, "prop_positive") <- as.numeric(dfm_guar_sent[,2] / 
                                                      ntoken(dfm_guar_sent))

# Calculate net sentiment
docvars(dfm_guar_sent, "net_sentiment") <- docvars(dfm_guar_sent, "prop_positive") - 
  docvars(dfm_guar_sent, "prop_negative")

# Plot net sentiment over time
docvars(dfm_guar_sent) %>%
  ggplot(aes(x = date, y = net_sentiment)) +
  geom_smooth() +
  labs(title = "Sentiment of Guardian articles about China over time", 
       x = "date", y = "net sentiment")

## STM Model ##

# Convert dfm to stm
stmdfm_guar <- convert(dfm_guar, to = "stm")

# Run stm algorithm for 35 topics (k=35)
modelFit_guar <- stm(documents = stmdfm_guar$documents,
                     vocab = stmdfm_guar$vocab,
                     K = 35,
                     prevalence = ~ s(as.numeric(date_month)),
                     data = stmdfm_guar$meta,
                     max.em.its = 500,
                     init.type = "Spectral",
                     seed = 1234,
                     verbose = TRUE)

# Save model
saveRDS(modelFit_guar, "data/STM_model_guar1")

# Read in model
stm_model_guar1 <- readRDS("data/STM_model_guar1")

# Check top terms per label
labelTopics(stm_model_guar1)

# Plot topic prevalence and top terms per topic by FREX metric
plot.STM(stm_model_guar1, 
         type = "summary", 
         labeltype = "frex",
         text.cex = 0.7,
         main = "Topic prevalence and top terms (FREX)")

# Plot topic prevalence and top terms per topic by Probability metric
plot.STM(stm_model_guar1, 
         type = "summary", 
         labeltype = "prob",
         text.cex = 0.7,
         main = "Topic prevalence and top terms (Probability)")

# Check topic headlines
findThoughts(stm_model_guar1,
             texts = dfm_guar@docvars$headline,
             topics = 35,
             n = 20)

# Check topic articles
findThoughts(stm_model_guar1,
             texts = dfm_guar@docvars$text,
             topics = 35,
             n = 20)

## Predictive validity using time series data ##

# Convert metadata to correct format
stmdfm_guar$meta$num_month <- month(stmdfm_guar$meta$date)

# Aggregate topic probability by month
agg_theta <- setNames(aggregate(stm_model_guar1$theta,
                                by = list(month = stmdfm_guar$meta$date_month),
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

## Semantic validation (topic correlations) ##
topic_correlations <- topicCorr(stm_model_guar1)
plot.topicCorr(topic_correlations,
               vlabels = seq(1:ncol(stm_model_guar1$theta)),
               vertex.color = "white",
               main = "Topic correlations")

## Topic quality (semantic coherence and exclusivity) ##
topicQuality(model = stm_model_guar1,
             documents = stmdfm_guar$documents,
             xlab = "Semantic Coherence",
             ylab = "Exclusivity",
             labels = 1:ncol(stm_model_guar1$theta),
             M = 15)
