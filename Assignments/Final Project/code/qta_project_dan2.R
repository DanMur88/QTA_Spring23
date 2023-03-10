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
library(dplyr)


## Acquire Corpus Documents via Guardian API ##

# Run interactive function for inputting key
gu_api_key()

# Query API for articles about China from 6-months prior to CPC National Congress to today  
dat2 <- gu_content(tag = "world/china",
                  from_date = "2022-04-16")

# Save data
saveRDS(dat2, "data/dat2")

# Read in data
dat2 <- readRDS("data/dat2")

# Create duplicate to work on
df2 <- dat2

# Inspect first data entries
head(df2)

# Check for duplicates and remove
which(duplicated(df2$web_title) == TRUE)
df2 <- df2[!duplicated(df2$web_title),]


# Check unique section names
print(unique(dat['section_id']),n=50)

# Count number of articles for specific sections
nrow(df[df$section_id == "science",])

# Filter data to include only specific sections and document types of interest
df2 <- df2[df2$type == "article" & df2$first_publication_date > "2022-04-16" & 
           df2$section_id %in% c("world","business","technology",
                                "commentisfree","us-news","australia-news",
                                "uk-news","politics","environment", "news"),]

## Create Corpus and DFM ##

# Create corpus
corpus_guar2 <- corpus(df2, 
                 docid_field = "web_title", 
                 text_field = "body_text")

# Check the corpus
summary(corpus_guar2, 5)

# Read first five entries
as.character(corpus_guar2)[1:3]

# Remove headline from body text
stri_replace_first(corpus_guar2, 
                   replacement = "",
                   regex = "^.+?\"")

# Add original text as docvar
docvars(corpus_guar2)$text <- texts(corpus_guar2)

# Inspect corpus attributes
summary(corpus_guar2,5)

# Tokenise text and remove punctuation/symbols/URLs, etc.
toks_guar2 <- quanteda::tokens(corpus_guar2,
                         include_docvars = TRUE,
                         remove_numbers = TRUE,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_hyphens = TRUE,
                         remove_separators = TRUE,
                         remove_url = TRUE)

# Convert tokens to lowercase
toks_guar2 <- tokens_tolower(toks_guar2)

# Create list of stopwords and remove these from tokens, retaining whitespaces in place of removed stopwords
stop_list <- stopwords("english")
toks_guar2 <- tokens_remove(toks_guar2, stop_list, padding=TRUE)

# Identify trigram collocations
trigrams_guar2 <- textstat_collocations(toks_guar2,
                                  method = "lambda",
                                  size = 3,
                                  min_count = 5,
                                  smoothing = 0.5)

# Create list of trigrams of interest (z-score > 30)
trigrams_guar2 <- trigrams_guar2[trigrams_guar2$z > 1,]

# Identify bigram collocations
bigrams_guar2 <- textstat_collocations(toks_guar2,
                                 method = "lambda",
                                 size = 2,
                                 min_count = 10,
                                 smoothing = 0.5)

# Create list of bigrams of interest (z-score > 30)
bigrams_guar2 <- bigrams_guar2[bigrams_guar2$z > 30,]

# Merge collocations with tokens and remove whitespaces
toks_col_guar2 <- tokens_compound(toks_guar2, pattern = trigrams_guar2)
toks_col_guar2 <- tokens_compound(toks_col_guar2, pattern = bigrams_guar2)
toks_col_guar2 <- tokens_remove(toks_col_guar2, "")

# Generate dfm and trim
dfm_guar2 <- dfm(toks_col_guar2)
dfm_guar2 <- dfm_trim(dfm_guar2, min_docfreq = 5)

# Check top features of dfm
topfeatures(dfm_guar2, 50)

# Parse dates within dfm docvars
dfm_guar2@docvars$date <- dfm_guar2@docvars[["first_publication_date"]]
dfm_guar2@docvars$date <- date(dfm_guar2@docvars[["first_publication_date"]])
dfm_guar2@docvars$date_month <- floor_date(dfm_guar2@docvars$date, "month")

# Save document feature matrix
saveRDS(dfm_guar2, "data/dfm_guar2")

# Read in document feature matrix
dfm_guar2 <- readRDS("data/dfm_guar2")

## Sentiment Analysis ##

# Create dfm object of LSD words matching corpus
dfm_guar_sent2 <- dfm_lookup(dfm_guar2, dictionary =
                            data_dictionary_LSD2015[1:2]) %>%
  dfm_group(groups = date)

# Check first 5 entries
head(dfm_guar_sent2,5)

# Calculate proportion of negative words to original dfm
docvars(dfm_guar_sent2, "prop_negative") <- as.numeric(dfm_guar_sent2[,1] / 
                                                      ntoken(dfm_guar_sent2))

# Calculate proportion of positive words to original dfm
docvars(dfm_guar_sent2, "prop_positive") <- as.numeric(dfm_guar_sent2[,2] / 
                                                      ntoken(dfm_guar_sent2))

# Calculate net sentiment
docvars(dfm_guar_sent2, "net_sentiment") <- docvars(dfm_guar_sent2, "prop_positive") - 
  docvars(dfm_guar_sent2, "prop_negative")

# Plot net sentiment over time
docvars(dfm_guar_sent2) %>%
  ggplot(aes(x = date, y = net_sentiment)) +
  geom_smooth() +
  labs(title = "Sentiment of Guardian articles about China over time", 
       x = "date", y = "net sentiment")

## STM Model ##

# Determine optimal k number
?searchK # Check help file to assist with interpreting results
kResult <- searchK(documents = stmdfm_guar2$documents,
                   vocab = stmdfm_guar2$vocab,
                   K=c(10:35), # search a range of possible K
                   init.type = "Spectral",
                   data = stmdfm_guar2$meta,
                   prevalence = ~ section_name + s(month(date)))

# Save model
saveRDS(kResult, "data/kResult")

# Plot kResult
plot(kResult)

# Convert dfm to stm
stmdfm_guar2 <- convert(dfm_guar2, to = "stm")

# Run stm algorithm for 20 topics (k=20)
STM_model_guar2 <- stm(documents = stmdfm_guar2$documents,
                     vocab = stmdfm_guar2$vocab,
                     K = 20,
                     prevalence = ~ s(as.numeric(date_month)),
                     data = stmdfm_guar2$meta,
                     max.em.its = 500,
                     init.type = "Spectral",
                     seed = 1234,
                     verbose = TRUE)

# Save model
saveRDS(STM_model_guar2, "data/STM_model_guar2")

# Read in model
STM_model_guar2 <- readRDS("data/STM_model_guar2")

# Check top terms per label
labelTopics(STM_model_guar2)

# Plot topic prevalence and top terms per topic by FREX metric
plot.STM(STM_model_guar2, 
         type = "summary", 
         labeltype = "frex",
         text.cex = 0.7,
         main = "Topic prevalence and top terms (FREX)")

# Plot topic prevalence and top terms per topic by Probability metric
plot.STM(STM_model_guar2, 
         type = "summary", 
         labeltype = "prob",
         text.cex = 0.7,
         main = "Topic prevalence and top terms (Probability)")

# Check topic headlines
findThoughts(STM_model_guar2,
             texts = dfm_guar2@docvars$headline,
             topics = 20,
             n = 20)

# Check topic articles
findThoughts(STM_model_guar2,
             texts = dfm_guar2@docvars$text,
             topics = 10,
             n = 20)

## Predictive validity using time series data ##

# Convert metadata to correct format
stmdfm_guar2$meta$num_month <- month(stmdfm_guar2$meta$date)

# Aggregate topic probability by month
agg_theta <- setNames(aggregate(STM_model_guar2$theta,
                                by = list(month = stmdfm_guar2$meta$date_month),
                                FUN = mean),
                      c("month", paste("Topic",1:20)))
agg_theta <- pivot_longer(agg_theta, cols = starts_with("T")) 

# Aggregate topic probability by month (Topic 1-5)
agg_theta1 <- agg_theta[agg_theta$name %in% c("Topic 1", "Topic 2", "Topic 3",
                                              "Topic 4", "Topic 5"),]
# Rename topics
agg_theta1 <- agg_theta1 %>%
  mutate(name = recode(name, "Topic 1" = 'Opinion pieces', "Topic 2" = 'China-Australia relations', 
                       "Topic 3" = 'Chinese interference abroad',
                       "Topic 4" = 'Covid', "Topic 5" = 'Business & economic development'))

# Aggregate topic probability by month (Topic 6-10)
agg_theta2 <- agg_theta[agg_theta$name %in% c("Topic 6", "Topic 7", "Topic 8",
                                              "Topic 9", "Topic 10"),]
# Rename topics
agg_theta2 <- agg_theta2 %>%
  mutate(name = recode(name, "Topic 6" = 'Extreme weather events', "Topic 7" = 'Taiwan', 
                       "Topic 8" = 'Hong Kong', "Topic 9" = 'Environment',
                       "Topic 10" = 'Ukraine war'))

# Aggregate topic probability by month (Topic 11-15)
agg_theta3 <- agg_theta[agg_theta$name %in% c("Topic 11", "Topic 12", "Topic 13",
                                              "Topic 14", "Topic 15"),]
# Rename topics
agg_theta3 <- agg_theta3 %>%
  mutate(name = recode(name, "Topic 11" = 'China-Australia trade', 
                       "Topic 12" = 'Protests & dissent', 
                       "Topic 13" = 'International travel restrictions',
                       "Topic 14" = 'Human rights abuses', "Topic 15" = 'Surveillance'))

# Aggregate topic probability by month (Topic 16-20)
agg_theta4 <- agg_theta[agg_theta$name %in% c("Topic 16", "Topic 17", "Topic 18",
                                              "Topic 19", "Topic 20"),]
# Rename topics
agg_theta4 <- agg_theta4 %>%
  mutate(name = recode(name, "Topic 16" = 'Global economy', "Topic 17" = 'Human interest/exposés',
                       "Topic 18" = 'Xi Jinping',
                       "Topic 19" = 'Oceania', "Topic 20" = 'Espionage'))

# Plot aggregated theta over time (Topic 1-5)
ggplot(data = agg_theta1,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

# Plot aggregated theta over time (Topic 6-10)
ggplot(data = agg_theta2,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

# Plot aggregated theta over time (Topic 11-15)
ggplot(data = agg_theta3,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

# Plot aggregated theta over time (Topic 16-20)
ggplot(data = agg_theta4,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

## Semantic validation (topic correlations) ##
topic_correlations <- topicCorr(STM_model_guar2)
plot.topicCorr(topic_correlations,
               vlabels = seq(1:ncol(STM_model_guar2$theta)),
               vertex.color = "white",
               main = "Topic correlations")

## Topic quality (semantic coherence and exclusivity) ##
topicQuality(model = STM_model_guar2,
             documents = stmdfm_guar2$documents,
             xlab = "Semantic Coherence",
             ylab = "Exclusivity",
             labels = 1:ncol(STM_model_guar2$theta),
             M = 15)
