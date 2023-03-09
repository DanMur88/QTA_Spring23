###############################
# Tutorial 5: Unsupervised ML #
###############################

## Load packages
pkgTest <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[,  "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg,  dependencies = TRUE)
  sapply(pkg,  require,  character.only = TRUE)
}

# Acquire stmBrowser package from github
if(!require(devtools)) install.packages("devtools")
library(devtools)
install_github("mroberts/stmBrowser",dependencies=TRUE)

lapply(c("tidyverse",
         "quanteda",
         "quanteda.textstats",
         "lubridate",
         "stm",
         "wordcloud",
         "stmBrowser",
         "LDAvis"),
       pkgTest)

## 1. Read in and wrangle data
#     a) In the data folder you'll find a large data.frame object called 
#        ukr_h1_2022. Read it in, and check the type of articles it contains.

dat <- readRDS("data/ukr_h1_2022")

#     b) Pre-process the data.frame.

# Inspect data
head(dat)
summary(dat)  

dat$body_text <- str_replace(dat$body_text, "\u2022.+$","")

corp <- corpus(dat,
               docid_field = "headline",
               text_field = "body_text")

source("code/pre_processing.R")
prepped_toks <- prep_toks(corp)
collocations <- get_coll(prepped_toks)
toks <- tokens_compound(prepped_toks, pattern = collocations[collocations$z > 10,])
toks <- tokens_remove(tokens(toks),"")
toks <- tokens_remove(tokens(toks),"said")

dfm <- dfm(toks)
dfm <- dfm_trim(dfm, min_docfreq = 20)

## 2. Perform STM
# Convert dfm to stm
stmdfm <- convert(dfm, to = "stm")

# Set k
K <- 7

# Run STM algorithm
modelFit <- stm(documents = stmdfm$documents,
                vocab = stmdfm$vocab,
                K = K,
                prevalence = ~ section_name + s(month(date)),
                #prevalence = ~ source + s(as.numeric(date_month)), 
                data = stmdfm$meta,
                max.em.its = 500,
                init.type = "Spectral",
                seed = 2023,
                verbose = TRUE)

# Save your model!
saveRDS(modelFit, "data/modelFit")

# Load model (in case your computer is running slow...)
#modelFit <- readRDS("data/modelFit")

## 3. Interpret Topic model 
# Inspect most probable terms in each topic
labelTopics(modelFit)

?labelTopics()

# Further interpretation: plotting frequent terms
plot.STM(modelFit, 
         type = "summary", 
         labeltype = "frex", # plot according to FREX metric
         text.cex = 0.7,
         main = "Topic prevalence and top terms")

plot.STM(modelFit, 
         type = "summary", 
         labeltype = "prob", # plot according to probability
         text.cex = 0.7,
         main = "Topic prevalence and top terms")

# Use wordcloud to visualise top terms per topic
cloud(modelFit,
      topic = 1,
      scale = c(3.5, 0.5),
      max.words = 50)

# Reading documents with high probability topics: the findThoughts() function
findThoughts(modelFit,
             texts = dfm@docvars$standfirst, # If you include the original corpus text, we could refer to this here
             topics = 1,
             n = 10)

## 4. Topic validation: predictive validity using time series data
#     a) Convert metadata to correct format
#stmdfm$meta$num_month <- as.numeric(stmdfm$meta$date_month)
stmdfm$meta$num_month <- month(stmdfm$meta$date)

#     b) Aggregate topic probability by month
agg_theta <- setNames(aggregate(modelFit$theta,
                                by = list(month = stmdfm$meta$num_month),
                                FUN = mean),
                      c("month", paste("Topic",1:K)))
agg_theta <- pivot_longer(agg_theta, cols = starts_with("T")) 
# Pivoting longer to have topics appear in one column, to allow 
# aggregation/grouping by topic

#     c) Plot aggregated theta over time
ggplot(data = agg_theta,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

## 5. Semantic validation (topic correlations)
topic_correlations <- topicCorr(modelFit)
plot.topicCorr(topic_correlations,
               vlabels = seq(1:ncol(modelFit$theta)), # we could change this to a vector of meaningful labels
               vertex.color = "white",
               main = "Topic correlations")

# Negative/neutral correlations show us that topics are distinct/have 
# minimal overlap.

## 6. Topic quality (semantic coherence and exclusivity)
topicQuality(model = modelFit,
             documents = stmdfm$documents,
             xlab = "Semantic Coherence",
             ylab = "Exclusivity",
             labels = 1:ncol(modelFit$theta),
             M = 15)

# We want our topics to appear in the top right here; bottom left might
# indicate junk topics. Review these and see if you may want to ignore
# them in analysis.

# An alternative approach, using underlying functions
SemEx <- as.data.frame(cbind(c(1:ncol(modelFit$theta)), 
                             exclusivity(modelFit),
                             semanticCoherence(model = modelFit,
                                               documents = stmdfm$documents,
                                               M = 15)))

colnames(SemEx) <- c("k", "ex", "coh")

SemExPlot <- ggplot(SemEx, aes(coh, ex)) +
  geom_text(aes(label=k)) +
  labs(x = "Semantic Coherence",
       y = "Exclusivity",
       title = "Topic Semantic Coherence vs. Exclusivity") +
  geom_rug() +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(colour = "gray", linewidth=1))
SemExPlot

# Inspect outliers
labelTopics(modelFit,
            topics = c(6,3))

## 7. Extended visualisations
#     a) Using LDAvis
toLDAvis(mod = modelFit,
         docs = stmdfm$documents,
         open.browser = interactive(),
         reorder.topics = TRUE)

#     b) Using stmBrowser
#        Warning: this will (silently) change your working directory!
stmBrowser(mod = modelFit,
           data = stmdfm$meta,
           covariates = c("section_name", "num_month"),
           text = "standfirst",
           n = 1000)

## 8. Estimating covariate effects
#     a) Calculate
estprop <- estimateEffect(formula = c(1:ncol(modelFit$theta)) ~ section_name + s(num_month),
                          modelFit,
                          metadata = stmdfm$meta,
                          uncertainty = "Global",
                          nsims = 25)

summary(estprop)

#     b) Plot topic probability differences
custom_labels <- seq(1:K)
plot.estimateEffect(x = estprop,
                    #model = modelFit,
                    method = "difference",
                    covariate = "section_name",
                    cov.value1 = "World",
                    cov.value2 = "Opinion",
                    topics = estprop$topics,
                    #xlim = c(-.05, .05),
                    labeltype = "custom",
                    custom.labels = custom_labels)

# Plots which categories are more/less likely to appear in reference
# category (ref = cov.value1? Check)

#     c) Plot topic probability over time (similar to above plot in 4.)
plot.estimateEffect(x = estprop,
                    #model = modelFit,
                    method = "continuous",
                    covariate = "num_month",
                    topics = estprop$topics,
                    #xlim = c(-.05, .05),
                    labeltype = "custom",
                    custom.labels = custom_labels,
                    printlegend = T) # Toggle this to see January

## 9. Using data to determine k
?searchK # Check help file to assist with interpreting results
kResult <- searchK(documents = stmdfm$documents,
                   vocab = stmdfm$vocab,
                   K=c(4:10), # search a range of possible K
                   init.type = "Spectral",
                   data = stmdfm$meta,
                   prevalence = ~ section_name + s(month(date)))
                   cores = 6) # This no longer works on windows 10 :(

#kResult <- readRDS("data/kResult")                   
plot(kResult)

## 10. Refine
# Now that we've run the model and evaluated the output, try to refine
# it for yourselves: can we further trim the tokens? Are there any cases
# we should drop from our corpus? (For instance, the daily digest) How
# large should we set our k? 