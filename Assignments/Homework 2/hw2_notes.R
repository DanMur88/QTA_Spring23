# Set working directory and read in data
setwd(getwd())
data <- read.csv("./data/yelp_data_small.csv", 
                 stringsAsFactors=FALSE,
                 encoding = "utf-8")

# Load required libraries
library(quanteda)
library(quanteda.textmodels)
library(quanteda.textstats)
library(caret)
library(stringi)
library(ggplot2)
library('MLmetrics')
library('doParallel')
library(textstem)

# Convert dataframe to corpus
corp <- quanteda::corpus(data,
                         text_field = "text")

# Inspect corpus attributes
summary(corp,5)

# Inspect sentiment frequencies of corpus
table(corp$sentiment)

# Calculate probability of "positive" class
cat("\n","Probability of 'positive' class: ", round(length(corp[corp$sentiment == "pos"])/length(corp),2))

# 2. Create a document-feature matrix using this corpus. 
# Process the text so as to increase predictive power of the features. 
# Justify each of your processing decisions in the context of the 
# supervised classification task.

# Remove line breaks from corpus
corp <- gsub("\\\\n", "", corp)

head(corp)

# Tokenise text and remove punctuation/symbols/URLs, etc.
toks <- quanteda::tokens(corp, 
                         include_docvars = TRUE,
                         remove_numbers = TRUE,
                         remove_punct = TRUE,
                         remove_symbols = TRUE,
                         remove_hyphens = TRUE,
                         remove_separators = TRUE,
                         remove_url = TRUE)

# Convert tokens to lowercase
toks <- tokens_tolower(toks)

# Create list of stopwords and remove these from tokens, retaining whitespaces in place of removed stopwords
stop_list <- stopwords("english")
toks <- tokens_remove(toks, stop_list, padding=TRUE)

# Identify trigram collocations and order by frequency
trigrams <- textstat_collocations(toks, 
                                  method = "lambda",
                                  size = 3,
                                  min_count = 5,
                                  smoothing = 0.5)

# Create list of trigrams of interest (z-score > 3)
trigrams <- trigrams[trigrams$z > 3,]

# Identify trigram collocations and order by frequency
bigrams <- textstat_collocations(toks, 
                                 method = "lambda",
                                 size = 2,
                                 min_count = 20,
                                 smoothing = 0.5)

# Create list of bigrams of interest (z-score > 30)
bigrams <- bigrams[bigrams$z > 30,]

# Compile a list of collocations to keep
keep_coll_list <- rbind(trigrams, bigrams)

# Merge collocations with tokens and remove whitespaces and ampersands
toks_col <- tokens_compound(toks, pattern = keep_coll_list)
toks_col <- tokens_remove(toks_col, "")

# Convert to document feature matrix
dfm <- dfm(toks_col)

# Check top features of dfm
topfeatures(dfm, 50)

# Check frequency of dfm features
tfreq <- textstat_frequency(dfm)

# Plot feature frequency charts
ggplot(tfreq[1:100,], aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency")

ggplot(tfreq[101:500,], aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency")

ggplot(tfreq[501:1000,], aes(x = reorder(feature, frequency), y = frequency)) +
  geom_point() +
  coord_flip() +
  labs(x = NULL, y = "Frequency")

# Trim and weight dfm 
dfm <- dfm_trim(dfm, min_docfreq = 20)
dfm <- dfm_tfidf(dfm)

# Save document feature matrix
saveRDS(dfm, "data/dfm")

# Convert dfm to data frame for input into supervised ML pipeline
tmpdata <- convert(dfm, to = "data.frame", docvars = NULL)

# Drop doc_id variable
tmpdata <- tmpdata[, -1]

# Get sentiment labels
human_labels <- dfm@docvars$sentiment

# Join sentiment labels to data frame
tmpdata <- as.data.frame(cbind(human_labels, tmpdata))

## Create Train, Test and Validation Sets ##

# Set seed for replicability
set.seed(1234)

# Randomly order labelled dataset
tmpdata <- tmpdata[sample(nrow(tmpdata)),]

# Determine cutoff point for 5% validation set
split <- round(nrow(tmpdata) * 0.05)

# Create validation set
vdata <- tmpdata[1:split,]

# Create labelled dataset minus validation set
ldata <- tmpdata[(split + 1):nrow(tmpdata),]

# Create training and test sets
train_row_nums <- createDataPartition(ldata$human_labels, 
                                      p=0.8, 
                                      list=FALSE) # set human_labels

Train <- ldata[train_row_nums, ] 
Test <- ldata[-train_row_nums, ]

# Specify cross-validation conditions
train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 3,
                              classProbs= TRUE, 
                              summaryFunction = multiClassSummary,
                              selectionFunction = "best",
                              verboseIter = TRUE)

## Train Model ##

# View default parameter settings for Naive Bayes algorithm
modelLookup(model = "naive_bayes")

# Set number of cores for parallel processing
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

# Train model
nb_train1 <- train(human_labels ~ ., 
                  data = Train,  
                  method = "naive_bayes", 
                  metric = "F1",
                  trControl = train_control,
                  tuneGrid = expand.grid(laplace = c(0,1),
                                         usekernel = c(TRUE, FALSE),
                                         adjust = c(0.75, 1, 1.25, 1.5)),
                  allowParallel= TRUE)


# Save model
saveRDS(nb_train, "data/nb_train1")

# Print cross-validation results
print(nb_train1)

## Evaluate Model Performance on Test Set ##

# Generate prediction on Test set using training set model
pred1 <- predict(nb_train1, newdata = Test)
head(pred1) # first few predictions

# Generate confusion matrix 
confusionMatrix(reference = as.factor(Test$human_labels),
                data = pred1, 
                mode='everything', 
                positive='neg')

nb_final1 <- train(human_labels ~ ., 
                  data = ldata,  
                  method = "naive_bayes", 
                  trControl = trainControl(method = "none"),
                  tuneGrid = data.frame(nb_train1$bestTune))

nb_final1

# Generate prediction on Validation set using training set model
pred2 <- predict(nb_final1, newdata = vdata)
head(pred2) # first few predictions

# Generate confusion matrix
confusionMatrix(reference = as.factor(vdata$human_labels), 
                data = pred2, 
                mode='everything', 
                positive='neg')

## Iteration 2 ##

# Create and trim new dfm
dfm2 <- dfm(toks_col)

# Check term keyness and save separate objects for "pos" and "neg"
dfm2_keyness <- dfm_group(dfm2, groups = sentiment)
keyness_stat <- textstat_keyness(dfm2_keyness, target = "pos")

# Compile list of features with low predictive power
key_features <- keyness_stat[keyness_stat$p > 0.05,]

# Remove features with low predictive power from dfm
dfm2 <- dfm_remove(dfm2, pattern = key_features$feature)

# Weight dfm
dfm2 <- dfm_tfidf(dfm2)

# Save document feature matrix
saveRDS(dfm2, "data/dfm2")

# Convert dfm to data frame for input into supervised ML pipeline
tmpdata2 <- convert(dfm2, to = "data.frame", docvars = NULL)

# Drop doc_id variable
tmpdata2 <- tmpdata2[, -1]

# Get sentiment labels
human_labels2 <- dfm2@docvars$sentiment

# Join sentiment labels to data frame
tmpdata2 <- as.data.frame(cbind(human_labels2, tmpdata2))

## Create Train, Test and Validation Sets ##

# Set seed for replicability
set.seed(1234)

# Randomly order labelled dataset
tmpdata2 <- tmpdata2[sample(nrow(tmpdata2)),]

# Determine cutoff point for 5% validation set
split2 <- round(nrow(tmpdata2) * 0.05)

# Create validation set
vdata2 <- tmpdata2[1:split2,]

# Create labelled dataset minus validation set
ldata2 <- tmpdata2[(split2 + 1):nrow(tmpdata2),]

# Create training and test sets
train_row_nums2 <- createDataPartition(ldata2$human_labels2, 
                                      p=0.8, 
                                      list=FALSE) # set human_labels

Train2 <- ldata2[train_row_nums2, ] 
Test2 <- ldata2[-train_row_nums2, ]

# Specify cross-validation conditions
train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 3,
                              classProbs= TRUE, 
                              summaryFunction = multiClassSummary,
                              selectionFunction = "best",
                              verboseIter = TRUE)

## Train Model ##

# View default parameter settings for Naive Bayes algorithm
modelLookup(model = "naive_bayes")

# Set number of cores for parallel processing
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

# Train model
nb_train2 <- train(human_labels2 ~ ., 
                   data = Train2,  
                   method = "naive_bayes", 
                   metric = "F1",
                   trControl = train_control,
                   tuneGrid = expand.grid(laplace = c(0,1),
                                          usekernel = c(TRUE, FALSE),
                                          adjust = c(0.75, 1, 1.25, 1.5)),
                   allowParallel = TRUE)

# Save model
saveRDS(nb_train2, "data/nb_train2")

# Stop parallel processing
stopCluster(cl)

# Print cross-validation results
print(nb_train2)

## Evaluate Model Performance on Test Set ##

# Generate prediction on Test set using training set model
pred3 <- predict(nb_train2, newdata = Test2)
head(pred3) # first few predictions

# Generate confusion matrix 
confusionMatrix(reference = as.factor(Test2$human_labels2),
                data = pred3, 
                mode='everything', 
                positive='neg')

nb_final2 <- train(human_labels2 ~ ., 
                   data = ldata2,  
                   method = "naive_bayes", 
                   trControl = trainControl(method = "none"),
                   tuneGrid = data.frame(nb_train2$bestTune))

nb_final2

# Generate prediction on Validation set using training set model
pred4 <- predict(nb_final2, newdata = vdata2)
head(pred4) # first few predictions

# Generate confusion matrix
confusionMatrix(reference = as.factor(vdata2$human_labels2), 
                data = pred4, 
                mode='everything',
                positive='neg')

## Train SVM ###

# Check model parameters
modelLookup(model = "svmLinear")

# Define parameter settings to iterate over
tuneGrid = expand.grid(C = c(0.5, 1, 1.5))

# Allow parallel processing
registerDoParallel(cl)

# Train model
svm_train1 <- train(human_labels ~ ., 
                   data = Train,  
                   method = "svmLinear", 
                   metric = "F1",
                   trControl = train_control,
                   tuneGrid = expand.grid(C = c(0.5, 1, 1.5)),
                   allowParallel= TRUE)

# Save model
saveRDS(svm_train1, "data/svm_train1")

# Stop parallel processing
stopCluster(cl)

unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

unregister_dopar()

# Print cross-validation results
print(svm_train1)

# Generate prediction on test set using training set model
pred_svm1 <- predict(svm_train1, newdata = Test)

# Generate confusion matrix
confusionMatrix(reference = as.factor(Test$human_labels), 
                data = pred_svm1, 
                mode='everything', 
                positive='neg')

# Finalise model
svm_final1 <- train(human_labels ~ ., 
                   data = ldata,  
                   method = "svmLinear", 
                   trControl = trainControl(method = "none"),
                   tuneGrid = data.frame(svm_train1$bestTune))
# Save model
saveRDS(svm_final1, "data/svm_final1")

print(svm_final1)

# Run prediction on validation set
svm_pred2 <- predict(svm_final1, newdata = vdata)

# Generate confusion matrix
confusionMatrix(reference = as.factor(vdata$human_labels), 
                data = svm_pred2, 
                mode='everything', 
                positive='neg')


## Train SVM - Iteration 2 ###

# Define parameter settings to iterate over
tuneGrid = expand.grid(C = c(0.5, 1, 1.5))

# Allow parallel processing
registerDoParallel(cl)

# Train model
svm_train2 <- train(human_labels2 ~ ., 
                    data = Train2,  
                    method = "svmLinear", 
                    metric = "F1",
                    trControl = train_control,
                    tuneGrid = expand.grid(C = c(0.5, 1, 1.5)),
                    allowParallel= TRUE)

# Save model
saveRDS(svm_train1, "data/svm_train1")

# Stop parallel processing
stopCluster(cl)

# Print cross-validation results
print(svm_train1)

# Generate prediction on test set using training set model
pred_svm1 <- predict(svm_train1, newdata = Test)

# Generate confusion matrix
confusionMatrix(reference = as.factor(Test$human_labels), 
                data = pred_svm1, 
                mode='everything', 
                positive='neg')

# Finalise model
svm_final1 <- train(human_labels ~ ., 
                    data = ldata,  
                    method = "svmLinear", 
                    trControl = trainControl(method = "none"),
                    tuneGrid = data.frame(svm_train1$bestTune))
# Save model
saveRDS(svm_final1, "data/svm_final1")

print(svm_final1)

# Run prediction on validation set
svm_pred2 <- predict(svm_final1, newdata = vdata)

# Generate confusion matrix
confusionMatrix(reference = as.factor(vdata$human_labels), 
                data = svm_pred2, 
                mode='everything', 
                positive='neg')
