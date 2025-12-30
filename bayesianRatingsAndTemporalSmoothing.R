set.seed(151)


library("MASS")
library("readr")
library("dplyr")
library("knitr")
library("pander")
library("car")
library("klaR")
library("tree")
library("rpart")
library("rpart.plot")
library("randomForest")



mdtest <- file.choose()
mdtrain <- file.choose()
matchesDataTest <- read.csv(mdtest)
matchesDataTrain <- read.csv(mdtrain)

#switching test and train if needed
m <- matchesDataTest
matchesDataTest <- matchesDataTrain
matchesDataTrain <- m


#eda
boxplot(scoreDiff ~ bluewin,
        data = matchesDataTrain, main = "exp score vs blue win")
boxplot(rpDiff ~ bluewin,
        data = matchesDataTrain, main = "exp rp vs blue win")
boxplot(autoDiff ~ bluewin,
        data = matchesDataTrain, main = "exp auto vs blue win")
boxplot(lastFive ~ bluewin,
        data = matchesDataTrain, main = "exp lastFiveWR vs blue win")
boxplot(linkPoints ~ bluewin,
        data = matchesDataTrain, main = "exp LP vs blue win")
boxplot(endgamePoints ~ bluewin,
        data = matchesDataTrain, main = "exp EP vs blue win")



##lda
bluewin_lda <- lda(bluewin ~ scoreDiff,
                   data = matchesDataTrain)


lda_preds <- predict(bluewin_lda, as.data.frame(matchesDataTest))

conf_matrixLDA <- table(lda_preds$class, matchesDataTest$bluewin)
print(conf_matrixLDA)
TP <- conf_matrixLDA["1", "1"]
TN <- conf_matrixLDA["0", "0"]
FP <- conf_matrixLDA["1", "0"]
FN <- conf_matrixLDA["0", "1"]

accuracy <- (TP + TN) / sum(conf_matrixLDA)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

#73 to 78% accuracy


##qda
bluewin_qda <- qda(bluewin ~ scoreDiff + rpDiff + autoDiff + lastFive + linkPoints,
                   data = matchesDataTrain)


qda_preds <- predict(bluewin_qda, as.data.frame(matchesDataTest))

conf_matrixQDA <- table(qda_preds$class, matchesDataTest$bluewin)
print(conf_matrixQDA)
TP <- conf_matrixQDA["1", "1"]
TN <- conf_matrixQDA["0", "0"]
FP <- conf_matrixQDA["1", "0"]
FN <- conf_matrixQDA["0", "1"]

accuracy <- (TP + TN) / sum(conf_matrixQDA)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

#73 to 78% accuracy



#logistic

bluewin_logit <- glm(bluewin ~ scoreDiff + autoDiff + linkPoints + lastFive,
                     data = matchesDataTrain,
                     family = binomial(link = "logit"))

logit_prob <- predict(bluewin_logit,
                      as.data.frame(matchesDataTest),
                      type = "response")
logit_preds <- (logit_prob > 0.5)

conf_matrixLL <- table(logit_preds, matchesDataTest$bluewin)
print(conf_matrixLL)

TP <- conf_matrixLL["TRUE", "1"]
TN <- conf_matrixLL["FALSE", "0"]
FP <- conf_matrixLL["TRUE", "0"]
FN <- conf_matrixLL["FALSE", "1"]

accuracy <- (TP + TN) / sum(conf_matrixLL)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

#88.07% accuracy when ran with the optimal test train combination

#gets slightly better than 64% with score, rp; 55% with test

##tree
tree_model <- rpart(bluewin ~ scoreDiff + rpDiff + autoDiff + lastFive + linkPoints + endgamePoints, data = matchesDataTrain, method = "class")
predictions <- predict(tree_model, matchesDataTest, type = "class")
print(predictions)
conf_matrix <- table(matchesDataTest$bluewin, predictions)

TP <- conf_matrix["1", "1"]
TN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]

accuracy <- (TP + TN) / sum(conf_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

##73% acc with 3 vars in train data; 73 on test too



#random forest

matchesDataTrain$bluewin <- as.factor(matchesDataTrain$bluewin)
matchesDataTest$bluewin <- as.factor(matchesDataTest$bluewin)
rf_model <- randomForest(bluewin ~ scoreDiff + rpDiff + autoDiff + lastFive + linkPoints, data = matchesDataTrain, ntree = 500, importance = TRUE)
varImpPlot(rf_model)
rf_preds <- predict(rf_model, matchesDataTest)
conf_matrix <- table(rf_preds, matchesDataTest$bluewin)
print(conf_matrix)

TP <- conf_matrix["1", "1"]
TN <- conf_matrix["0", "0"]
FP <- conf_matrix["1", "0"]
FN <- conf_matrix["0", "1"]

accuracy <- (TP + TN) / sum(conf_matrix)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

#74% max with hyperparam tuning


#Bayesian Ratings and Temporal Smotthing Added

# Initialize team strength priors
initial_rating <- 0
k_update <- 0.15      # learning rate (Bayesian-style update strength)
alpha <- 0.9          # temporal smoothing factor (EWMA)

#  team rating tables
teams <- unique(c(matchesDataTrain$blueTeam, matchesDataTrain$redTeam))
ratings <- setNames(rep(initial_rating, length(teams)), teams)

# Function to update ratings after a match
update_ratings <- function(blue, red, outcome, ratings) {
  expected_blue <- 1 / (1 + exp(-(ratings[blue] - ratings[red])))
  error <- outcome - expected_blue
  
  ratings[blue] <- alpha * ratings[blue] + (1 - alpha) * (ratings[blue] + k_update * error)
  ratings[red]  <- alpha * ratings[red]  + (1 - alpha) * (ratings[red]  - k_update * error)
  
  return(ratings)
}

# Apply rating updates sequentially on training data
for (i in seq_len(nrow(matchesDataTrain))) {
  ratings <- update_ratings(
    blue = matchesDataTrain$blueTeam[i],
    red  = matchesDataTrain$redTeam[i],
    outcome = matchesDataTrain$bluewin[i],
    ratings = ratings
  )
}

# Add smoothed rating difference as a new feature
matchesDataTrain$ratingDiff <- ratings[matchesDataTrain$blueTeam] -
                               ratings[matchesDataTrain$redTeam]

matchesDataTest$ratingDiff <- ratings[matchesDataTest$blueTeam] -
                              ratings[matchesDataTest$redTeam]

# -------------------------------
# Re-train a simple classifier using the smoothed Bayesian rating
# -------------------------------

bluewin_logit_bayes <- glm(
  bluewin ~ ratingDiff + scoreDiff + autoDiff + linkPoints,
  data = matchesDataTrain,
  family = binomial(link = "logit")
)

bayes_prob <- predict(
  bluewin_logit_bayes,
  matchesDataTest,
  type = "response"
)

bayes_preds <- bayes_prob > 0.5

conf_matrix_bayes <- table(bayes_preds, matchesDataTest$bluewin)
print(conf_matrix_bayes)

TP <- conf_matrix_bayes["TRUE", "1"]
TN <- conf_matrix_bayes["FALSE", "0"]
FP <- conf_matrix_bayes["TRUE", "0"]
FN <- conf_matrix_bayes["FALSE", "1"]

accuracy <- (TP + TN) / sum(conf_matrix_bayes)
print(paste("Accuracy with Bayesian smoothed ratings:", round(accuracy * 100, 2), "%"))
