########################################################
##### TEAM 2. Business Case Presentation on Spotify ####
###   Authors:Altynai Beishenalieva,                  #
#             Olubiyi George,                         #
#             Diego Salinas,                          #
#             Daniel Esteban Torres,                  #
#             Polina Afanasyeva                       #
########################################################

#### Business Problem ####
#*Problem:
#* 1. Creating a model that predicts track popularity.
#* 2. Forecasting attribute values to adjust our revenue distribution algorithm.


#*Assumptions:
#*
#* Success: Popularity > 50
#* Failure: Popularity <= 50


#### Description of the initial data ####

#** tracks.csv

#* id - The Spotify ID for the track.
#* name - Name of the track.
#* popularity - The popularity of a track is a value between 0 and 100, with 100 being the most popular. 
#*              The popularity is calculated by algorithm and is based, in the most part, on the total number
#*              of plays the track has had and how recent those plays are. Generally speaking, songs that are
#*              being played a lot now will have a higher popularity than songs that were played a lot in the
#*              past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently.
#*              Artist and album popularity is derived mathematically from track popularity.
#* duration_ms - The track length in milliseconds.
#* explicit - Whether or not the track has explicit lyrics.
#* artists - The artists' names who performed the track. 
#* id_artists - ID of the artist.
#* release_date - Date of release of the track.
#* danceability - Danceability describes how suitable a track is for dancing based on a combination of musical 
#*                elements including tempo, rhythm stability, beat strength, and overall regularity. A value of
#*                0.0 is least danceable and 1.0 is most danceable.
#* energy - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. 
#*          Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, 
#*          while a Bach prelude scores low on the scale.
#* key -  The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 
#*        1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
#* loudness -  The overall loudness of a track in decibels (dB).
#* mode - Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic
#*        content is derived. Major is represented by 1 and minor is 0.
#* speechiness - Speechiness detects the presence of spoken words in a track. The more exclusively speech-like 
#*               the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. 
#*               Values above 0.66 describe tracks that are probably made entirely of spoken words. Values 
#*               between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections
#*               or layered, including such cases as rap music. Values below 0.33 most likely represent music and
#*               other non-speech-like tracks.
#* acousticness - A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 
#*                1.0 represents high confidence the track is acoustic.
#* instrumentalness - Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental
#*                    in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness
#*                    value is to 1.0, the greater likelihood the track contains no vocal content.
#* liveness - Detects the presence of an audience in the recording. Higher liveness values represent an increased 
#*            probability that the track was performed live. A value above 0.8 provides strong likelihood that 
#*            the track is live.
#* valence - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high 
#*           valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound 
#*           more negative (e.g. sad, depressed, angry).
#* tempo - The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is 
#*         the speed or pace of a given piece and derives directly from the average beat duration.
#* time_signature - An estimated time signature. The time signature (meter) is a notational convention to specify
#*                 how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating 
#*                 time signatures of 3/4, to 7/.4.



#### Libraries ####
library(tidyr)
library(dplyr)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(ggthemes)
library(plotly)
library(tseries)
library(rugarch)

#### Data Import and Cleaning ####
tracks <- read.csv('C:/Users/polin/Desktop/HULT/Forecasting/group assignment/data/tracks.csv')
head(tracks)

#artists <- read.csv('C:/Users/polin/Desktop/HULT/Forecasting/group assignment/data/artists.csv')
#head(artists)

# Checking the tracks data set
sapply(tracks,class)
summary(tracks)
dim(tracks)
str(tracks)

#From the initial analysis there are no serious outliers in the data. 

# Checking for NA data 
sum(is.na(tracks))

# Cleaning [] from the data
tracks$artists <- gsub("\'", "", tracks$artists)
tracks$artists <- gsub("\\[|\\]", "", tracks$artists)
tracks$id_artists <- gsub("\'", "", tracks$id_artists)
tracks$id_artists <- gsub("\\[|\\]", "", tracks$id_artists)


# Splitting data for further processing into year, month, and day
tracks <- separate(tracks, release_date, into = c("year", "month", "day"), sep = "-")

# Converting data character to integer type
tracks <- tracks %>%
  mutate(year = as.integer(year),
         month = as.integer(month),
         day = as.integer(day))

# Converting duration in ms to duration in min
tracks <- tracks %>%
  mutate(duration_min = duration_ms / 60000)


# Creating additional binary field for popularity flag, which will be used for predictive models.
# Track will be considered popular if the popularity value is bigger than 50
tracks <- tracks %>%
  mutate(popular = ifelse(popularity > 50, 1, 0))


##### PREDICTION MODELS #####

# Sub-setting the data set saving only the important columns
subset_tracks <- tracks[,c("popularity","duration_min","explicit","id_artists","year","month","day","danceability", "energy", "key", "loudness",
                           "mode", "speechiness", "acousticness","instrumentalness", "liveness", "valence", "tempo", "time_signature", "popular")]


# Checking the correlation between variables with the created function
corr_simple <- function(data=df,sig=0.5){
  #Convert data to numeric in order to run correlations
  #Convert to factor first to keep the integrity of the data - each value will become a number rather than turn into NA
  df_cor <- data %>% mutate_if(is.character, as.factor)
  df_cor <- df_cor %>% mutate_if(is.factor, as.numeric)
  #Run a correlation and drop the insignificant ones
  corr <- cor(df_cor)
  #Prepare to drop duplicates and correlations of 1     
  corr[lower.tri(corr,diag=TRUE)] <- NA 
  #Drop perfect correlations
  corr[corr == 1] <- NA 
  #Turn into a 3-column table
  corr <- as.data.frame(as.table(corr))
  #Remove the NA values from above 
  corr <- na.omit(corr) 
  #Select significant values  
  corr <- subset(corr, abs(Freq) > sig) 
  #Sort by highest correlation
  corr <- corr[order(-abs(corr$Freq)),] 
  #Print table
  print(corr)
  #Turn corr back into matrix in order to plot with corrplot
  mtx_corr <- reshape2::acast(corr, Var1~Var2, value.var="Freq")
  
  #Plot correlations visually
  corrplot(mtx_corr, is.corr=FALSE, tl.col="black", na.label=" ")
}

corr_simple(tracks,0.5)
corr_simple(subset_tracks,0.5)

# As we can see, there is a high correlation between such variables as energy and loudness, and energy and acousticness.
# It can be seen in the description of the variables, that energy includes both of these tracks' aspects.
# Therefore, only energy variable will be taken into account for further creation of the predictive models.

# Moreover, as the flag of popular is based on the popularity, this field will be also excluded from the final dataset.

tracks_fin <- subset_tracks[,c("duration_min","explicit","year","month","day","danceability", "energy", "key",
                           "mode", "speechiness", "instrumentalness", "liveness", "valence", "tempo",
                           "time_signature", "popular")]

# All NA's in variables month and day will be changed for 0 as the flag of the date unknown, because it also can be
# considered as an important insight in the model
tracks_fin$month[is.na(tracks_fin$month)] <- 0
tracks_fin$day[is.na(tracks_fin$day)] <- 0

# Checking that there are no more NA's in the data
sum(is.na(tracks_fin))



#### Sub-setting data into train and test ####
training_idx <- sample(1:nrow(tracks_fin), size=0.8*nrow(tracks_fin))
tracks_fin_train <- tracks_fin[training_idx,]
tracks_fin_test <- tracks_fin[-training_idx,]


#Check the data for over-/under- sampling
table(tracks_fin_train$popular)
round(prop.table(table(tracks_fin_train$popular))*100,2)

table(tracks_fin_test$popular)
round(prop.table(table(tracks_fin_test$popular))*100,2)


#### Training models ####

# Logistic regression
my_logit <- glm(popular ~ duration_min + explicit + year + day + danceability + energy + mode + speechiness +
                  instrumentalness+liveness + valence #+ month + key + tempo + time_signature
                , 
                data=tracks_fin_train, family="binomial" )

summary(my_logit)

# Calculations of coefficients of the logistic regression for further interpretation
exp(-2.224e-02)-1
# for every additional minute of music, the odd of business failure increases by 2.2%
exp(9.766e-01)-1
# For every increase in explicitness, the odd of business success increases by 166%
exp(6.392e-02)-1
# For every increase in year, the odd of business success increases by 6.6%
exp(8.778e-01)-1
# For every increase in the rating for danceability, the odd of business success increases by 140%
exp(2.788e-01)-1           
# For every unit increase in the energy, the odd of business success increases by 32%
exp(-1.437e+00)-1
# For every increase in speechiness, the odd of business failure increases by 76%
exp(-9.616e-01)-1


#########
# Extract coefficient estimates
coefficients <- coef(my_logit)

# Define the inverse of the logistic function
inv_logistic <- function(logodds) {
  odds <- exp(logodds)
  coeff <- (odds - 1)*100
  return(coeff)
}

# Transform the coefficients to the actual business units
transformed_coefficients <- inv_logistic(coefficients)

# Display the transformed coefficients and their interpretations
coefficient_names <- c("Intercept", "Duration (min)", "Explicit", "Year", "Day", "Danceability",
                       "Energy", "Mode", "Speechiness", "Instrumentalness", "Liveness", "Valence")

for (i in 1:length(transformed_coefficients)) {
  cat("Coefficient:", coefficient_names[i], "\n")
  cat("Transformed Estimate:", transformed_coefficients[i], "\n")
  cat("\n")
}


###############

# From the first run of the model we can see that month, key, tempo and time_signiture are insignificant for our model.
# Therefore, they will be excluded from consideration in the final model. Initial accuracy of the model is 0.88.

my_prediction <- predict(my_logit, tracks_fin_test, type="response")

confusionMatrix(data= as.factor(as.numeric(my_prediction>0.5)),
                reference= as.factor(as.numeric(tracks_fin_test$popular)))


# Decision tree model
my_tree <- rpart(popular ~ duration_min + explicit + year + day + danceability + energy + mode + speechiness +
                   instrumentalness+liveness + valence #+ month + key + tempo + time_signature
                 , 
                 data=tracks_fin_train, method="class" , cp=0.004)
rpart.plot(my_tree, type=1, extra=1)

my_df_tree_predict <- predict(my_tree, tracks_fin_test, type="prob")

confusionMatrix(data = as.factor(as.numeric(my_df_tree_predict[,2]>0.5)) ,
                reference= as.factor(as.numeric(tracks_fin_test$popular)))


# As we can see, decision tree is slightly better in terms of accuracy, but much better in predicting true positives.
# At the same time, decision tree generates bigger Type II error.

# Looking at variance importance
varImpDF <- data.frame(variables = names(my_tree$variable.importance),
                       importance = my_tree$variable.importance,
                       row.names = NULL)
varImpDF <- varImpDF[order(varImpDF$importance, decreasing = T),]
ggplot(varImpDF, aes(x=importance, y = reorder(variables, importance))) + 
  geom_bar(stat='identity', position = 'dodge') + 
  ggtitle('Variable Importance') + 
  theme_gdocs()  +
  ylab('Names of variables') +
  xlab('Importance')



##### FORECASTING MODELS#####

# Creating a grouped field and mean for each of the variables in a group
tracks_grouped <- tracks %>%
  group_by(year) %>%
  summarize(mean_instr = mean(instrumentalness, na.rm = TRUE),
            mean_explicit = mean(explicit, na.rm = TRUE),
            mean_dur_min = mean(duration_min, na.rm = TRUE))

# Excluding year 1900
tracks_grouped <- tracks_grouped %>%
  filter(year != 1900)

# Graphs for each variable
compare_chart1 <- ggplot(data=tracks_grouped)+
  geom_line(aes(x=year, y=mean_instr), color="blue")
print(compare_chart1)

compare_chart2 <- ggplot(data=tracks_grouped)+
  geom_line(aes(x=year, y=mean_explicit), color="green")
print(compare_chart2)

compare_chart3 <- ggplot(data=tracks_grouped)+
  geom_line(aes(x=year, y=mean_dur_min), color="red")
print(compare_chart3)


#### ADF, ACF and pACF ####

# Checking an ADF test for non-stationarity of the data
adf.test(tracks_grouped$mean_instr)
adf.test(tracks_grouped$mean_explicit)
adf.test(tracks_grouped$mean_dur_min)
#p-value is high so all are non-stationary

# Decomposition of the non-stationary data
tracks_inst_ts <- ts(tracks_grouped[,c("year", "mean_instr")], frequency = 5, start=c(1922))
dec <- decompose(tracks_inst_ts)
plot(dec)

tracks_exp_ts <- ts(tracks_grouped[,c("year", "mean_explicit")], frequency = 5, start=c(1922))
dec <- decompose(tracks_exp_ts)
plot(dec)

tracks_dur_ts <- ts(tracks_grouped[,c("year", "mean_dur_min")], frequency = 5, start=c(1922))
dec <- decompose(tracks_dur_ts)
plot(dec)


#ACF - show the moving average (MA) component
acf(tracks_grouped$mean_instr)
acf(tracks_grouped$mean_explicit)  
acf(tracks_grouped$mean_dur_min)  

#pACF - show the lag - autocorrelation - AR
pacf(tracks_grouped$mean_instr)
pacf(tracks_grouped$mean_explicit)
pacf(tracks_grouped$mean_dur_min)



#### ARIMA model ####

#Instrumentalness
instr_arima <- arima(tracks_grouped$mean_instr, 
                    order=c(2,0,5)) 
dt_pred <- predict(instr_arima, n.ahead = 5) #forecast for 5 years
dt_pred

# Create dataset for predicted values
years <- seq(2022, 2026)

lower_ci <- dt_pred$pred - 1.96 * dt_pred$se
upper_ci <- dt_pred$pred + 1.96 * dt_pred$se

initial_value <- tail(tracks_grouped$mean_instr, 1)

# Create a data frame for the initial data in 2021
initial_data <- data.frame(year = 2021, predicted_value = initial_value, lower_ci = 0, upper_ci = 0)

# Create a data frame with the predicted values and years
predicted_df <- data.frame(year = years, predicted_value = dt_pred$pred[1:5],
                           lower_ci = lower_ci,
                           upper_ci = upper_ci)

combined_data <- rbind(initial_data, predicted_df)

# Final plot for instrumentalness
ggplot(data = tracks_grouped, aes(x = year, y = mean_instr)) +
  geom_line() +
  labs(x = "Year", y = "Av. Instrumentalness", title = "Instrumentalness Forecast for 5 years") +
  
  # Overlay forecasted values
  geom_line(data = combined_data, aes(x = year, y = predicted_value), color = "red") #+
  
    #Adding CI on the graph
  #geom_ribbon(data = combined_data, aes(x = year, ymin = lower_ci, ymax = upper_ci), fill = "blue", alpha = 0.2)



#---
# Duration_min
dur_arima <- arima(tracks_grouped$mean_dur_min, 
                     order=c(3,0,5)) 
dt_pred <- predict(dur_arima, n.ahead = 5) #want to get forecasted values for 3 days out
dt_pred

# Create dataset for predicted values

lower_ci <- dt_pred$pred - 1.96 * dt_pred$se
upper_ci <- dt_pred$pred + 1.96 * dt_pred$se

initial_value <- tail(tracks_grouped$mean_dur_min, 1)

# Create a data frame for the initial data in 2021
initial_data <- data.frame(year = 2021, predicted_value = initial_value, lower_ci = 0, upper_ci = 0)

# Create a data frame with the predicted values and years
predicted_df <- data.frame(year = years, predicted_value = dt_pred$pred[1:5],
                           lower_ci = lower_ci,
                           upper_ci = upper_ci)

combined_data <- rbind(initial_data, predicted_df)

# Final plot for instrumentalness
ggplot(data = tracks_grouped, aes(x = year, y = mean_dur_min)) +
  geom_line() +
  labs(x = "Year", y = "Av. Duration in min", title = "Duration in min Forecast for 5 years") +
  
  # Overlay forecasted values
  geom_line(data = combined_data, aes(x = year, y = predicted_value), color = "red") #+

#Adding CI on the graph
#geom_ribbon(data = combined_data, aes(x = year, ymin = lower_ci, ymax = upper_ci), fill = "blue", alpha = 0.2)



