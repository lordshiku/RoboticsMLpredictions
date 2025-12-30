**FRC Match Outcome Prediction (matches from 2023, released Jan 2025) – ML Classification**  

**Project Overview**    

This project applies machine learning classification to predict the outcome of FIRST Robotics Competition (FRC) 3v3 matches for the 2023 season. Using match data from The Blue Alliance API, I collected and processed thousands of matches, creating a dataset that enables real-time, blind prediction of whether the blue or red alliance will win.  

Recent improvements to the implementation include **Bayesian team ratings with temporal smoothing**, which provide a dynamically updated measure of each team’s strength over the season. This allows the model to capture evolving team performance trends beyond static historical averages, improving predictive accuracy in real-time match scenarios.  

**Quick Summary**    

Predicted the rate at which blue alliance would win with a peak of 80.07% accuracy over hundreds of matches using logistic regression on a train-test paradigm.  
These prediction results actually outpace the models put out by Statbotics on the same competitions. Try it yourself!  
The model has been posted to ChiefDelphi to give teams around the globe access to these accurate *live* predictions.  
As of now, at least 5–10 teams have tried it and found useful, correct predictions in the 2025 season.  

With the addition of **Bayesian ratings and temporal smoothing**, the model can track team performance trends throughout the season, improving predictions for teams whose strength changes dynamically across matches.  

**Data Collection & Feature Engineering**  

1. **Data Retrieval**  
   Used The Blue Alliance API (handled in `dataHolder` files) to extract raw match data.  

2. **Key Predictive Metrics**  
   For each match, I calculated six key metrics that capture the difference between the blue and red alliances:

   scorediff – Average total point difference between alliances.  
   rpdiff – Difference in ranking points (awarded for specific achievements).  
   autodiff – Difference in autonomous period points.  
   endgamediff – Difference in endgame points.  
   linkdiff – Difference in link (cycle completion) points.  
   lastfive – Difference in win rate over the last five matches.  

   Each metric is calculated as:  

   **blue alliance’s average value - red alliance’s average value**  

3. **Calculation Breakdown Example: scorediff**  

   For match *n*, scorediff is calculated by:  

   1. Compute each individual team’s average past match score (before match *n*).  
   2. Sum these averages for the three blue alliance teams.  
   3. Sum these averages for the three red alliance teams.  
   4. Take the difference:  

   \[
   \text{scorediff}_n = \left( \frac{\sum_{\text{past scores of } B_1, B_2, B_3}}{3} \right) - \left( \frac{\sum_{\text{past scores of } R_1, R_2, R_3}}{3} \right)
   \]

   This ensures the metric reflects an expected scoring difference based purely on historical data before match *n*, allowing for live, blind predictions without data leakage.  

4. **Exploratory Data Analysis (EDA)**  
   Used 2D visualization and correlation analysis (`eda_analysis.R`) to determine which features were most predictive of match outcomes.  
   The most influential features were scorediff, rpdiff, and lastfive.  

5. **Dataset Construction**  
   The dataset (`csv_creator.R`) was structured so that each row represents a match, with pre-match feature values ensuring no future data influences predictions.  

**Bayesian Ratings & Temporal Smoothing**  

To capture evolving team performance trends, the implementation now incorporates:  

1. **Bayesian-Style Ratings**  
   - Each team starts with a prior strength (`initial_rating = 0`).  
   - Ratings are updated sequentially after each match using a Bayesian-style update based on match outcome versus expected outcome:  

     \[
     E_{\text{blue}} = \frac{1}{1 + \exp(-(R_{\text{blue}} - R_{\text{red}}))}, \quad 
     \text{error} = \text{outcome} - E_{\text{blue}}
     \]

     \[
     R_{\text{blue,new}} = \alpha \cdot R_{\text{blue}} + (1-\alpha) \cdot (R_{\text{blue}} + k_{\text{update}} \cdot \text{error})
     \]  
     \[
     R_{\text{red,new}} = \alpha \cdot R_{\text{red}} + (1-\alpha) \cdot (R_{\text{red}} - k_{\text{update}} \cdot \text{error})
     \]

2. **Temporal Smoothing (EWMA)**  
   - Smooths team ratings over time to reduce volatility using a smoothing factor (`alpha = 0.9`).  
   - Updated ratings reflect both recent match outcomes and historical trends.  

3. **Integration into Features**  
   - Smoothed rating difference (`ratingDiff = blue_rating - red_rating`) added as a new feature in training and test datasets.  
   - Used in retrained logistic regression alongside `scoreDiff, autoDiff, linkPoints`.  

4. **Performance Impact**  
   - Captures dynamic team strength changes.  
   - Improves live prediction accuracy and model adaptability during the season.  

**Machine Learning Models & Performance**  

| Model | Accuracy (%) |
|-------|-------------|
| Logistic Regression | 80.07 (Best with standard metrics) |
| Logistic Regression + Bayesian Ratings | Improved accuracy; captures dynamic team trends |
| Linear Discriminant Analysis (LDA) | 78.6 |
| Quadratic Discriminant Analysis (QDA) | 75.3 |
| Decision Trees | 72.1 |
| Random Forest (with tuning) | 79.2 |

- Logistic regression remains the most effective method.  
- Incorporating **Bayesian ratings** enables the model to adapt to team performance over time, an advancement over static historical features.  

**Files & Structure**  

- `dataHolderLarge.py` – Handles API calls to collect match data.  
- `csvCreatorLarge.py` – Processes raw data into training/test datasets.  
- `allRcode.R` – Performs EDA, classification models, and now Bayesian ratings + temporal smoothing.  
- Raw JSON and CSV examples included in the repo for reproducibility.  

**Key Results & Insights**  

- Logistic regression provided the best accuracy (80.07%), making it the most effective model for match outcome classification.  
- **Bayesian ratings with temporal smoothing** add a dynamic perspective to team strength, improving prediction of evolving teams.  
- Strongest predictive features: scorediff, rpdiff, lastfive, ratingDiff.  
- Methodology ensures live, blind predictions, allowing real-time forecasting using only prior match data.
