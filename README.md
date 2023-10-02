# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
&ensp;The purpose of the analysis is to evaluate how accuarte the model is at predicting whether a borrower status is healthy or high-risk loan. 
* Explain what financial information the data was on, and what you needed to predict.
&ensp; The financial is based off of historical lending activity from a peer-to-peer lending services company. Features include loan size, interest rate, debt to income, borrower's income, number of accounts, total debt. What we are trying to predict is the loan status: Healthy or High Risk.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
&ensp;Trying to predict 1's and 0's. 0 being healthy, 1 being high risk. Value counts lets us know the number of each class in the dataset before the the data set is split into testing and training. 
* Describe the stages of the machine learning process you went through as part of this analysis.
**1. Process:**
&ensp;Clean the data by sepearting the Features(X) from the Target (y). This was done byh dropping ['loan_status'] column and assigning it to why before doing so. Then split the data into training and testing dataset.
**2. Train:**
&ensp;Fit/train the model using the training X, y data
**3. Validate:**
&ensp;Validate by using the subset of data that is meant for testing 
**4. Predict:**
&ensp;Using the predict function, we use the y test data to show the prediction results 
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
&ensp;I used the logistic regression method as the classification model to predict high-risk and healthy loan. The prediction values are descrete and binary (1, 0). We also used Random Over Sampler Model for resampling as we could see that there was an imbalance between the 2 loans. There were 75036 healthy loan vs 2500 high-risk loan. Using RandomOverSample, it ended up adding more rows of high-risk loans to only the training data so that both loans have the exact same amount of rows. 

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
 The dataset can be susceptible to an imbalance since the number of healthy loan outweights high-risk loans. When it came to predicting borrowers that had healthy loan status (0), all 100% were actually healthy loans. Out of all the borrowers that did have healthy loan status, the model predicted correctly for 99% of those borrowers. Along with the high precision and recall percentage score, it doesn't surprise us that the f1-score is 1.00 indicating that the model does a good job at predicting healthy loans. <br>On the other hand, when it comes to predicting high-risk loan, it doesn't do as good of a job. Precision is only 85%. Looking at the confusing matrix, you can see why that is. there were 102 False Positive, which is quite high considering 563 were True Positive. The recall sits at 91%, which is a better score than precision since there were 56 False Negative. F1-score also demonstrates the model isn't as strong at predicting whether or not a borrower is a high-risk or healthy loan status. 


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
Based on the f1-score, nothing has really changed for predicting healthy loan since the model has proven that it can do so very well. Accuracy still remains the same at 99%.<br>However, there was a decline in the precision going from 85% to 84%. If we look at the Confusion Matrix, the number of False Positive did increase 102 to 116, as well as the True Positive from 563 to 615. The recall on the other hand did increase from 91% to 99%. Looking at the confusion matrix, it makes sense since the number of False Negative did decrease from 56 to 4. The f1-score did increase by 3% to 91%, meaning that the model did improve slightly when it came to predicting high-risk loan. 

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
&ensp;When we resampled the training data, it performed than the orginial training data. Even though the precision for the high-risk data decrease by 1%, overall it is still perform the best. The recall had improved and so did the f-1 score. 
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? ) 
&ensp; The performance depends on the given data and how we process it. However, depending on what the bank values more between 1 and 0 then yes we would want to either increase the precentage for recalls or precision. Do we care more about minimizing the mistake the model makes at guessing high-risk correctly or value more that the model guess the number of high-risk correctly out of the borrowers that are actually high-risk. 

* If you do not recommend any of the models, please justify your reasoning.
&ensp; I would recommend this model because it has demonstrated high accuracy especially for healthy loans. high-risk loan prediction performed fairly well. Since we're not trying to prediction more than 2 classes, it also fits what you're trying to do. It's easy to understand and the model accomplishes our goal in predicting the probabilty of 1's and 0's. However, down the line if the model doesn't seem to be working and getting too big to handle, then we may have to shift to random forest. 

## Resources
1. How to Interprest classification Report: https://www.statology.org/sklearn-classification-report/
<br>2. Random Resampling Method for Imbalanced Data with Imblear: https://hersanyagci.medium.com/random-resampling-methods-for-imbalanced-data-with-imblearn-1fbba4a0e6d3