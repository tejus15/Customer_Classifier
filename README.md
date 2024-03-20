# CLASSIFICATION MODELS

# PROBLEM STATEMENT
A direct marketing firm mails catalogs to its customer base of about 5 million households. Customers respond either by ordering items from the catalog, or do not respond. The firm distinguishes itself by mailing expensive catalogs, and – while the response rates to the firm's mailers are higher than the industry average (30% vs 22%) – they incur considerable printing and mailing costs. They are trying to improve their performance by identifying and targeting profitable customers, i.e., customers who are likely to respond (and order items that would justify the printing and mailing costs). They are particularly interested in lapsing customers (customers who made their last purchase 13 to 24 months ago). A preliminary study shows that customers seem to make their buying decision in two phases – they decide whether or not to respond first and, if they decide to respond, make a follow-up decision on what to order. dmtrain.csv contains information about 2,000 customers from the last mailing campaign. Everyone included has made at least one purchase from the firm in the past. The variables involved are:
<br>
<table>
<th>Variables</th><th>Description</th>
  <tr><td>id</td><td>customer ID</td> </tr>
  <tr><td>n24</td><td>number of orders in the last 24 months</td> </tr>
  <tr><td>rev24</td><td>total order amount ($) in the last 24 months</td> </tr>
  <tr><td>revlast</td><td>amount of last order ($)</td> </tr>
  <tr><td>elpsdm</td><td>time elapsed since last order (months)</td> </tr>
  <tr><td>ordfreq</td><td> order frequency over the last 24 months <br>
(1, 2, 3 → actual number of orders; 4 → 4 or more orders)</td> </tr>
  <tr><td>ordcat</td><td> order amount category <br>
(1 → $0.01–$1.99, 2 → $2.00–$2.99, 3 → $3.00–$4.99, 4 → $5.00–$9.99, <br>
5 → $10.00–$14.99, 6 → $15.00–$24.99, 7 → over $25.00)</td> </tr>
  <tr><td>response</td><td>1 → customer responded, 0 → no response</td> </tr>
</table>

# OBJECTIVE
The broad objective is to classify customers who are likely to respond to mailers, and customers who are not (i.e., response) is the dependent variable


# PROCESS
We created histograms for all numeric variables. We found that n24, rev24, and revlast were strongly right skewed. Elpsdm was far less skewed, and ordfreq and ordcat were not skewed. We transformed the first three variables with a log transformation to reduce skewness. We also checked if our dataset had any missing values. 
![image](https://github.com/tejus15/Customer_Classifier/assets/78174194/1cecb0a1-3d68-4eeb-892f-867bd934a430)

With the data ready, we decided to build our first classication model - Decision Tree. Our decision tree had a depth of 26. 

![image](https://github.com/tejus15/Customer_Classifier/assets/78174194/74bfbc82-6be1-40ee-89a1-f58908790085)

<br>To avoid overfitting, we will be pruning the decision tree. To determine the best tree depth, we used 10-fold cross validation, by trying as many possible depths as we could. Based on the results decision tree of depth 1 has the highest accuracy (70.8%) and that is what we recommend the depth of the tree should be after pruning it.   
  
The figure below is the decision tree with depth 1. Here, it shows that ordfreq, is significant in influence the response class.

 
![image](https://github.com/tejus15/Customer_Classifier/assets/78174194/1a002430-2d39-4bd4-adb0-28c1d8557fa4)


Now we are developing a random forest classifier with 100 trees. We used 10-fold cross-validation to be consistent and to compare with other models built in this project. We ran tests on the values of 1, 2, and 3 depths with 100 trees in the random forest. We found that, in this version, depth 3 performed the best, with 71.2% accuracy.
This model performs slightly better with depth 3 as the accuracy is 71.2%. The accuracy of depth 1 is the same for decision tree and random forest.  

We performed the same test with 50 trees and received the same results. Depth 3 remains our recommendation with the highest accuracy of 71.20%. Since 50 and 100 trees give the same result, we chose the random forest classifier with 50 trees.
 
 
We now consider k-nearest neighbor modes. Using the 10-fold cross validation, we identified the best value of k, by trying various values of k (5-10). 
Before proceeding with KNN, we standardized the independent variables because KNN depends on distances between data points. Standardizing the data ensures that few variables does not have a greater impact than the others. 
We performed a k-nearest-neighbor validation for values of k from 5-10. We recommend k values of 6, 8, 10 with accuracies of 69.5%, 69.7%, and 70.65% respectively. We chose k=10 since it has the highest accuracy among the others. Among the models built so far, this model performs worst in terms of accuracy.
![image](https://github.com/tejus15/Customer_Classifier/assets/78174194/0850faca-a6fe-403b-90ca-5ae4f827fb1e)


Next we developed a logistic model using the entire training data. Using 10-fold cross validation to evaluate the model, we found it had an accuracy of 71.15%, which is has higher accuracy than other models expect for Random Forest with 50 trees. 
 
Having created all the models, we ran a 10-fold cross validation model on all the best models from each classifier based on accuracy, and compared their cross-validation scores.
We found that, in this version, all models performed between 70.8% and 71.2%. However, KNN (with 10 neighbors) performed the worst, and Random Forest model performed the best with 71.2%.
![image](https://github.com/tejus15/Customer_Classifier/assets/78174194/c53827cd-7576-4c40-8d1a-89143c70c108)

We fit the Random Forest model with 50 trees and depth 3 to the entire dataset and created a final recommended model. This model has an accuracy of 71.2%.

We used this model to make our predictions. We read the file dmtest.csv and made predictions (using the final model) on which customers are likely to respond, and which are not. The predicted values were either 0 or 1.
As part of this project, we created a file group05dmtest.csv that adds a column named “prediction” to the original variables in dmtest.csv. 

Using the best classifier model on the test dataset (Random Forest Classifier), we found that it predicted only 2 ‘1s’ out of 2000 i.e. 0.1% of the customers will respond to the mailers. Since most people will not respond to the mailers, the expected count of 0 as a response is significantly higher than 1. Thus, a model that predicts more zeroes is most likely to have high accuracy. This is one of the pitfalls of using accuracy as a metric to compare different models.

We filtered the rows and created a dataset for the lapsing customer (customers who made their last purchase 13 to 24 months ago). Approximately 85% of the records in the training dataset are from lapsing customers, 15% of the records from the non-lapsing customers. We compared the accuracy scores of all the best models from each category (KNN, logistic regression, random forest, and decision tree) to find the model with best accuracy when it is trained using the lapsing customer data. We used 10 folds cross validation to evaluate our models, with accuracy as the scoring metric. We found that the best model for lapsing customer data is the Logistic Regression Model which is different from our earlier best model (Random Forest) as it gives the highest accuracy.

![image](https://github.com/tejus15/Customer_Classifier/assets/78174194/e8bf3d77-85a0-4a57-85ef-09beed89ad95)

 
 

We created two disjoint datasets from our original dataset – One containing lapsing customers and the other containing non-lapsing customers. 
We evaluated our Logistic Regression model using the non-lapsing customer dataset using 10 folds cross validation. These are the results we got:<br>
Lapsing Customers: Accuracy of Logistic Regression = 72.59% <br>


# CREDITS

Tejus Sanjay Sharma
Sumi Batas
Mihir Hirave
Vipul Suresh Sonje
Rishika Rakwal
Syam Menon

