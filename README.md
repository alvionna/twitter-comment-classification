# twitter-comment-classification

## About the Project

We build 3 classification Machine Learning models: k-Nearest Neighbour (kNN), Support Vector Machine (SVM), and Random Forest, to classify twitter comments into 6 categories: happy, sad, angry, fear, disgust, and surprise

### Data Sources

- [Twitter sentiment classification using distant supervision by Go, A., Bhayani, R. and Huang, L (2009)](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

## Data Analysis

### Missing Values

There is **no** missing values, and therefore, there's no need to remove any rows or process the data to handle missing values.

### Data Distribution

As we can see from the pictures below, the data distribution is right-skewed. So, it's pretty **imbalance**. Skewed data does not work well with many statistical methods. However, _tree based models_ are not affected.

- Since it's right-skewed, the model will predict better on data points with lower value compared to those with higher values
- With these facts in mind, we will see how the skewness affect the result.

![data distribution](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/data-dist.png?raw=true)
![the length of the tweet for each feeling](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/feeling-length.png?raw=true)

## Data Preprocessing

If we print the first 5 tweets of the comments, we can see that the comments/tweets have emojis and unnecessary characters in it.
We can improve them by doing processing of the comments so that it's clearer and help the model to understand better regarding the context of the tweet.

1. **Remove stopwords** - We are removing words that hold little meaning, such as "a", "the", "is", "are", "and", "or", etc.
2. **Lemmatization** - group together different inflected forms of the same word

## Tokenize

We use the **TF-IDF** method to tokenize the words, in which will be fed into the model for training and testing

## Dataset Split

We decided to split the dataset into a 80:20 ratio for training and testing.

## Models Construction

### K-Nearest Neighbour (KNN)

To build the optimal KNN, we experimented with 2 methods to choose the best value of _k_ where we arbitrarily set the max _k_ = 15.

1. Elbow Method
   - We plot each iteration of _k_ with the error rate at that _k_ and choose the _k_ that yields the lowest error rate.
   - Using this method, we found that _k = 11_
     ![elbow-knn-graph](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/knn-elbow.png?raw=true)
2. Cross Validation Method
   - This time, we decided to use Cross Validation to found the best _k_ by plotting each iteration of _k_ against the accuracy rate.
   - The value of _k_ that yields the highest accuracy will be the best _k_
   - Using this method, we found that _k = 8_
     ![cv-knn-graph](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/knn-cv.png?raw=true)

### Random Forest

To build the optimal Random Forest model, we experimented with the model:

1. Without Hyperparameter Tuning
   - In this random forest model, we decided set the hyperparameter arbitrarily where
     ```
     random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True
     ```
2. With Hyperparameter Tuning (**GridSearch**)

   - In this random forest model, we tried to find the best hyperparameters value by conducting _GridSearchCV()_ where

     ```
     params = {
     'max_depth': [2,3,5,10,20],
     'min_samples_leaf': [5,10,20,50,100,200],
     'n_estimators': [10,25,30,50,100,200]
     }

     grid_search = GridSearchCV(estimator = random_forest_tuned,
                           param_grid = params,
                           cv = 4, # cross validation fold
                           n_jobs = -1,
                           verbose = 1,
                           scoring = "accuracy")
     ```

### Support Vector Machine (SVM)

Similar to Random Forest, to build the optimal SVM model, we experimented with the model:

1. Without Hyperparameter Tuning
   - In this random forest model, we decided set the hyperparameter arbitrarily where
     ```
     random_state=42
     ```
2. With Hyperparameter Tuning (**GridSearch**)

   - In this random forest model, we tried to find the best hyperparameters value by conducting _GridSearchCV()_ where

     ```
     params = {
         'C':[0.01,0.1,1,10],
         'kernel' : ["linear","rbf","sigmoid"],
         'degree' : [1,3,5,7],
         'gamma' : [0.01,1] # kernel coefficient for ‘rbf’ and ‘sigmoid’.
     }

     svm_cv = GridSearchCV(estimator = svm,
                      param_grid = params,
                      cv = 4,
                      verbose = 1)
     ```

## Results

We will report the Accuracy, Precision, Recall, and F1 Score values and display the confusion matrix

### KNN

1. Elbow Method:
   - Accuracy Score: 0.7325349301397206
   - Precision Score: 0.7325349301397206
   - Recall Score: 0.7325349301397206
   - F1 Score: 0.7325349301397206
     ![elbow-knn-confusion-matrix](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/knn_elbow_cm.png?raw=true)
2. CV Method:
   - Accuracy Score: 0.7145708582834331
   - Precision Score: 0.7145708582834331
   - Recall Score: 0.7145708582834331
   - F1 Score: 0.7145708582834331
     ![cv-knn-confusion-matrix](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/knn_cv_cm.png?raw=true)

### Random Forest

1. Without Hyperparameter Tuning:
   - Accuracy Score: 0.4301397205588822
   - Precision Score: 0.430139720558882
   - Recall Score: 0.4301397205588822
   - F1 Score: 0.4301397205588822
     ![rf-ori-confusion-matrix](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/rf_ori_cm.png?raw=true)
2. With Grid Search:
   - Best Parameters: {'max_depth': 20, 'min_samples_leaf': 5, 'n_estimators': 25}
   - Accuracy Score: 0.5858283433133733
   - Precision Score: 0.5858283433133733
   - Recall Score: 0.5858283433133733
   - F1 Score: 0.5858283433133733
     ![rf-gridsearch-confusion-matrix](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/rf_gridsearch_cm.png?raw=true)

### Support Vector Machine

1. Without Hyperparameter Tuning:

- Accuracy Score: 0.7894211576846307
- Precision Score: 0.7894211576846307
- Recall Score: 0.7894211576846307
- F1 Score: 0.7894211576846307
  ![svm-ori-confusion-matrix](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/svm_ori_cm.png?raw=true)

2. With Grid Search:

- Best Parameters: {'C': 10, 'degree': 1, 'gamma': 1, 'kernel': 'sigmoid'}
- Accuracy Score: 0.8977045908183633
- Precision Score: 0.8977045908183633
- Recall Score: 0.8977045908183633
- F1 Score: 0.8977045908183633
  ![svm-gridsearch-confusion-matrix](https://github.com/[alvionna]/[twitter-comment-classification]/blob/[images]/svm_gridsearch_cm.png?raw=true)

## Conclusion and Discussion

1. SVM with `GridSearch` yields the best result with accuracy of 89%
2. Random Forest yields the worst result among the models with accuracy of 43% without hyperparameter tuning and accuracy of 58% with `GridSearch`
3. `GridSearch` improves the performance of the model
4. KNN with Cross Validation yields poorer result than KNN without Cross Validation

## In-Progress

1. LSTM-based Models
2. Transformer-based Models
