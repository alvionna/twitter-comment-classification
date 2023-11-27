# twitter-comment-classification

## About the Project 

We build 3 classification Machine Learning models: k-Nearest Neighbour (kNN), Support Vector Machine (SVM), and Random Forest, to classify twitter comments into 6 categories: happy, sad, angry, fear, disgust, and surprise 

### Data Sources
* [Twitter sentiment classification using distant supervision by Go, A., Bhayani, R. and Huang, L (2009)](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)

## Data Preprocessing 

### Missing Values 
There is **no** missing values, and therefore, there's no need to remove any rows or process the data to handle missing values. 

### Data Distribution 
As we can see, the data distribution is right-skewed. So, it's pretty **imbalance**. Skewed data does not work well with many statistical methods. However, *tree based models* are not affected.
* Since it's right-skewed, the model will predict better on data points with lower value compared to those with higher values
* With these facts in mind, we will see how the skewness affect the result. 

