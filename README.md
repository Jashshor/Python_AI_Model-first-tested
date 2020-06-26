# 第一个简单的Python AI模型训练

In these two model, I change something to see difference by changing the number of “n”(also
called “k”), the splitting of data. The AUC of model is changing at the range of more than 0.01 (the
“random_state” of “train_test_split” is also varying). And by doing random splitting, training and
comparing for 50 times, it is also found that KNN model, with “n” equaling 5, won Decision Tree for
25 times by chance. They always vary nearby 0.62, meaning that the area enclosed by coordinate axis
under ROC curve is bigger than a half. So it is clear that both two models are useful because it can
make decision better than a person logically. The two models are both using the characteristics of
data to predict and it is somehow at the same way to predict, both belonging to classification algorithm.
All in all, both are very easy ways to train models and they can actually predict. I believe in
our daily life I could solve some problems by using the two models and the evaluation criterions. And
it is a happy thing to get into this area and train my first model.