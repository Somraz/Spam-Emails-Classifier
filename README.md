# Spam-Emails-Classifier
This project uses a Naive Bayes model to identify spam and ham(non-spam) emails.
First, I have pre-processed the data by performing lemmetization, stop-word removal and frequency pruning so that it is suitable for the ML algorithm that creates the model. 
I have used the public lingspam data set to test the model. lingspam contains ten folders each of which contains some spam and some ham emails. 
I have used 9 folders for creating the model and the last one for testing. This is done 10 times wherein each folder is used as a test set exactly once. This is called 10-fold cross validation. 
The average of the 10 performance values is the final performance on this data set.
