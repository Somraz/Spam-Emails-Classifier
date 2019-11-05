import re
import csv
import os
import pandas as pd
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk import stem
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer

row=['tag','data']
with open('emails.csv', 'a',newline='') as csvFile:
             writer = csv.writer(csvFile)
             writer.writerow(row)
csvFile.close()
for i in range(1,11):
    for filename in os.listdir("D:\\Semester 5th\\ML\\bare\\part"+str(i)+""):
        if filename.endswith(".txt"):
            loc="D:\\Semester 5th\\ML\\bare\\part"+str(i)+"\\"+filename
            f = open(loc, "r")
            text=f.read()
            name=filename[0]
            if(name=='s'):
                name="spam"
            else:
                name="ham"
            row=[name,text]
        f.close()
        with open('emails.csv', 'a',newline='') as csvFile:
             writer = csv.writer(csvFile)
             writer.writerow(row)
        csvFile.close()

stopwords = set(stopwords.words('english'))
stemmer = stem.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def lemmetization(email):
    
    email = " ".join([stemmer.stem(word) for word in email.split()])
    email = " ".join([lemmatizer.lemmatize(word, pos='v') for word in email.split()])
    return email

def stop_words(email):

    email = " ".join([word for word in email.split() if word not in stopwords])
    return email

def clean_email(email):
    
    email = re.sub(r'http\S+', ' ', email)
    email = re.sub("\d+", " ", email)
    email = email.replace('\n', ' ')
    email = email.translate(str.maketrans("", "", punctuation))
    email = email.lower()
    email = re.sub(' +', ' ',email)
    return email

with open('emails.csv', 'r') as readFile:
           reader = csv.reader(readFile)
           lines = list(reader)
for data in lines:
    data[1]=clean_email(data[1])
    
with open('emails1.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
writeFile.close()

for data in lines:
    data[1]=stop_words(data[1])
    
with open('emails2.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
writeFile.close()


for data in lines:
    data[1]=lemmetization(data[1])
    
with open('emails3.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
readFile.close()
writeFile.close()

result1=['TP','TN','FP','FN','recall','precision','f1 score','accuracy']
with open('results.csv', 'a',newline='') as csvFile:
             writer = csv.writer(csvFile)
             writer.writerow(result1)
csvFile.close()
for i in range(0,4):
    if (i==3):
        my_file = pd.read_csv("emails"+str(i)+".csv")
    else:
        my_file = pd.read_csv("emails"+str(i+1)+".csv")
    my_file = my_file[['tag', 'data']]
    final_score=0
    total_TN,total_TP,total_FN,total_FP,total_recall,total_precision,total_f1_score=0,0,0,0,0,0,0
    start_row=0
    end_row=289

    for j in range(0,10):
        msg_train=pd.concat([my_file.iloc[1:start_row,1],my_file.iloc[end_row:,1]])
        msg_test=my_file.iloc[start_row:end_row,1]
        class_train=pd.concat([my_file.iloc[1:start_row,0],my_file.iloc[end_row:,0]])
        class_test=my_file.iloc[start_row:end_row,0]

        if i==3:
            vectorizer = CountVectorizer(max_df=0.99,min_df=0.008)
        else:
            vectorizer = CountVectorizer()
        counts = vectorizer.fit_transform(msg_train.values)

        classifier= MultinomialNB()
        targets=class_train.values
        classifier.fit(counts, targets)

        test_count=vectorizer.transform(msg_test)
        predictions=classifier.predict(test_count)
        #final_score=final_score+accuracy_score(class_test,predictions)
        #print(classification_report(class_test,predictions))
        CM = confusion_matrix(class_test,predictions)
        #print(CM)

        total_TN,TN=total_TN+CM[0][0],CM[0][0]
        total_FN,FN=total_FN+CM[1][0],CM[1][0]
        total_TP,TP=total_TP+CM[1][1],CM[1][1]
        total_FP,FP=total_FP+CM[0][1],CM[0][1]
        print("True Negative=",TN)
        print("False Negative=",FN)
        print("True Positive=",TP)
        print("False Positive=",FP)
        
        recall=CM[1][1]/(CM[1][1]+CM[1][0])
        total_recall=recall+total_recall
        print("recall=",recall)

        precision=CM[1][1]/(CM[1][1]+CM[0][1])
        precision=total_precision+precision
        print("precision=",precision)

        f1_score=2*((precision*recall)/(precision+recall))
        total_f1_score=total_f1_score+f1_score
        print("f1 score=",f1_score)

        accuracy=(CM[0][0]+CM[1][1])/(CM[0][0]+CM[1][1]+CM[0][1]+CM[1][0])
        print("accuracy=",accuracy)

        final_score=final_score+accuracy
        result1=[TP,TN,FP,FN,recall,precision, f1_score,accuracy]
        with open('results.csv', 'a',newline='') as csvFile:
             writer = csv.writer(csvFile)
             writer.writerow(result1)
        csvFile.close()
        
        start_row=start_row+289
        end_row=end_row+289

    result1=[total_TP/10,total_TN/10,total_FP/10,total_FN/10,total_recall/10,total_precision/10,total_f1_score/10,final_score/10,"Average scores"]
    with open('results.csv','a') as csvFile:
        writer=csv.writer(csvFile)
        writer.writerow(result1)
    csvFile.close()












