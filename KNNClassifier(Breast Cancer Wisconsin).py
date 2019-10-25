import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random


def KNN(data, predict_data, k=3):
    if len(data) >= 3:
        warnings.warn('K is less than the number of classes')
    #KNN Algo
    euclidean_distances = []
    
    for cls in data:
        for point in data[cls]:
            distance = np.linalg.norm(np.array(point) - np.array(predict_data)) #calculating the distance of the point from the other given points
            euclidean_distances.append([distance, cls]) #giving a list of distances and how far is this new point from that class
    euclidean_distances = sorted(euclidean_distances)
    #print('Distance from given points: ',euclidean_distances)    
    classes = [i[1] for i in euclidean_distances[:k]]
    #print('K-Classes: ',classes)
    #print('Number of classes close to the new data: ',Counter(classes))
    #print('Most Common class: ', Counter(classes).most_common(1))
    prediction = Counter(classes).most_common(1)[0][0] #getting the most common occuring class
    return prediction


#Class: (2 for benign, 4 for malignant)
df = pd.read_csv('breast-cancer-wisconsin.csv')
df.replace('?',-99999,inplace=True) #making the missing value an outlier
df.drop(['id'],1,inplace=True) #dropping the id column it doesn't cause cancer
dataset = df.astype(float).values.tolist() #converting the dataframe to a 2D list

random.shuffle(dataset) #shuffling the dataset

test_size = 0.25 #setting the size for the training and testing the data

train_data = dataset[:-int(test_size * len(dataset))]
test_data = dataset[-int(test_size * len(dataset)):]

#print("Training Data: ", train_data)
#print("Test Data: ", test_data)

#data from the data set i.e the list will be kept according their class in these dictionaries
#empty training set
train_set = {2 : [], 4 : []}

#empty testing set
test_set = {2 : [], 4 : []}


for i in train_data:
    train_set[i[-1]].append(i[:-1]) #i[-1] as the last column is the class and the appending it to the dictionary according to the class

for j in test_data:
    test_set[j[-1]].append(j[:-1])

correct = 0
total = 0


def name(num):
    if group == 2:
        return 'Benign'
    elif group == 4:
        return 'Malignant'

for group in test_set:
    for data in test_set[group]:
        prediction = KNN(train_set, data, k=5)
        '''
        if prediction == 2:
            print("Predicted: Benign", ", Actual:", name(group))
        if prediction == 4:
            print("Predicted: Malignant", ", Actual:", name(group))
        '''
        if group == prediction:
            correct += 1
        total += 1

print("Accuracy: ", correct/total)








