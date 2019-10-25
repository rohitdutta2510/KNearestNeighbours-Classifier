import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')


#dataset
#a, b are two classes
data = {
        'a' : [[1,2],[2,3],[3,1]],
        'b' : [[6,5],[7,7],[8,6]],
        }

new_feature = [5,7]



def KNN(data, predict_data, k=4):
    if len(data) >= 3:
        warnings.warn('K is less than the number of classes')
    #KNN Algo
    euclidean_distances = []
    
    for cls in data:
        for point in data[cls]:
            distance = np.linalg.norm(np.array(point) - np.array(predict_data)) #calculating the distance of the point from the other given points
            euclidean_distances.append([distance, cls]) #giving a list of distances and how far is this new point from that class
    euclidean_distances = sorted(euclidean_distances)
    print('Distance from given points: ',euclidean_distances)    
    classes = [i[1] for i in euclidean_distances[:k]]
    print('K-Classes: ',classes)
    print('Number of classes close to the new data: ',Counter(classes))
    print('Most Common class: ', Counter(classes).most_common(1))
    prediction = Counter(classes).most_common(1)[0][0] #getting the most common occuring class
    return prediction

prediction = KNN(data, new_feature)
print("Prediction: ", prediction)




