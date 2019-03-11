import numpy as np
import operator

def manhattan(vector_1, vector_2):
    """
        This is the L1 Norm
    """
    if len(vector_1)!=len(vector_2):
        print("Error! Vectors are not of the same length!")
        return
    else:
        new_value = 0
        for i in range(len(vector_1)):
            new_value+=np.abs(vector_1[i] - vector_2[i])
        return new_value
    
def cosine(vector_1, vector_2):
    """
        This is 1 - cosine of the angle between the two vector
        This ensures ( I am assuming ) all the properties that every standard norm will satisfy
    """
    if len(vector_1)!=len(vector_2):
        print("Error! Vectors do not have the same dimension")
        return
    else:
        new_value = 0
        for i in range(len(vector_1)):
            new_value+=vector_1[i]*vector_2[i]
        new_value/=(np.linalg.norm(vector_1)*np.linalg.norm(vector_2))
        return (1 - new_value)

def knn_predict(train_x, train_y, test_x, problem_type='classification',k=5, distance_metric = 'Euclidean'):
    
    if problem_type=='classification':
        """
            Do classification
        """
        if distance_metric == 'Euclidean':
            final_labels = []
            for elem in test_x:
                distance = []
                for index in range(len(train_x)):
                    distance.append((np.linalg.norm(train_x[index] - elem),index))
                distance = sorted(distance, key=lambda x:x[0])
                result = distance[:k]
                new_dict = {}
                for elem in result:
                    if y_train[elem[1]] in new_dict:
                        new_dict[y_train[elem[1]]]+=1
                    else:
                        new_dict[y_train[elem[1]]]=1
                key_max_value = max(new_dict.items(), key=operator.itemgetter(1))[0]
                final_labels.append(key_max_value)
            return np.array(final_labels)

        
        if distance_metric=='Manhattan':
            final_labels = []
            for elem in test_x:
                distance = []
                for index in range(len(train_x)):
                    distance.append((manhattan(train_x[index],elem),index))
                distance = sorted(distance, key=lambda x:x[0])
                result = distance[:k]
                new_dict = {}
                for elem in result:
                    if y_train[elem[1]] in new_dict:
                        new_dict[y_train[elem[1]]]+=1
                    else:
                        new_dict[y_train[elem[1]]]=1
                key_max_value = max(new_dict.items(), key=operator.itemgetter(1))[0]
                final_labels.append(key_max_value)
            return np.array(final_labels)
                
        if distance_metric == 'Cosine':
            final_labels = []
            for elem in test_x:
                distance = []
                for index in range(len(train_x)):
                    distance.append((cosine(elem,train_x[index]),index))
                distance = sorted(distance, key=lambda x:x[0])
                result = distance[:k]
                new_dict = {}
                for elem in result:
                    if y_train[elem[1]] in new_dict:
                        new_dict[y_train[elem[1]]]+=1
                    else:
                        new_dict[y_train[elem[1]]]=1
                key_max_value = max(new_dict.items(), key=operator.itemgetter(1))[0]
                final_labels.append(key_max_value)
            return np.array(final_labels)
            
    if problem_type == 'regression':
        """
            Do Regression
        """
        if distance_metric == 'Euclidean':
            final_value = []
            for elem in test_x:
                distance = []
                for index in range(len(train_x)):
                    distance.append((np.linalg.norm(train_x[index] - elem),index))
                distance = sorted(distance, key=lambda x:x[0])
                result = distance[:k]
                mean_value = 0
                for i in range(len(result)):
                    mean_value+=y_train[result[i][1]]
                mean_value/=k
                final_value.append(mean_value)
            return np.array(final_value)
        
        if distance_metric=='Manhattan':
            final_labels = []
            for elem in test_x:
                distance = []
                for index in range(len(train_x)):
                    distance.append((manhattan(train_x[index],elem),index))
                distance = sorted(distance, key=lambda x:x[0])
                result = distance[:k]
                mean_value = 0
                for i in range(len(result)):
                    mean_value+=y_train[result[i][1]]
                mean_value/=k
                final_value.append(mean_value)
            return np.array(final_value)
        
        if distance_metric == 'Cosine':
            final_labels = []
            for elem in test_x:
                distance = []
                for index in range(len(train_x)):
                    distance.append((cosine(elem,train_x[index]),index))
                distance = sorted(distance, key=lambda x:x[0])
                result = distance[:k]
                mean_value = 0
                for i in range(len(result)):
                    mean_value+=y_train[result[i][1]]
                mean_value/=k
                final_value.append(mean_value)
            return np.array(final_value)
        
    if problem_type!='classification' or problem_type!='regression':
        print("Invalid Type of Problem - Kindly choose between classification and regression")
        return
