
# coding: utf-8

# In[128]:


import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from itertools import combinations


# In[129]:


dataset = load_iris()['data']
target = load_iris()['target']
target[:50] = 0
target[51:100] = 1
target[100:] = 2
df = pd.DataFrame(dataset)
df = df.rename( columns={0: "SL", 1: "SW", 2:"PL",3:"PW"})


# In[130]:


df = df.assign(Target=pd.Series(target))


# In[131]:



def class_aggregate(dataset):
    temp = []
    for i in range(len(dataset)):
        temp.append(dataset[i][-1])
    temp = np.unique(np.array(temp))
    return temp

cls = class_aggregate(np.array(df))

def val_replace(cls,dataset):
    for i in range(len(cls)):
        dataset = dataset.replace(cls[i],i)
    return dataset

df = val_replace(cls,df)

## Iris Dataset Classification
def gini(splits,classes):
    total_rows = 0
    final = 0
    for i in range(len(splits)):
        total_rows+=len(splits[i])
    final = 0
    for split in splits:
        if len(split)==0:
            continue
        gscore = 0
        for cls in classes:
            p = [row[-1] for row in split].count(cls)/len(split)
            gscore+=p*p
        final+= (1-gscore)*(len(split)/total_rows)
    return final
## This needs to be done very carefully
## Binary Splits
def divide_data(dataset,feature,threshold):
    new_data1 = []
    new_data2 = []
    for elem in dataset:
        if elem[feature]<threshold:
            new_data1.append(elem)
        else:
            new_data2.append(elem)
    return new_data1, new_data2

def best_split(dataset,classes):
    best_so_far = np.inf
    best_splits = 0
    feature = np.inf
    value_split = 1000
    for i in range(len(dataset[0])-1):
        for elem in dataset:
            splits = divide_data(dataset,i,elem[i])
            gscore = gini(splits,classes)
            if gscore<best_so_far:
                best_so_far = gscore
                best_splits = splits
                feature = i
                value_split = elem[i]
    return {'split':best_splits,'Feature':feature,'Value':value_split}


def leaf_node(split):
    cls = [elem[-1] for elem in split]
    # Take Majority Vote
    count = {}
    for i in cls:
        if i not in count:
            count[i]=1
        else:
            count[i]+=1
    return max(count, key = count.get)

def partition(node, maxdepth, minsize, depth):
    ## Each partition from above gives a node in essence.
    lchild, rchild = node['split']
    if not lchild or not rchild:
        node['left']= leaf_node(lchild+rchild)
        node['right'] = leaf_node(lchild+rchild)
        return
    if depth>maxdepth:
        node['left'] = leaf_node(lchild)
        node['right'] = leaf_node(rchild)
        return
    if len(lchild)<=minsize:
        node['left'] = leaf_node(lchild)
    else:
        node['left'] = best_split(lchild,cls)
        partition(node['left'],maxdepth,minsize, depth+1)
    if len(rchild)<=minsize:
        node['right'] = leaf_node(rchild)
    else:
        node['right'] = best_split(rchild,cls)
        partition(node['right'],maxdepth,minsize,depth+1)
        

def tree_iris(dataset, maxdepth, minsize):
    root = best_split(dataset,cls)
    partition(root,maxdepth,minsize,1)
    ## Printing the made Decision Tree
    return root

def predict(node, row):
    if row[node['Feature']] < node['Value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def class_aggregate(dataset):
    temp = []
    for i in range(len(dataset)):
        temp.append(dataset[i][-1])
    temp = np.unique(np.array(temp))
    return temp

cls = class_aggregate(np.array(df))

def val_replace(cls,dataset):
    for i in range(len(cls)):
        dataset = dataset.replace(cls[i],i)
    return dataset

df = val_replace(cls,df)

def accurate(test,tree):
    temp = 0
    for i in test:
        val = predict(tree,i)
        if i[-1]==val:
            temp+=1
    return temp/len(test)*100


# We actualy have 4 features for the IRIS Dataset. We will build decision tree using only two of them. This is because of the general heuristic that is used of choosing $\sqrt n$ features for each tree, where $n$ is the number of features, so that we avoid high correlation among the built trees. Since $4\choose2$ = 6, we'll build 6 such decision trees for this particular case. 
# We'll build a random forest this way.

# In[148]:


def random_forest(df,number=-1):
    lstcol = df.columns.tolist()
    lstcol.remove('Target')
    possible = []
    n_feature = int(np.sqrt(len(lstcol)))
    if number==-1:
        possible = []
        for i in combinations(lstcol,n_feature):
            possible.append(i)
        tree_avg_accuracy = []
        for i in possible:
            index = []
            for j in range(len(i)):
                index.append(i[j])
            index.append('Target')
            ndf = df[index]
            datas = np.array(ndf)
            train, test = train_test_split(datas, test_size = 0.3,random_state =42)
            tree = tree_iris(train,4,1)
            tree_avg_accuracy.append(accurate(test,tree))
    else:
        tree_avg_accuracy = []
        for n in range(number//len(lstcol)):
            for i in combinations(lstcol,n_feature):
                possible.append(i)
            for i in possible:
                index = []
                for j in range(len(i)):
                    index.append(i[j])
                index.append('Target')
                ndf = df[index]
                datas = np.array(ndf)
                train, test = train_test_split(datas, test_size = 0.3,random_state =42)
                tree = tree_iris(train,4,1)
                tree_avg_accuracy.append(accurate(test,tree))
    return  np.array(tree_avg_accuracy).mean()


# In[158]:


output = mp.Queue()
def random_forest_parallel(df,number = -1):
    lstcol = df.columns.tolist()
    lstcol.remove('Target')
    n_feature = int(np.sqrt(len(lstcol)))
    if number==-1:
        possible = []
        for i in combinations(lstcol,n_feature):
            possible.append(i)
        tree_avg_accuracy = []
        for i in possible:
            index = []
            for j in range(len(i)):
                index.append(i[j])
            index.append('Target')
            ndf = df[index]
            datas = np.array(ndf)
            train, test = train_test_split(datas, test_size = 0.3,random_state =42)
            tree = tree_iris(train,4,1)
            tree_avg_accuracy.append(accurate(test,tree))
    else:
        tree_avg_accuracy = []
        mp_ = []
        possible = []
        for n in range(number//len(lstcol)):
            for i in combinations(lstcol,n_feature):
                possible.append(i)
            for i in possible:
                index = []
                for j in range(len(i)):
                    index.append(i[j])
                index.append('Target')
                mp_.append(mp.Process(target=tree_perf,args=(index,df)))
        for p in mp_:
            p.start()
        for p in mp_:
            p.join()
        results = [output.get() for p in mp_]
    print(np.array(results).mean())


# In[159]:


def tree_perf(index,df):
    tree_avg_accuracy = []
    ndf = df[index]
    datas = np.array(ndf)
    train, test = train_test_split(datas, test_size = 0.3,random_state =42)
    tree = tree_iris(train,4,1)
    tree_avg_accuracy.append(accurate(test,tree))
    output.put( np.array(tree_avg_accuracy).mean())


# In[161]:


random_forest_parallel(df,100)

