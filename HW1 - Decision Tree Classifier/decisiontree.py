import numpy as np
import pandas as pd
import collections
import csv
import sys
from pprint import pprint

#Extracting Train data and Test data path from Command Line Arguments
no_of_cli_arguments = len(sys.argv)
if no_of_cli_arguments!=3:
    print("Train data and Test data paths are not specified; Please check the CLI arguments")
    sys.exit(0)
args = sys.argv
train_df_path = args[1]
test_df_path = args[2]
indice = train_df_path.index("blackbox")
output_name = "blackbox1{}_predictions.csv".format(train_df_path[indice+9])  #OutputFile


#Loading train data
df = pd.read_csv(r"{}".format(train_df_path),header=None)
train_df = df  #DataFrame
train_data =df.values #Numpy array - 10x faster than DataFrame


#If all the labeles are same for the given data, stop spitting further
def check_if_unique (data):
    labels = data[:,-1]
    if len(np.unique(labels))==1:
        return True
    else:
        return False


# If decided to stop splitting further,then return the
# label with highest frequency in the given data
def stop_recursion(data):
    labels = data[:,-1]
    label_dict = collections.Counter(list(labels))
    keys= list(label_dict.keys())
    val = list(label_dict.values())
    return keys[val.index(max(val))]


# Find all possible splits by traversing through each value of every feauture
# Split point of feautue A with point i is val(i) + val(i-1) / 2
# i.e. avergae of previous and next points within a column
def find_all_splits(data):
    _,n_col = data.shape
    all_splits={}
    for i in range(n_col-1): #n_col-1 skips last column which is labels
        all_splits[i]=[]
        val = data[:,i]
        unique_val = list(np.unique(val))  #finding out unique values in a feature, as split value for repeated values will be same
        if len(unique_val) > 1: #Handling the edge case if all the samples are same
            for j in range(1,len(unique_val)):
                all_splits[i].append((unique_val[j]+unique_val[j-1])/2)
    return all_splits


# For the given data, feauture and split value, split given data into
# 2 parts based on splitvalue
def split_data(data, col, split_val):
    data_new = data[:,col]
    left_child = data[data_new <= split_val]
    right_child = data[data_new> split_val]
    return left_child,right_child


# Finding entropy i.e. sigma of -plogp
# Where ‘p’ is simply the frequentist probability of an element/class ‘i’ in our data
def find_entropy(data):
    labels = data[:,-1]
    unique_count= collections.Counter(labels)
    values= list(unique_count.values())
    total_val= sum(values)
    probs={}
    for val in values: probs[val] = val/total_val
    entropy=0
    for prob in probs:
        entropy =  entropy + (probs[prob] * -np.log2(probs[prob]))
    return entropy


# Finding overall entropy between 2 data sets
# To determine to which group the sample belongs to
def find_overall_entropy(left_child, right_child):
    total_len = len(left_child) + len(right_child)
    prob_left = len(left_child) / total_len
    prob_right = len(right_child) / total_len
    return ((prob_left * find_entropy(left_child))+(prob_right * find_entropy(right_child)))


# From all_possible_splits, traverse through each split,
# find out overall entropy for each split and return the split which gives
# least entropy i.e. a split which gives the purest possible data.
def find_best_node(data, all_splits):
    entropy = 5.0
    for col in all_splits:
        for val in all_splits[col]:
            left_child, right_child = split_data(data,col,val)
            temp_entropy = find_overall_entropy(left_child, right_child)
            if temp_entropy < entropy :
                entropy = temp_entropy
                best_col, best_split_val = col,val
    return best_col,best_split_val,entropy



# Core logic: Building the decision tree
def build_decision_tree(data,depth=0):

    #Stop splitting if data contains the same labels and
    #if depth crosses the given value or if there are very few sample in dataset
    if check_if_unique(data) or depth>12 or len(data)<3:
        return stop_recursion(data)
    else:
        #Find all possible splits
        all_splits= find_all_splits(data)
        #If all the samples are same, when split, all the data will lie on
        #either left/right part then stop splitting further
        if len(all_splits)==0:
            return stop_recursion(data)

        #Find best split from the all possible splits
        best_node, best_split_val, entropy = find_best_node(data,all_splits)

        # if entropy<0.01:
        #     return stop_recursion(data)

        #Split data into 2 parts based on the best split from above step
        left_child, right_child = split_data(data,best_node,best_split_val)

        ans_tree={}
        #Once the best split feature along with its split value is found, add it to final decision tree
        key = str(best_node)+" , "+str(best_split_val)
        ans_tree[key]=[]

        #Repeat the same i.e. recursively call the same method for left part of split and right part of split
        key_left_child= build_decision_tree(left_child,depth+1)
        key_right_child = build_decision_tree(right_child,depth+1)

        #Split only one child, if left child is equal to right child
        if key_left_child == key_right_child: ans_tree = key_left_child
        else:
            ans_tree[key].append(key_left_child)
            ans_tree[key].append(key_right_child)
        return ans_tree

ans_decision_tree = build_decision_tree(train_data) #Calling decision tree method
pprint(ans_decision_tree)



# Finding out the class label for the given sample of test data from the
# decision tree built from above steps
def classify(ans_decision_tree, test_data):
    col_name, op, val = list(ans_decision_tree.keys())[0].split()
    if test_data[int(col_name)] <= float(val):
        ans = ans_decision_tree[list(ans_decision_tree.keys())[0]][0]
    else:
        ans = ans_decision_tree[list(ans_decision_tree.keys())[0]][1]

    if not isinstance(ans, dict):
        return ans
    else:
        return classify(ans,test_data)


# Loading test data as DataFrame
test_df = pd.read_csv(r"{}".format(test_df_path),header=None)
test_data =test_df.values #Converting to numpy array

# Classify all the samples and output it into a csv file
def accuracy(test_df, ans_decision_tree):
    actual_test_labels = [i[-1] for i in test_df.values]
    predicted_test_labels = []
    l = len(test_df)
    for i in range(l):
        predicted_test_labels.append(classify(ans_decision_tree,test_df.iloc[i]))
    with open(output_name, 'w',newline='') as f:
        writer = csv.writer(f)
        for val in predicted_test_labels:
            writer.writerow([val])
    correct = 0
    for i in range(l):
        if actual_test_labels[i] == predicted_test_labels[i]: correct+=1
    return (correct/l)*100


# Accuracy
print(accuracy(test_df,ans_decision_tree))
#accuracy(test_df,ans_decision_tree)
