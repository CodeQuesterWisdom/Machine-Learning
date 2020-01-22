import math
import sys
from math import sqrt, exp, pi
from Blackbox31 import blackbox31
from Blackbox32 import blackbox32
import matplotlib.pyplot as plt



#Extracting Train data and Test data path from Command Line Arguments
no_of_cli_arguments = len(sys.argv)
if no_of_cli_arguments!=2:
    print("Blackbox path not specified; Please check the CLI arguments")
    sys.exit(0)
args = sys.argv
train_df_path = args[1]
indice = train_df_path.index("blackbox")
output_name = "results_blackbox3{}.txt".format(train_df_path[indice+9])  #OutputFile Name


# Input Parameters
# Intializing blackbox
if train_df_path=="blackbox31":
    blackbox = blackbox31
elif train_df_path=="blackbox32":
    blackbox = blackbox32
# Train_data format
# Label :{class: [prior, [feature1_mean,feature1_variance],[feature2_mean,feature2_variance],[feature3_mean,feature3_variance],count]}
train_data_by_class= {0:[0,[0,0],[0,0],[0,0],0],1:[0,[0,0],[0,0],[0,0],0],2:[0,[0,0],[0,0],[0,0],0]}
count=[]
test_data = []
test_labels = []
test_data_predictions = []
test_accuracy=[]
results = []


# Storing Test Data and Test Labels
for i in range(200):
    feature, label = blackbox.ask()
    test_data.append(feature)
    test_labels.append(label)


# Updating Mean
def update_mean(old_mean, n, new_val):
    return (((n-1) * old_mean + new_val )/n)


# Updating Variance
def update_variance(old_variance, n , feature, previous_mean):
    if n==1: return 0
    return ( (((n-2)/(n-1))*old_variance) + (((feature - previous_mean)**2)/n) )


# Training the modeld
def train_model():

    # Get the data from blackbox incrementally
    feature, label = blackbox.ask()

    #Update total count and its coressponding class count
    total_count= train_data_by_class[0][4]+ train_data_by_class[1][4]+train_data_by_class[2][4]+1
    train_data_by_class[label][4]+=1

    # Update prior, mean, variance for current class and its corresponding features
    current_class_count = train_data_by_class[label][4]
    x_mean = update_mean(train_data_by_class[label][1][0],current_class_count,feature[0])
    x_var = update_variance(train_data_by_class[label][1][1],current_class_count,feature[0],train_data_by_class[label][1][0])
    y_mean = update_mean(train_data_by_class[label][2][0],current_class_count,feature[1])
    y_var = update_variance(train_data_by_class[label][2][1],current_class_count,feature[1],train_data_by_class[label][2][0])
    z_mean = update_mean(train_data_by_class[label][3][0],current_class_count,feature[2])
    z_var = update_variance(train_data_by_class[label][3][1],current_class_count,feature[2],train_data_by_class[label][3][0])

    prior = train_data_by_class[label][4]/total_count
    train_data_by_class[label]=[prior,[x_mean,x_var],[y_mean,y_var],[z_mean,z_var],current_class_count]

   # Update prior for all the classes as total count is changed
    for label in train_data_by_class:
        prior = train_data_by_class[label][4]/total_count
        train_data_by_class[label][0] = prior



# Calculate Guassian probability
def likelihood_guassian(feature, mean, variance):
    # Edge case when variance is Zero: replace it with epsilon
    if variance==0: variance = sys.float_info.epsilon
    exponent = math.exp(-((feature-mean)**2 / (2 * variance)))
    return (1 / (math.sqrt(2 * pi* variance))) * exponent


# Find Test Data Labels
def test_model():
    test_data_predictions = []
    for row in test_data:
        # likelihood_guassian of each class
        prob_0 = train_data_by_class[0][0] * likelihood_guassian(row[0],train_data_by_class[0][1][0],train_data_by_class[0][1][1])*likelihood_guassian(row[1],train_data_by_class[0][2][0],train_data_by_class[0][2][1]) * likelihood_guassian(row[2],train_data_by_class[0][3][0],train_data_by_class[0][3][1])
        prob_1 = train_data_by_class[1][0] * likelihood_guassian(row[0],train_data_by_class[1][1][0],train_data_by_class[1][1][1])*likelihood_guassian(row[1],train_data_by_class[1][2][0],train_data_by_class[1][2][1]) * likelihood_guassian(row[2],train_data_by_class[1][3][0],train_data_by_class[1][3][1])
        prob_2 = train_data_by_class[2][0] * likelihood_guassian(row[0],train_data_by_class[2][1][0],train_data_by_class[2][1][1])*likelihood_guassian(row[1],train_data_by_class[2][2][0],train_data_by_class[2][2][1]) * likelihood_guassian(row[2],train_data_by_class[2][3][0],train_data_by_class[2][3][1])

       # Assign to the class with highest probability
        if prob_0 >= prob_1 and prob_0>=prob_2:
            test_data_predictions.append(0)
        elif prob_1 >= prob_0 and prob_1>=prob_2:
            test_data_predictions.append(1)
        else:
            test_data_predictions.append(2)
    return test_data_predictions



# Finding Accuracy of Test Data
def accuracy(test_labels, test_data_predictions):
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i] == test_data_predictions[i]:
            correct += 1
    acc = correct / float(len(test_labels))
    return round(acc, 3)  # Rounding off accuracy to 3 decimal points



# Training the model incrementally and testing accuracy after every 10 data points
for i in range(100):
    for i in range(10):
        train_model()
    test_data_predictions = test_model()
    test_accuracy.append(accuracy(test_labels, test_data_predictions))

# Storing the count with diff of 10
for c in range(10,1010,10):
    count.append(c)

# Storing the final results i.e. in format [count, test_accuracy]
for count1, accuracy in zip(count,test_accuracy):
    temp = str(count1) + ", " + str(accuracy)
    results.append(temp)

# plt.plot(count,test_accuracy)
# plt.ylabel('Test Accuracy')
# plt.xlabel('Training Times')
# plt.show()

# Writing the results to the output file
with open(output_name,"w") as results_file:
    for item in results:
        results_file.write("%s\n" % item)
