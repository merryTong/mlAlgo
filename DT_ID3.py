import numpy as np
import pandas as pd
from math import log


# 计算熵
def entropy(data):
    '''
    function: Calculating entropy value.
    input: A list contain categorical value.
    output: Entropy value.
    entropy = - sum(p * log(p)), p is a prob value.
    '''
    # Calculating the probability distribution of list value
    probs = [data.count(i)/len(data) for i in set(data)]    
    # Calculating entropy value
    entropy = -sum([prob*log(prob, 2) for prob in probs])    
    return entropy

def split_dataframe(data, col):
    '''
    function: split pandas dataframe to sub-df based on data and column.
    input: dataframe, column name.
    output: a dict of splited dataframe.
    '''
    # unique value of column
    unique_values = data[col].unique()    
    # empty dict of dataframe
    result_dict = {elem : pd.DataFrame for elem in unique_values}    
    # split dataframe based on column value
    for key in result_dict.keys():
        result_dict[key] = data[:][data[col] == key]    
    return result_dict

def choose_best_col(df, label):    
    '''
    funtion: choose the best column based on infomation gain.
    input: datafram, label
    output: max infomation gain, best column, 
            splited dataframe dict based on best column.
    '''
    # Calculating label's entropy
    entropy_D = entropy(df[label].tolist())    
    # columns list except label
    cols = [col for col in df.columns if col not in [label]]    
    # initialize the max infomation gain, best column and best splited dict
    max_value, best_col = -np.inf, None
    max_splited = None
    # split data based on different column
    for col in cols:
        splited_set = split_dataframe(df, col)
        entropy_DA = 0
        for subset_col, subset in splited_set.items():            
            # calculating splited dataframe label's entropy
            entropy_Di = entropy(subset[label].tolist())            
            # calculating entropy of current feature
            entropy_DA += len(subset)/len(df) * entropy_Di        
        # calculating infomation gain of current feature
        info_gain = entropy_D - entropy_DA        
        if info_gain > max_value:
            max_value, best_col = info_gain, col
            max_splited = splited_set    
    return max_value, best_col, max_splited


df = pd.read_csv('data/example_data.csv')
print(df)

for idx in range(len(df.columns)):
    entr = entropy(df[df.columns[idx]].tolist())
    print(df.columns[idx], entr)

split_data = split_dataframe(df, "temp")
print(split_data)
res = choose_best_col(df, "play")
print(res)


# define a Node class
class Node:
    def __init__(self, name):
        self.name = name
        self.connections = {}    
        
    def connect(self, label, node):
        self.connections[label] = node  

class ID3Tree(object):    
    def __init__(self, data, label):
        self.columns = data.columns
        self.data = data
        self.label = label
        self.root = Node("Root")

    # print tree method
    def print_tree(self, node, tabs):
        print(tabs + node.name)        
        for connection, child_node in node.connections.items():
            print(tabs + "\t" + "(" + str(connection) + ")")
            self.print_tree(child_node, tabs + "\t\t")    
    
    def construct_tree(self):
        self.construct(self.root, "root", self.data, self.columns)
    
    # construct tree
    def construct(self, parent_node, parent_connection_label, input_data, columns):
        max_value, best_col, max_splited = choose_best_col(input_data[columns], self.label)        
        if not best_col:
            node = Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label, node)
            return 
        node = Node(best_col)
        parent_node.connect(parent_connection_label, node)

        new_columns = [col for col in columns if col != best_col]
        # Recursively constructing decision trees
        for splited_value, splited_data in max_splited.items():
            self.construct(node, splited_value, splited_data, new_columns)

tree = ID3Tree(df, "play")
tree.construct_tree()
tree.print_tree(tree.root, "")
