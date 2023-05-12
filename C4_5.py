import numpy as np
import pandas as pd
import math
class Node:
    def __init__(self, isLeaf, label, threshold, gainRatio=None):
        self.label = label
        self.threshold = threshold
        self.category = []
        self.isLeaf = isLeaf
        self.children = []
        self.gainRatio = gainRatio

class C45:
    def __init__(self, limitPartition=2):
        self.tree = None
        self.predict_label = None
        self.limitPartition = limitPartition

    def fit(self, data , labels):
        self.data = data
        self.data['label'] = labels
        self.columns = list(self.data.columns)
        self.data = self.data.values
        self.tree = self.createNode(self.data, self.columns[:-1])
        
    
    def createNode(self, data, columns):
        allSameClass = self.checkAllSameClass(data[:,-1])

        # if len(data) == 0:
        #     return Node(True, None, None)
        
        if allSameClass is not None:
            return Node(True, allSameClass, None)
        
        elif columns is None or len(columns) == 1:
            majorLabel = self.majorityLabel(data[:, -1])
            return Node(True, majorLabel, None)

        #  Chọn thuộc tính tốt nhất để chia
        else :
            bestAttribute, bestThreshold, bestPartitions, gainRatio = self.findBestAttribute(data, columns)

            if len(bestPartitions) == 1:
                return Node(True, self.majorityLabel(data[:, -1]), None)
            
            node = Node(False, bestAttribute, bestThreshold, gainRatio)
            index = columns.index(bestAttribute)
            if bestThreshold is None:
                for partition in bestPartitions:
                    node.category.append(partition[0][index])
            remainingColumns = columns[:]
            remainingColumns.remove(bestAttribute)

            for partition in bestPartitions:
                node.children.append(self.createNode(np.delete(partition,index,1), remainingColumns))
            # node.children = [self.createNode(np.delete(partition,index,1), columns) for partition in bestPartitions]
            return node

    def findBestAttribute(self, data, columns):
        splitted = []
        maxGainRatio = -1*float('inf')
        bestAttribute = -1
        threshold = None
        bestThreshold = None

        for attribute in columns:
            # labels = labels[data.index]
            index_of_attribute = columns.index(attribute)
            
            if self.isAttrDiscrete(data[:, index_of_attribute]):
                unique_values = np.unique(data[:, index_of_attribute])
                partitions = [[] for _ in range(len(unique_values))]

                for row in data:
                    partitions[unique_values.index(row[index_of_attribute])].append(row)
                
                gr = self.gainRatio(data, partitions)
                if gr >= maxGainRatio:
                    splitted = partitions
                    maxGainRatio = gr
                    bestAttribute = attribute
                    best_threshold = None 
            
            else:     
                data = data[data[:, index_of_attribute].argsort(kind='stable')]
                threshold = []
                threshold_value=[]
                sameLabelPrevious = None
                sameLabelCurrent = data[0][-1]

                # Chọn số ngưỡng để chia
                for i in range(len(data) - 1):
                    #  Nếu 2 giá trị liền kề của thuộc tính khác nhau thì chia ngưỡng
                    if data[i][index_of_attribute] != data[i+1][index_of_attribute]:

                        # Nếu cả 2 giá trị đều có số nhãn giống nhau
                        if sameLabelPrevious is not None or sameLabelCurrent is not None and sameLabelPrevious == sameLabelCurrent:
                            threshold.pop()
                            threshold_value.pop()

                        # Thêm vào giá trị ngưỡng 
                        threshold.append(i+1)
                        threshold_value.append((data[i][index_of_attribute] + data[i+1][index_of_attribute]) / 2)
                        sameLabelPrevious = sameLabelCurrent
                        sameLabelCurrent = data[i+1][-1]

                    elif data[i][-1] != data[i+1][-1]:
                        sameLabelCurrent = None

                lenThreshold = len(threshold)
                partition = 2 
                limitPartition = self.limitPartition
                if lenThreshold == 0:
                    e = self.gainRatio(data, [data])
                    if e >= maxGainRatio:
                        splitted = [data]
                        maxGainRatio = e
                        bestAttribute = attribute
                        bestThreshold = None

                while(partition <= limitPartition and lenThreshold >= partition-1):
                    indexSplits = self.fillAllTheWaySplit(lenThreshold, partition)
                    # Tìm được số cách chia
                    for indexSplit in indexSplits:
                        start_index = 0
                        partitions = []
                        best_threshold = []
                        for index in indexSplit:
                            partitions.append(data[start_index:threshold[index]])
                            best_threshold.append(threshold_value[index])
                            start_index = threshold[index]
                        partitions.append(data[start_index:])


                        e = self.gainRatio(data, partitions)
                        if e >= maxGainRatio:
                            splitted = partitions
                            maxGainRatio = e
                            bestAttribute = attribute  
                            bestThreshold = best_threshold 
                    partition += 1     

        return (bestAttribute, bestThreshold, splitted, maxGainRatio)
                 

    def fillAllTheWaySplit(self, lenThreshold, partition):

        def Try(currIndexOfThreshold, currPartition):
            if currIndexOfThreshold == lenThreshold:
                return
            
            if currPartition == partition:
                element.append(currIndexOfThreshold)
                ans.append(element.copy())
                element.pop()
                Try(currIndexOfThreshold + 1, currPartition)

            if currPartition < partition:
                element.append(currIndexOfThreshold)
                Try(currIndexOfThreshold + 1, currPartition + 1)
                element.pop()
                Try(currIndexOfThreshold + 1, currPartition)
                
        ans = []
        element = []
        Try(0, 2)
        return ans


    def isAttrDiscrete(self, attrValues):
        type = attrValues.dtype
        return (type!= 'int' and type != 'float')


    def gainRatio(self, data, partitions):
        gain = self.gainSplit(data, partitions)
        split_info = self.splitInfo(data, partitions)
        if split_info == 0:
            gain_ratio = gain
        else:
            gain_ratio = gain / split_info
        return gain_ratio


    def gainSplit(self, data, partitions):
        N = len(data)
        impurity_before = self.entropy(data)
        impurity_after = 0

        for partition in partitions:
            impurity_after += len(partition) / N * self.entropy(partition)
        total_gain = impurity_before - impurity_after
        return total_gain

    
    def entropy(self, data):
        N = len(data)
        if N == 0:
            return 0
        _, counts = np.unique(data[:, -1], return_counts=True)
        proportions = counts / N
        proportions[proportions == 0] = 1
        entropy = 0
        for pi in proportions:
            entropy += pi * self.log(pi)
        return entropy * -1
    

    def splitInfo(self, data, partitions):
        N = len(data)
        weights = [len(partition) / N for partition in partitions]
        split_info = 0
        for weight in weights:
            split_info += weight * self.log(weight)
        return split_info * -1


    def log(self, x):
        if x==0:
            return 0
        else:
            return math.log(x,2)

    def checkAllSameClass(self, labels):
        if labels.dtype == 'float':
            col = labels.astype(int)
            if np.isclose(col, labels).all():
                labels = col
        if len(np.unique(labels)) == 1:
            return labels[0]
        return None
    

    def majorityLabel(self, labels):
        if labels.dtype == 'float':
            col = labels.astype(int)
            if np.isclose(col, labels).all():
                labels = col
        counts = np.bincount(labels)
        return np.argmax(counts)


    def printTree(self):
        self.printNode(self.tree)


    def printNode(self, node, space=""):
        if not node.isLeaf:
            if node.threshold is None:
                # Categorical
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        print(space + "{} = {} : {}".format(node.label, node.category[index] , child.label))
                    else:
                        print(space + "{} = {} :".format(node.label, node.category[index]))
                        self.printNode(child, space + "     ")
            else:
                for index, child in enumerate(node.children):
                    if index == 0:
                        if child.isLeaf:
                            print(space + "{} <= {} : {}".format(node.label, node.threshold[index] , child.label))
                        else:
                            print(space + "{} <= {} :".format(node.label, node.threshold[index]))
                            self.printNode(child, space + "     ")

                    elif index == len(node.children) - 1:
                        if child.isLeaf:
                            print(space + "{} > {} : {}".format(node.label, node.threshold[index - 1] , child.label))
                        else:
                            print(space + "{} > {} :".format(node.label, node.threshold[index - 1]))
                            self.printNode(child, space + "     ")

                    else:
                        if child.isLeaf:
                            print(space + "{} > {} : {}".format(node.label, node.threshold[index - 1] , child.label))
                        else:
                            print(space + "{} < {} <= {} :".format(node.threshold[index - 1], node.label, node.threshold[index]))
                            self.printNode(child, space + "     ")


    def predict(self, data):
        results = []
        for id, row in data.iterrows():
            self.predict_label = None
            self.predictRow(self.tree, row)
            results.append(self.predict_label)
        return results
    

    def predictRow(self, node, row):
        if not node.isLeaf:
            if node.threshold is None:
                # Categorical
                for index, child in enumerate(node.children):
                    if row[node.label] == node.category[index]:
                        if child.isLeaf:
                            self.predict_label = child.label
                        else:
                            self.predictRow(child, row)
            else:
                # Dữ liệu liên tục
                for index, child in enumerate(node.children):
                    if index == 0:
                        if row[node.label] <= node.threshold[index]:
                            if child.isLeaf:
                                self.predict_label = child.label
                            else:
                                self.predictRow(child, row)

                    elif index == len(node.children) - 1:
                        if row[node.label] > node.threshold[index - 1]:
                            if child.isLeaf:
                                self.predict_label = child.label
                            else:
                                self.predictRow(child, row)

                    else:
                        if row[node.label] > node.threshold[index - 1] and row[node.label] <= node.threshold[index]:
                            if child.isLeaf:
                                self.predict_label = child.label
                            else:
                                self.predictRow(child, row)
