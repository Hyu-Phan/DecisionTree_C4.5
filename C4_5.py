import numpy as np
import pandas as pd

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

    def fit(self, data, labels):
        self.tree = self.createNode(data, labels)
        
    
    def createNode(self, data, labels):
        allSameClass = self.checkAllSameClass(labels)

        if len(data) == 0:
            return Node(True, None, None)
        
        if allSameClass is not None:
            return Node(True, allSameClass, None)
        
        elif len(data) == 0:
            majorLabel = self.majorityLabel(labels)
            return Node(True, majorLabel, None)
        
        else :
            bestAttribute, bestThreshold, bestPartitions, labelsOfPartition, gainRatio = self.findBestAttribute(data, labels)
            
            node = Node(False, bestAttribute, bestThreshold, gainRatio)

            if bestThreshold is None:
                for partition in bestPartitions:
                    node.category.append(partition.iloc[0][bestAttribute])
            
            node.children = [self.createNode(partition.drop([bestAttribute], axis=1), labels) for partition, labels in zip(bestPartitions, labelsOfPartition)]
            return node

    def findBestAttribute(self, data, labels):
        splitted = []
        maxGainRatio = -1*float('inf')
        bestAttribute = -1
        threshold = None
        label_splitted = []
        best_threshold = None

        for attribute in data.columns:
            data = data.sort_values(attribute)
            labels = labels[data.index]
            if self.isAttrDiscrete(data[attribute]):
                partitions = []
                labelsOfPartition = []

                start_index = 0
                for index in range(len(data) - 1):
                    if data.iloc[index][attribute] != data.iloc[index + 1][attribute]:
                        partitions.append(data.iloc[start_index:index+1])
                        labelsOfPartition.append(labels.iloc[start_index:index+1])
                        start_index = index+1
                partitions.append(data.iloc[start_index:])
                labelsOfPartition.append(labels.iloc[start_index:])
                # partitions = data.groupby(attribute)
                # labelsOfPartition = labels.groupby(data[attribute])
                # print(labelsOfPartition.get_group('Overcast'))

                e = self.gainRatio(labels, labelsOfPartition)
                if e >= maxGainRatio:
                    splitted = partitions
                    label_splitted = labelsOfPartition
                    maxGainRatio = e
                    bestAttribute = attribute
                    best_threshold = None    
            
            else:     
                threshold = []
                threshold_value=[]
                for i in range(len(data) - 1):
                    if data.iloc[i][attribute] != data.iloc[i + 1][attribute]:
                        threshold_value.append((data.iloc[i][attribute] + data.iloc[i+1][attribute]) / 2)
                        threshold.append(i+1)

                lenThreshold = len(threshold)
                partition = 2 
                limitPartition = self.limitPartition

                while(partition <= limitPartition):
                    indexSplits = self.fillAllTheWaySplit(lenThreshold, partition)
                    for indexSplit in indexSplits:
                        start_index = 0
                        partitions = []
                        labelsOfPartition = []
                        best_threshold = []
                        for index in indexSplit:
                            partitions.append(data.iloc[start_index:threshold[index]])
                            labelsOfPartition.append(labels.iloc[start_index:threshold[index]])
                            best_threshold.append(threshold_value[index])
                            start_index = threshold[index]
                        partitions.append(data.iloc[start_index:]) 
                        labelsOfPartition.append(labels.iloc[start_index:])

                        e = self.gainRatio(labels, labelsOfPartition)
                        if e >= maxGainRatio:
                            splitted = partitions
                            label_splitted = labelsOfPartition
                            maxGainRatio = e
                            bestAttribute = attribute   
                    partition += 1     

        return (bestAttribute, best_threshold, splitted ,label_splitted, maxGainRatio)
                 

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
        return (type!= 'int64' and type != 'float')


    def gainRatio(self, data, partitions):
        gain = self.gainSplit(data, partitions)
        split_info = self.splitInfo(data, partitions)
        if split_info == 0:
            gain_ratio = gain
        else:
            gain_ratio = gain / split_info
        return gain_ratio


    def gainSplit(self, labels, labelsOfPartition):
        N = len(labels)
        impurity_before = self.entropy(labels)
        impurity_after = 0

        for partition in labelsOfPartition:
            impurity_after += len(partition) / N * self.entropy(partition)
        total_gain = impurity_before - impurity_after
        return total_gain

    
    def entropy(self, labels):
        N = len(labels)
        if N == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        proportions = counts / N
        proportions[proportions == 0] = 1
        return -np.sum(proportions * np.log2(proportions))
    

    def splitInfo(self, data, partitions):
        N = len(data)
        weights = np.array([len(partition) / N for partition in partitions])
        split_info = -np.sum(weights * np.log2(weights))
        return split_info


    def checkAllSameClass(self, labels):
        if len(np.unique(labels)) == 1:
            return labels.iloc[0]
        return None
    

    def majorityLabel(self, labels):
        _, counts = np.unique(labels[:, -1], return_counts=True)
        return np.take(labels[:, -1], np.argmax(counts))


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
                            print(space + "{} < {} <= {} : {}".format(node.threshold[index - 1], node.label, node.threshold[index] , child.label))
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
