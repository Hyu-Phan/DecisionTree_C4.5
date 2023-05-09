import numpy as np
import pandas as pd

class Node:
    def __init__(self, isLeaf, label, threshold):
        self.label = label
        self.threshold = threshold
        self.category = []
        self.isLeaf = isLeaf
        self.children = []


class C45:
    def __init__(self):
        self.tree = None
        self.predict_label = None


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
            bestAttribute, bestThreshold, bestPartitions = self.findBestAttribute(data, labels)
            
            return 

    def findBestAttribute(self, data, labels):
        splitted = []
        maxGainRatio = -1*float('inf')
        bestAttribute = -1
        threshold = None

        for attribute in data.columns:
            data = data.sort_values(attribute)
            if self.isAttrDiscrete(attribute):
                partitions = []
                start_index = 0
                for index in range(len(data) - 1):
                    if data[attribute][index] != data[attribute][index + 1]:
                        partitions.append(data.iloc[start_index:index])
                partitions.append(data.iloc[start_index:])
                e = self.gainRatio(partitions)
                if e >= maxGainRatio:
                    splitted = partitions
                    maxGainRatio = e
                    bestAttribute = attribute
                    best_threshold = threshold    
            else:
                threshold = []
                prev = data[attribute][0]
                for i in range(len(data) - 1):
                    if data[attribute][i] != data[attribute][i + 1] and labels[i] != labels[i + 1]:
                        threshold.append((data[attribute][i + 1] + prev) / 2)
                        prev = data[attribute][i + 1]
                lenThreshold = len(threshold)
                partition = 2 
                limitPartition = 4
                while(partition <= limitPartition):
                    indexSplits = self.fillAllTheWaySplit(lenThreshold, partition)
                    for indexSplit in indexSplits:
                        start_index = 0
                        partitions = []
                        for index in indexSplit:
                            partitions.append(data.iloc[start_index:index])
                            start_index = index
                        partitions.append(data[start_index:])    
                        e = self.gainRatio(partitions)
                        if e >= maxGainRatio:
                            splitted = partitions
                            maxGainRatio = e
                            bestAttribute = attribute
                            best_threshold = threshold       
                    partition += 1     
        return (bestAttribute,best_threshold,splitted)
                 

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
        return (attrValues.dtype != 'int64' and attrValues.dtype != 'float')


    def gainRatio(self, data, partitions):
        gain = self.gainSplit(data, partitions)
        split_info = self.splitInfo(data, partitions)
        if split_info == 0:
            gain_ratio = gain
        gain_ratio = gain / split_info
        return gain_ratio


    def gainSplit(self, data, partitions):
        N = len(data)
        impurity_before = self.entropy(data[:, -1])
        impurity_after = 0
        weights = np.array([len(partition) / N for partition in partitions])
        for i in range(len(partitions)):
            impurity_after += weights[i] * self.entropy(partitions[i][:, -1])

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
            return labels[0]
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
                        print(space + "Attribute {} = {} : {}".format(node.label, node.category[index] , child.label))
                    else:
                        print(space + "Attribute {} = {} :".format(node.label, node.category[index]))
                        self.printNode(child, space + "  ")
            else:
                for index, child in enumerate(node.children):
                    if index == 0:
                        if child.isLeaf:
                            print(space + "Attribute {} <= {} : {}".format(node.label, node.threshold[index] , child.label))
                        else:
                            print(space + "Attribute {} <= {} :".format(node.label, node.threshold[index]))
                            self.printNode(child, space + "  ")

                    elif index == len(node.children) - 1:
                        if child.isLeaf:
                            print(space + "Attribute {} > {} : {}".format(node.label, node.threshold[index - 1] , child.label))
                        else:
                            print(space + "Attribute {} > {} :".format(node.label, node.threshold[index - 1]))
                            self.printNode(child, space + "  ")

                    else:
                        if child.isLeaf:
                            print(space + "{} < Attribute {} <= {} : {}".format(node.threshold[index - 1], node.label, node.threshold[index] , child.label))
                        else:
                            print(space + "{} < Attribute {} <= {} :".format(node.threshold[index - 1], node.label, node.threshold[index]))
                            self.printNode(child, space + "  ")


    def predict(self, data):
        results = []
        for row in data:
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
    
