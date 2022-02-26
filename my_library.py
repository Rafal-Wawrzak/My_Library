
# -*- coding: utf-8 -*-
'''
This may be editied so please be adviesed of that. if you are relying on these to do something, then it stops working
don't be surprised 
'''
#pandas
import pandas as pd

#Visual Exploration
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def LibraryHelp(function   =  None):
    if function != None:
        if function.lower() == 'dataframevaluecounts':
            print('''
This functions prints out the value counts of each 
column in a DataFrame
            ''')
        else:
            print('Help is WIP')
    else:
        print('''
        dataFrameValueCounts(dataFrame)
        
        pams(getMe)
        
        startUpDatacleaning(dataFrame)
        
            printNull(dataFrame)
            
            grouopByDataFrame(dataFrame, printOut = True)
            
                groupByCount(dataFrame, columnName =None , printOut = True)
                
            minMaxMeanDF(dataFrame)
            
            sumColumn(dataFrame, columnName)
            
            evalute(y_true, y_pred, class_report=True, matrix=True, color ='Reds')
            
            barGroupCount(dataFrame, mainColumn, secoundColumn= None,plot = 'bar',
                          titleFontSize = 25, tickSize = 10, labelSize = 25,
                          figureSize = (12, 5),y_rot = 0,  x_rot = 0,gridAxis = 'y')
                          
            plotGridData(group,Column, Z = (3,4), title_Text = '', size=(20, 10))
        ''')

def dataFrameValueCounts(dataFrame):
    for i in dataFrame:
        print(f'------ {i} ------')
        print('\n\n')
        print(dataFrame[i].value_counts())
        print('-------------------')


def pams(getMe):
    try:
        pam = getMe.get_params(deep=False)
        for i in pam:
            print(i,'   :   ', pam[i])
    except AttributeError:
        print("This Object has no attribute to Paramaters")

def startUpDatacleaning(dataFrame):
    print('----- Values Types -----')    
    dataFrame.info()
    print('\n\n\n')
    print('----- Null Values -----')
    printNull(dataFrame)
    print('\n\n\n')
    print('----- Group By Data -----')
    grouopByDataFrame(dataFrame)
    print('\n\n\n')
    print('----- Min Max, and Mean -----')
    minMaxMeanDF(dataFrame)

def printNull(dataFrame):
    '''
    prints the Nulls in a Data Set
    '''

    print(dataFrame.isnull().sum())

def grouopByDataFrame(dataFrame, printOut = True):
    '''
    This will Print out The DataFrame by Group count, split into Columns
    '''
    
    for i in dataFrame:
        groupByCount(dataFrame, i, printOut)

def groupByCount(dataFrame, columnName =None , printOut = True):
    '''
    Prints out the count of a Column from a dataset  if printOut is ture

    Will Return Value if printOut is False
    '''
    
    if (printOut == True):
        print((dataFrame.groupby(by = dataFrame[columnName]).count()))
    else:
        return (dataFrame.groupby(by = dataFrame[columnName]).count())

def minMaxMeanDF(dataFrame):
    
    for i in dataFrame:
        if dataFrame.dtypes[i] == 'int64':
            minMaxMean(dataFrame, i, printOut=True)
        elif dataFrame.dtypes[i] == 'float64':
            minMaxMean(dataFrame, i, printOut=True)

def minMaxMean(dataFrame, columnName, printOut = False):
    '''
    Returns an array with [ColumnName, Min_Value, Max_Value, Mean_Value]
    '''
    
    arrayOut = []
    arrayOut.append(columnName)
    arrayOut.append(dataFrame[columnName].min())
    arrayOut.append(dataFrame[columnName].max())
    arrayOut.append(dataFrame[columnName].mean())
    if printOut == True:
        print(arrayOut)
    elif printOut == False:
        return arrayOut

def sumColumn(dataFrame, columnName):
    '''
    This will print out the Nulls in a dataFrame, columnName, and the sums of those
    '''
    
    print("Is Null     :",dataFrame[columnName].isnull().sum())
    print("Is Not Null :",dataFrame[columnName].count())

def rowsColums(dataFrame):
    '''
    This will print out the rows and colums

    Example out Put:
    > Rows    : 10
    > Columns : 2 
    '''
    
    print("Rows    : ",len(dataFrame))
    print("Columns : ",len(dataFrame.columns))

def  evalute(y_true, y_pred, class_report=True, matrix=True, color ='Reds'):
    from sklearn.metrics import classification_report, confusion_matrix
    
    if class_report ==  True:
        report = classification_report(y_true, y_pred)
        print(report)
    if matrix == True:
        confuse = confusion_matrix(y_true, y_pred)
        sns.heatmap(confuse,
                    cmap = color,
                    annot= True,
                    fmt='g'
                    )

def barGroupCount(dataFrame, mainColumn, secoundColumn= None,plot = 'bar',titleFontSize = 25,  tickSize = 10, labelSize = 25, figureSize = (12, 5),y_rot = 0,  x_rot = 0,gridAxis = 'y'):
    if secoundColumn == None:
        for i in dataFrame:
            plt.figure(figsize = figureSize)
            plt.legend()
            plt.title(f'This graph shows  the count by {i}',
                      fontsize = titleFontSize)
            plt.ylabel('Count',
                       fontsize = labelSize)
            plt.yticks(fontsize=tickSize,
                       rotation = y_rot)
            plt.xlabel(i,
                       fontsize = labelSize)
            plt.xticks(fontsize=tickSize,
                       rotation = x_rot)
            plt.grid(axis = gridAxis)
            dataFrame.groupby(i)[i].count().plot(kind = plot)
            plt.show()
            print(i)    
    else:
        dataFrame.groupby(mainColumn).count().plot(kind = plot)

def plotMake(dataFrame, mainColumn, secoundColumn= None,plot = 'bar',titleFontSize = 25,  tickSize = 10, labelSize = 25, figureSize = (12, 5),y_rot = 0,  x_rot = 0,gridAxis = 'y'):
        if secoundColumn== None:
            secoundColumn = mainColumn
        plt.figure(figsize = figureSize)
        plt.legend()
        plt.title(f'This graph shows  the count by {mainColumn}',
                    fontsize = titleFontSize)
        plt.ylabel('Count',
                    fontsize = labelSize)
        plt.yticks(fontsize=tickSize,
                    rotation = y_rot)
        plt.xlabel(mainColumn,
                    fontsize = labelSize)
        plt.xticks(fontsize=tickSize,
                    rotation = x_rot)
        plt.grid(axis = gridAxis)
        
        dataFrame.groupby(mainColumn)[secoundColumn].count().plot(kind = plot) ###subplots=True look to ad dthis to code.
        plt.tight_layout()
        plt.show()

def plotGridData(group,Column, Z = (3,4),title_Text = '', size=(20, 10)):
    
    try:
        len(Z) # looking to see if there is 1 in len() if there is it throws an error
    except TypeError:
        raise ValueError('Z has incorect amount of values packed (expected 2) currenctly:1')
    if len(Z) != 2:# looking to see if there is more then 2 values in Z if there is it throws an error
        raise ValueError(f'Z has incorect amount of values packed (expected 2) currenctly:{len(Z)}')
    else:
        x,y = Z

        # Visualize means 
        fig, axes = plt.subplots(x, y, figsize = size)
        
        axes = axes.ravel()
        # Loop over columns and plot each in a separate figure, skip 'cluster' column
        for i, col in enumerate(group.columns[1:]):
            axes[i].bar(group[Column], group[col])
            axes[i].set_title(f'{title_Text} {col}')
            plt.tight_layout()
