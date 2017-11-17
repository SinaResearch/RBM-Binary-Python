import codecs
import os
from random import shuffle

def add1Backslash(text): 
    return text.replace("\\", "\\\\")

def eliminate_space(list_input):
    list_input_temp=list()
    for i in list_input:
        if not i=="":
            list_input_temp.append(i)
    return list_input_temp

def getName(command): # name of the input file.
    while(True): 
        File_name=input(command)
        try:
            if(File_name.endswith('.txt') and os.path.isfile(File_name) and os.access(File_name, os.R_OK)):
                File_name=add1Backslash(File_name)
                break
            else:
                raise NameError('File not found!')
        except:
            print("File not found! Try again.")
    return File_name

def attributeFinder(inputArray):
    max=0
    for values in inputArray:
        values=values.split(" ")
        values=eliminate_space(values)
        var=int((values[len(values)-1].split(':'))[0])
        if(max<var):
            max=var

    return max
    
def valueFinder(inputArray,start,end,pos_valuesFileName,neg_valuesFileName,trainigSet_all_filename,max):
    valueSet= [0]*max
    trainigSet_pos=codecs.open(pos_valuesFileName,'w','utf-8')
    trainigSet_neg=codecs.open(neg_valuesFileName,'w','utf-8')
    trainigSet_all=codecs.open(trainigSet_all_filename,'w','utf-8')
        
    for values in inputArray[start:end]:
        values=values.split(" ")
        values=eliminate_space(values)
                      
        for m in values[1:]:
            att=m.split(":")
            if len(att)==2:
                valueSet[int(att[0])-1]=str(att[1])
        
        if(values[0]=="+1"):
            for o in range(len(valueSet)):
                trainigSet_pos.write(str(valueSet[o]))
                trainigSet_all.write(str(valueSet[o]))
                if o<len(valueSet)-1:
                    trainigSet_pos.write(",")
                    trainigSet_all.write(",")
            trainigSet_pos.write('\n')
            trainigSet_all.write('\n')
        else:
            for o in range(len(valueSet)): 
                trainigSet_neg.write(str(valueSet[o]))
                trainigSet_all.write(str(valueSet[o]))
                if o<len(valueSet)-1:
                    trainigSet_neg.write(",")
                    trainigSet_all.write(",")
            trainigSet_neg.write('\n')  
            trainigSet_all.write('\n')
             
def vectorisation(inputArray,start,end,pos_valuesFileName,neg_valuesFileName,trainigSet_all_filename):        
        max=attributeFinder(inputArray)
        valueFinder(inputArray,start,end,pos_valuesFileName,neg_valuesFileName,trainigSet_all_filename,max)
#--------------------------------- MAIN ----------------
trainigFile=codecs.open(getName("File name: "),'r','utf-8')
testSize = input("Test size: ")
trainingSet_filename = input("Training set filename: ")
testSet_filename = input("Test set filename: ")

trainingList=(trainigFile.read()).split('\n')
trainingList=eliminate_space(trainingList)
shuffle(trainingList)

os.makedirs('Train', exist_ok=True)
os.makedirs('Test', exist_ok=True)

vectorisation(trainingList,0,len(trainingList)-int(testSize),"Train//"+'+'+trainingSet_filename,"Train//"+'-'+trainingSet_filename,"Train//"+trainingSet_filename) # training set
vectorisation(trainingList,-int(testSize),len(trainingList),"Test//"+'+'+testSet_filename,"Test//"+'-'+testSet_filename,"Test//"+testSet_filename) # test set

print("Done------------------------------")