###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
import numpy as np
import copy

class Solver:

    def __init__(self):
        self.word_count = {}
        self.emissprob = []
        self.emiss_train_prob = {}
        self.prob = {}
        self.pos = {".": 0, "adj": 1, "adp": 2, "adv": 3, "conj": 4, "det": 5, "noun": 6, "num": 7, "pron": 8, "prt": 9,
                    "verb": 10, "x": 11}
        self.postarray = []
        self.hmmpostarray = []
        self.initialwordpos = []
        self.transMatrix = np.zeros((12,12)) #[[0 for x in range(12)] for y in range(12)]

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            simpleprod = np.log(np.prod(self.postarray))
            return simpleprod
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return np.log(np.prod(self.hmmpostarray))
        else:
            print("Unknown algo!")

    # Initial state probability for HMM(Viterbi)
    def hmminitialprob(self,sentence):
        hmminitial = [] #np.empty((len(sentence),12))
        '''for eachword in sentence:
            if eachword in self.prob:
                val = self.prob[eachword]
            else:
                val = [[1] * 12]'''

        for eachword in sentence:
            #print(eachword)
            if eachword in self.word_count:
                val_lst = self.word_count[eachword]
                val = np.array(val_lst)
            else:
                val = np.zeros((12))#[[0] * 12]
            #val = np.reshape(val_lst,(12))
            #print(val)
            hmminitial.append(val)
        temp = np.array(hmminitial)
        return temp
        #unique_count = np.unique(self.initialwordpos, return_counts=True)
        #return np.divide(np.array(unique_count[1]), sum(np.array(unique_count[1])))

    # Transition state probability for HMM(Viterbi)
    def hmmtransprob(self):
        #print(self.transMatrix)
        '''self.hmmtranspro = np.zeros((12,12), dtype=float) #[[0 for x in range(12)] for y in range(12)]
        #self.hmmtranspro = np.divide(self.transMatrix.T,np.sum(self.transMatrix, axis=0)).T'''
        self.hmmtranspro = self.transMatrix.T
        #i=0
        #print(np.nan_to_num(self.hmmtranspro))
        #self.hmmtranspro = np.c_[self.hmminitialprob(), self.hmmtranspro]



    # Emission state probability for HMM(Viterbi)
    def hmmemissprob(self,sentence):
        temp = []
        for eachword in sentence:
            if eachword in self.emiss_train_prob:
                val = self.emiss_train_prob[eachword]
                #print(val)
            else:
                val = [[1] * 12]
            temp.append(val)
        self.emissprob = np.array(temp)
        #print((self.emissprob))


    # Do the training!
    #
    def train(self, data):
        for i in range(len(data)):
            for j in range(len(data[i][0])):
                if data[i][0][j] not in self.word_count:
                    self.word_count.setdefault(data[i][0][j], [int(0)] * 12)
                val = self.word_count[data[i][0][j]]
                if (len(data[i][1]) != j + 1):
                    self.transMatrix[self.pos[data[i][1][j]]][self.pos[data[i][1][j + 1]]] += 1
                    #print(self.transMatrix)

                val[self.pos[data[i][1][j]]] += 1
            self.initialwordpos.append(data[i][1][0])
            self.word_count.update({data[i][0][j]: val})
            # word_counttemp={}
        word_counttemp = copy.deepcopy(self.word_count)

        for i, (k, v) in enumerate(word_counttemp.items()):
            row = np.array(v)
            total = sum(v)
            temp = row / total
            self.prob.setdefault(k, temp)
            #print(np.log(row+1/(sum(v)+len(self.word_count.keys()))))
            '''self.emiss_train_prob.setdefault(k,((row+1)/(sum(v)+len(self.word_count.keys()))))'''
            self.emiss_train_prob.setdefault(k,(row+1))
        #print(self.prob)

        # print(self.emiss_train_prob)
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        testpos = []
        for eachword in sentence:
            if eachword in self.prob:
                v = self.prob[eachword]
                for i, val in enumerate(v):
                    if val == np.max(v):
                        testpos.append((list(self.pos.keys())[list(self.pos.values()).index(i)]))
                        self.postarray.append((np.max(v)))
                        break
            else:
                testpos.append("X")
                self.postarray.append(0.00001)
        return testpos

    def complex_mcmc(self, sentence):
        initial_pos=["noun"]*len(sentence)
        for i in range(0, 100):
            for j in range(0, len(sentence)):
                probability_values = []

        return ["noun"] * len(sentence)

    def hmm_viterbi(self, sentence):
        hmminit = self.hmminitialprob(sentence)
        # print(hmminit)
        self.hmmtransprob()

        self.hmmemissprob(sentence)
        # print(self.emissprob)
        '''#prod1 = np.dot(np.prod(self.hmmtranspro.T, axis=0),np.prod(self.emissprob.T, axis=0))
        #prod1 = np.prod(self.hmmtranspro, axis=0) * np.prod(self.emissprob.T, axis=0)
        #print(prod1)
        vij = np.dot(hmminit,np.dot(self.hmmtranspro, self.emissprob.T))
        #print((np.prod(self.hmmtranspro, axis=0) @ np.prod(self.emissprob.T, axis=
        prod1 = np.dot(hmminit, self.hmmtranspro)
        print(np.dot(prod1.T, self.emissprob).shape)
        exit(0)
        tr_vij = vij.T'''
        #print((self.emissprob.T).shape)
        vij = hmminit + np.sum(self.hmmtranspro) + np.sum(self.emissprob.T)
        lvij = np.log(vij)
        #print(vij)
        #exit(0)
        hmmpos = []
        for row in lvij:
            #print(np.max(row))
            for coord, val in enumerate(row):
                if val == np.max(row):
                    #print(coord)
                    hmmpos.append((list(self.pos.keys())[list(self.pos.values()).index(coord)]))
                    self.hmmpostarray.append((np.max(row)))
                    break
        #print(hmmpos)

        return hmmpos

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")