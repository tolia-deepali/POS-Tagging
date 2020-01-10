###################################
# CS B551 Fall 2019, Assignment #3
#
# Your names and user ids:
# Kaustubh Bhalerao - kbhaler, Deepali Tolia - dtolia, Suyash Poredi - sporedi
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

    #  Constructor for all the dictionaries and variables used in program globally
    def __init__(self):
        self.pos = {".": 0, "adj": 1, "adp": 2, "adv": 3, "conj": 4, "det": 5, "noun": 6, "num": 7, "pron": 8, "prt": 9,
                    "verb": 10, "x": 11}
        self.initialState = {}
        self.transitionState = {x: {y: 0.000001 for y in self.pos.keys()} for x in self.pos.keys()}
        self.emissionState = {}
        for i in self.pos.keys():
            self.emissionState[i] = {}
        self.simplifiedPosterior = 0
        self.complexPosterior = 0
        self.viterbiPosterior = 0
        self.pos_total = {}
        self.gibs_samples = []
        self.complexDict = {x: {y: {z: 0.000001 for z in self.pos.keys()} for y in self.pos.keys()} for x in
                            self.pos.keys()}

    # Calculate the log of the posterior probability of a given sentence
    #  Posteriors are calculated in respective functions and global variables are set which are used in the below function
    def posterior(self, model, sentence, label):
        if model == "Simple":
            simple_prod = self.simplifiedPosterior
            return simple_prod
        elif model == "Complex":
            return self.complexPosterior
        elif model == "HMM":
            return self.viterbiPosterior
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):

        for i in data:
            # Initial State Dictionary
            if i[1][0] in self.initialState:
                self.initialState[i[1][0]] += 1
            else:
                self.initialState[i[1][0]] = 1
            # total number of POS in training set
            for pos in i[1]:
                try:
                    self.pos_total[pos] += 1
                except:
                    self.pos_total[pos] = 1
            # Transition State Count
            for pos in range(len(i[1])):
                if len(i[1]) != pos + 1:
                    try:
                        self.transitionState[i[1][pos]][i[1][pos + 1]] += 1
                    except:
                        self.transitionState[i[1][pos]][i[1][pos + 1]] = 1
            # Emission State Count
            for j in range(len(i[1])):
                try:
                    self.emissionState[i[1][j]][i[0][j]] += 1
                except:
                    self.emissionState[i[1][j]][i[0][j]] = 1

            # Triple transition count
            for k in range(len(i[1])):
                if len(i[1]) != 2:
                    if k + 2 < (len(i[1])):
                        self.complexDict[i[1][k]][i[1][k + 1]][i[1][k + 2]] += 1
                    # print(i[1][k],i[1][k+1],i[1][k+2])
                    # print(self.complexDict)
                    # exit(0)

        # Probabilities for Initial, Transition, Emission, Tripple

        sum_pos_total = sum(self.pos_total.values())
        for i in self.pos_total:
            self.pos_total[i] /= sum_pos_total

        sum_ini_total = sum(self.initialState.values())
        for i in self.initialState:
            self.initialState[i] /= sum_ini_total

        for i in self.transitionState:
            temp_sum = sum(self.transitionState[i].values())
            for j in self.transitionState[i]:
                self.transitionState[i][j] /= temp_sum

        for i in self.emissionState:
            temp_sum = sum(self.emissionState[i].values())
            for j in self.emissionState[i]:
                self.emissionState[i][j] /= temp_sum

        for i in self.complexDict:
            for j in self.complexDict[i]:

                temp_sum = sum(self.complexDict[i][j].values())

                for k in self.complexDict[i][j]:
                    self.complexDict[i][j][k] /= temp_sum

    # We detect the part of speech for a word based on maximum of the probabilities of part of speech given the word
    def simplified(self, sentence):

        posList = list(self.pos.keys())

        resultPos = []
        posterior = 1
        for word in sentence:
            temp_probability_list = []
            for everyPos in posList:
                # POS prior probability * emission probability
                try:
                    temp_prob = self.pos_total[everyPos] * self.emissionState[everyPos][word]
                    temp_probability_list.append(temp_prob)
                except:
                    self.emissionState[everyPos][word] = 0.000001
                    temp_prob = self.pos_total[everyPos] * self.emissionState[everyPos][word]
                    temp_probability_list.append(temp_prob)

            max_prob_index = temp_probability_list.index(max(temp_probability_list))

            posterior *= temp_probability_list[max_prob_index]

            resultPos.append(posList[max_prob_index])

        self.simplifiedPosterior = math.log(posterior)
        # print(resultPos)
        return resultPos

    # Gibs Sampling MCMC
    # WE generate 100 samples discarding 1st 30 for Accuracy
    def complex_mcmc(self, sentence):

        posList = list(self.pos.keys())
        initial_pos = []
        for p in range(len(sentence)):
            initial_pos.append(random.choice(posList))
        # initial_pos = self.simplified(sentence)

        # print(self.complexDict)
        for i in range(100):
            for j in range(len(sentence)):
                prob_values = []
                sum_prob = 0
                for pos in range(len(posList)):
                    value = 0.000001

                    if j == 0:
                        value = self.initialState[posList[pos]]
                    elif j == 1:
                        value = self.initialState[initial_pos[j - 1]] * self.transitionState[initial_pos[j - 1]][
                            posList[pos]]
                    else:
                        value = 1
                        if not (j >= (len(sentence) - 2)):
                            index = j + 2
                        if j == len(sentence) - 2:
                            index = j + 1
                            value *= self.transitionState[initial_pos[j]][initial_pos[index]]
                        if j == len(sentence) - 1:
                            index = j

                        while index >= 1:

                            if index == 1:
                                value *= self.initialState[initial_pos[index - 1]] * \
                                         self.transitionState[initial_pos[index - 1]][initial_pos[index]]
                            else:   #for ith word we use p(Si/Si-1,Si-2)
                                if index == j:
                                    value *= self.complexDict[initial_pos[index - 2]][initial_pos[index - 1]][
                                        posList[pos]]
                                else:
                                    value *= self.complexDict[initial_pos[index - 2]][initial_pos[index - 1]][
                                        initial_pos[index]]
                            index -= 1
                    if self.emissionState[posList[pos]][sentence[j]]:
                        word_temp_prob = self.emissionState[posList[pos]][sentence[j]]
                    else:
                        word_temp_prob = 0.000001;
                    probability = value * word_temp_prob
                    sum_prob += probability
                    prob_values.append(probability)
                for l in range(len(prob_values)):
                    prob_values[l] = prob_values[l] / sum_prob

                # Creating random Index choice of POS for every word
                random_indexchoice = int(np.random.choice(len(posList), p=np.array(prob_values)))

                initial_pos[j] = posList[random_indexchoice]
            # Adding new generated sample to list of samples
            self.gibs_samples.append(initial_pos)
        top_samples = self.gibs_samples[len(self.gibs_samples) - 30:]    # Discarding 1st 30 samples


        # Maximizing over the List of Samples and taking max values of POS
        result_pos = []
        for wrd in range(len(sentence)):
            temp_list = []
            for r in top_samples:
                temp_list.append(r[wrd])
            pos = max(temp_list, key=temp_list.count)
            result_pos.append(pos)
        # print(pos_array)

        # calcularing Posterior using the last selected POS list for given Sentence
        posterior = 1

        for pos in range(len(result_pos)):

            value = 0.000001

            if pos == 0:
                value = self.initialState[result_pos[pos]]
            elif pos == 1:
                try:
                    value = self.transitionState[result_pos[pos - 1]][result_pos[pos]] * self.initialState[
                        result_pos[pos - 1]]
                except:
                    pass
            else:
                value = self.complexDict[result_pos[pos - 2]][result_pos[pos - 1]][result_pos[pos]]

            if self.emissionState[result_pos[pos]][sentence[pos]]:
                word = self.emissionState[result_pos[pos]][sentence[pos]]
            else:
                word = 0.000001
            posterior *= value * word

        self.complexPosterior = math.log(posterior)
        return result_pos

    # Based on estimated probabilities and transition probabilities, the best once are use to determine tags at next state.
    def hmm_viterbi(self, sentence):
        posList = list(self.pos.keys())
        viterbi = np.zeros((12, len(sentence)))
        trace_matrix = np.zeros((12, len(sentence)), dtype=int) # Matrix to Backtrack
        for i in range(len(sentence)):
            for j in range(len(posList)):
                if i == 0:
                    if self.emissionState[posList[j]][sentence[i]]:
                        viterbi[j][i] = math.log(self.initialState[posList[j]]) + math.log(
                            self.emissionState[posList[j]][sentence[i]])
                    else:
                        self.emissionState[posList[j]][sentence[i]] = 0.000001
                        viterbi[j][i] = math.log(self.initialState[posList[j]]) + math.log(
                            self.emissionState[posList[j]][sentence[i]])
                else:
                    temp_vit_list = []
                    for pos in range(len(posList)):
                        if self.transitionState[posList[pos]][posList[j]]:
                            value1 = self.transitionState[posList[pos]][posList[j]]
                        else:
                            value1 = 0.000001

                        value2 = viterbi[pos][i - 1]
                        temp_vit_list.append(math.log(value1) + value2)

                    max_index = temp_vit_list.index(max(temp_vit_list))
                    max_value = max(temp_vit_list)
                    if self.emissionState[posList[j]][sentence[i]]:
                        viterbi[j][i] = max_value + math.log(self.emissionState[posList[j]][sentence[i]])
                    else:
                        viterbi[j][i] = max_value + math.log(0.000001)
                    trace_matrix[j][i] = max_index

        # Taking the last max for back tracking
        last_max = np.argmax(viterbi[:, -1])
        result_pos = [posList[last_max]]

        # Backtrack and return the result
        for i in range(1, len(sentence)):
            back_last = trace_matrix[last_max, -i]
            result_pos.append(posList[back_last])

        result_pos.reverse()
        # Calculate Posterior
        self.viterbiPosterior = np.max(viterbi[:, -1])

        return result_pos

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
