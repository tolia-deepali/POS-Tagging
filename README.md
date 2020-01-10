# Parts of Speech Tagger
##### ----------------------------------------------------------------------------------------------------------------------------------------------
#### Problem Statement :
Implement a part of speech tagger, using Bayesian networks, in which the goal was to mark every word in a sentence with its correct part of speech. Use 3 different models of Bayes Networks 1) Simplified 2) Viterbi(HMM) 3) Complex Model (MCMC with Gibs Sampling).

#### Code Description :
##### Training The Data -
While training the data we calculated initial probabilities by counting the number of times a particular part of speech that is occurring at the first position of the sentence. This is stored in initialState dictionary.
Further we calculated the total number of times each part of speech present in whole training data(posTotal Dict).
Transition Probabilities of one part of speech to another is stored in the transitionState dictionary which is a nested dictionary.
emissionState dictionary is used to store the emission probabilities that are word given part of speech based on the training data. It is a count of number of word that belong to particular POS.
To calculate p(Si/Si-1,Si-2) we created dictionary of dictionary of dictionary which stores transition probability from one part of speech to next to next. This is used in MCMC with Gibs Sampling.

##### Simplified Model -
In this part we detect the part of speech for a word based on maximum of the probabilities of part of speech given the word.
i.e We calculated the p(word/part of speech) * p(part of speech) as we are maximizing over the probabilities we are neglecting the denominator i.e p(word). In this model we consider that observed word is only depending on the pos tag.

##### HMM Viterbi Model -
Based on estimated probabilities and transition probabilities, the best once are use to determine tags at next state. We have used two arrays viterbi and trace_matrix rows as number of POS and columns as number of words per sentence. Viterbi is used to store posteriors of all pos for particular word in sentence. trace_matrix stores the indexes of the posterior which had maximum Viterbi coefficient. For the first word, the Viterbi coefficient depend on the initial probability * emission probability. For all the other words we maximize over the probabilities that are calculated for previous word and its transitions to the current state. Viterbi Algorithm gives best results as it calculates the probability of entire sequence instead of just each word.

##### Complex Model (MCMC Gibs Sampling) -
For Gibs sampling we need to generate initial sample. Firstly we used noun as a part of speech for all the words in the sentence. This is because the noun has highest probability in the training data. But it required more iterations to generate perfect sample. Thats why then we generated initial sample, choosing randomly selected tags from POS list distribution for each word separately. We are generating 100 samples and discarding first 30 to improve the accuracy of the samples.

First we consider the POS of first word using initial and emission probability. Similarly for second we use transition probabilities to assign new POS considering POS given part of speech of 1st one. But for the remaining we use p(Si/Si-1,Si-2) i.e calculate the pos of iTH word we consider 2 words before that word.

For every iteration after calculating every possible POS for word a random index is generated for assigning POS which is dependent on the probabilities derived earlier. This states are appended to the sample list. After generating all the samples the maximum occurring POS is consider for each word of the sentence.

##### Posteriors

`P(S1,S2....Sn|W1,W2....Wn)`

1) Simplified :  
`P(S1/W1) = P(W1/S1) * P(S1) neglecting denominator`

2) Viterbi :   Here we calculate probability considering only 1 prior state

`P(S1,S2,..Sn /W1, W2,.. Wn) = P(W1/S1) * P(W1) * P(W2/S2) * P(W2) * P(S2/S1).. and so on`


3) Complex :  Here we calculate probability considering 2 prior states
`P(S1,S2,..Sn /W1, W2,.. Wn) = P(W1/S1) * P(W1) * P(W2/S2) * P(W2) * P(S2/S1) * P(W3/S3) * P(W3) * P(S3/S1,S2) .. and so on
`
##### Challenges and Assumptions-
1) For all the 3 models if any values were not found, we initialized the values to small one i.e 0.000001

2) We had problem deciding the initial probability for Viterbi and initial sample set for MCMC Gibs Sampling.
@ Deepali Suggested to go with the approach of calculating all the POS of 1st words in the training data for Viterbi as initial probabilities.
For Gibs we tried 2 different approaches finding which will suit better.
a) Noun for all the word
b) Random POS
Went with Random POS as it gave good results in less iterations.

3) Challenge: Fine tuning Viterbi to get best results

#### Final Output -

==> So far scored 2000 sentences with 29442 words.

                                Words correct:     Sentences correct:
              0. Ground truth:      100.00%              100.00%
                    1. Simple:       93.92%               47.45%
                    2. HMM:          92.68%               52.20%
                    3. Complex:      90.08%               33.85%
----
#### Run Code :

`python ./label.py bc.train bc.test.tiny`


#### References :

1) https://twiecki.io/blog/2015/11/10/mcmc-sampling/

2) https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm/9730083
