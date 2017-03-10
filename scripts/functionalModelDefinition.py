# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:10:46 2017

@author: g623829
"""

from keras.layers import Input, LSTM, Dense, merge
from keras.models import Model
import numpy as np


#define the MM multi-input, multi-output RNN
timeSteps = 15
gameVecSize = 50
pomVecSize = 20

#define the season inputs
seasonInput1 = Input(shape=(timeSteps,gameVecSize),name="season1Input")
seasonInput2 = Input(shape=(timeSteps,gameVecSize),name="season2Input")

#this LSTM processes the season vector for each team
sharedSeasonLSTM = LSTM(64)

#hook the LSTM up to its inputs
seasonVector1 = sharedSeasonLSTM(seasonInput1)
seasonVector2 = sharedSeasonLSTM(seasonInput2)

#merge the two seasonVectors
mergedSeasons = merge([seasonVector1,seasonVector2],mode='concat',concat_axis=-1)
#first output - based solely on the season-level info
seasonPreds = Dense(1,activation='sigmoid')(mergedSeasons)

#team statistics inputs
teamStatsInput1 = Input(shape=(pomVecSize,),name="teamStat1Input")
teamStatsInput2 = Input(shape=(pomVecSize,),name="teamStat12nput")

#this Dense layer processes the pom statistics for each team
sharedStatsDense = Dense(32,activation='relu')

#hook the Dense layer up to the season stats inputs
teamStats1 = sharedStatsDense(teamStatsInput1)
teamStats2 = sharedStatsDense(teamStatsInput2)

#merge the two team statistics layers
mergedStats = merge([teamStats1,teamStats2],mode='concat',concat_axis=-1)

#merge all of the layers
mergedAll = merge([mergedSeasons,mergedStats],mode='concat',concat_axis=-1)

#put 2 Dense layers on top
final = Dense(64,activation='relu')(mergedAll)
final = Dense(64,activation='relu')(final)
#final output predictions
predictions = Dense(1,activation='sigmoid')(final)

model = Model(input=[seasonInput1,seasonInput2,teamStatsInput1,teamStatsInput2],
              output=[seasonPreds,predictions])
model.compile(optimizer='rmsprop', loss='binary_crossentropy')



numExamples = 500
season1Matrix = np.random.rand(numExamples,timeSteps,gameVecSize)
season2Matrix = np.random.rand(numExamples,timeSteps,gameVecSize)
team1StatVec = np.random.rand(numExamples,pomVecSize)
team2StatVec = np.random.rand(numExamples,pomVecSize)
out = np.random.rand(numExamples)
out = np.where(out>0.5,1.,0.)

model.fit([season1Matrix,season2Matrix,team1StatVec,team2StatVec],[out,out],nb_epoch=2)
from keras.utils.visualize_util import plot
plot(model, to_file='../model.png')