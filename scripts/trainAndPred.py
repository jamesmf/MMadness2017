from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense,RepeatVector,Merge
from keras.layers.recurrent import GRU,LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.datasets.data_utils import get_file
import pandas as pd
import numpy as np
from scipy.stats import norm
from os import mkdir
from os.path import isdir
import sys
import csv
from keras.models import model_from_json



class Team():
    
    def __init__(self,ID,year):
        self.ID     = ID
        self.year   = int(year)
        self.name   = self.IDtoName().strip()
        self.games  = self.getGames()
        self.stats  = self.getStats()
#        self.season = self.getSeason()
#        self.seed   = self.getSeed()
        
    def IDtoName(self):
        return IDtoPom[self.ID]
        
        
    def getGames(self):
        gc = [i for i in games.columns]
        WcolNames = ["Daynum"] + [i for n,i in enumerate(games.columns) if (n >= gc.index("Wfgm")) and (n <= gc.index("Wpf"))]
        WcolNames = WcolNames + [i for n,i in enumerate(games.columns) if (n >= gc.index("Wins_LT")) and (n <= gc.index("NCSOS Pyth Rank_LT"))]
        WcolNames = WcolNames + ["Wresult"]
        LcolNames = ["Daynum"] + [i for n,i in enumerate(games.columns) if (n >= gc.index("Lfgm")) and (n <= gc.index("Lpf"))]
        LcolNames = LcolNames + [i for n,i in enumerate(games.columns) if (n >= gc.index("Wins")) and (n <= gc.index("NCSOS Pyth Rank"))]
        LcolNames = LcolNames + ["Lresult"]
        gy  = games[games["Season"] == self.year]
        Ws  = gy[gy["Wteam"] == self.ID]
        Ls  = gy[gy["Lteam"] == self.ID]
        WGames = Ws[WcolNames]
        LGames = Ls[LcolNames]
        WG = WGames.as_matrix()
        LG = LGames.as_matrix()
        gs  = np.append(WG,LG,axis=0)
        gs  = gs[gs[:,0].argsort()]
        return gs
 
    def getStats(self):
        temp = pom.loc[(self.name,self.year)].drop("Conference")
        seed = temp["Seed"]
        self.seed = np.max([np.isnan(seed)*20,seed])
        return temp.as_matrix()

        
#    def getSeed(self):
#        py  = pom[pom["Year"] == self.year]
#        if len(py[py["Team"] == self.name].as_matrix()) > 0:
#            py  = py[py["Team"] == self.name].as_matrix()[0]   
#            py  = py[6]
#        else:
#            py  = 32
#        return py        
        
    def sampleFromSeason(self,numGames=15):
        out     = []
        inds    = np.random.choice(len(self.games),numGames,replace=False)
        inds    = sorted(inds)
#        print(inds)
#        while (len(out) < numGames) and (len(season)>0):
#            ind     = int(np.floor(len(season)*np.random.rand()))
#            out.append(season.pop(ind))
        out = self.games[inds]
        return out
        


def getPomName(val):
    if val in IDtoPom:
        return IDtoPom[val]
    else:
        return None


def getIDtoPom():
    with open("../fullPomMap.csv",'rb') as f:
        l   = [x for x in f.read().split("\n") if x != '']
    IDtoPom     = {}
    for row in l:
        s   = row.split(",")
        IDtoPom[int(s[0])] = s[1]
    return IDtoPom

def submissionToTourney(sub):
    sub = sub.as_matrix()
    out = []
    for s in sub:
        a   = []
        sp  = s[0].split("_")
        a.append(sp[0])
        a.append(136) #Day number
        a.append(int(sp[1]))
        a.append(1)
        a.append(int(sp[2]))
        a.append(2)
        out.append(a)
        
    cols    = ["Season","Daynum","Wteam","Wscore","Lteam","Lscore"]
    out     = pd.DataFrame(data=out,columns=cols)
    return out
    

def getResult(score1,score2,resultType):
    if resultType == "categorical":
        if score1 > score2:
            return 1
        else:
            return 0
    else:
        return score1 - score2

def getData(allTeams,tourney,predict=False,resultType="categorical"):
    Xtrain = []
    ytrain = []
    seasonInds = []
    for rownum, row in enumerate(tourney.as_matrix()):
        print(rownum,end="\r")
        sys.stdout.flush()
        example = np.zeros((timeSteps,2*vecSize))
        year    = row[0]
        wteamID = row[2]
        wScore  = row[3]
        lteamID = row[4]
        lScore  = row[5]
        check   = np.random.rand()
        if predict:
            check = 1
        if check > 0.5:
            team1   = Team(wteamID,year)
            team2   = Team(lteamID,year)
            result  = getResult(wScore,lScore,resultType=resultType)
        else:
            team1   = Team(lteamID,year)
            team2   = Team(wteamID,year)
            result  = getResult(lScore,wScore,resultType=resultType)
        
        t1  = str(team1.ID)+"_"+str(year)
        t2  = str(team2.ID)+"_"+str(year)
        
        if t1 not in allTeams:
            allTeams[t1] = Team(team1.ID,year)
        if t2 not in allTeams:
            allTeams[t2] = Team(team2.ID,year)
        
        if predict:
            nTop = numPreds
        else:
            nTop = iterations
            
        for ind in range(0,nTop):
            season1 = team1.sampleFromSeason(numGames=timeSteps)
            season2 = team2.sampleFromSeason(numGames=timeSteps)
            for n,game in enumerate(season1):
#                if n==0:
#                    print(season1[n])
                g = np.append(season1[n],season2[n])
                #g           = np.append(g,result)
                example[n,:] = g
#            print(example[n])
            #print(example[n])
            
            Xtrain.append(example)
            ytrain.append(result)
            seasonInds.append(year)
#            
#            if len(Xtrain) > 1:
#                print("X[-1]", Xtrain[-1][-1])
#                print("X[-2]", Xtrain[-2][-1])
#            stop=raw_input("")
        
    Xtrain  = np.reshape(Xtrain,(len(Xtrain),timeSteps,2*vecSize))
    ytrain = np.array(ytrain).reshape((len(ytrain),1))
    seasonInds = pd.DataFrame(seasonInds,columns=["year"])
    print()
    return Xtrain,ytrain,seasonInds


def defineModel():
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(timeSteps,2*vecSize)))
    model.add(Dropout(.25))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dense(256))
    model.add(Dropout(.25))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model

def replaceMissingSeed(x):
    try:
        out     = int(x)
        return out
    except ValueError:
        return 30

def loadThatModel(folder):
    with open(folder+".json",'rb') as f:
        json_string     = f.read()
    model = model_from_json(json_string)
    model.load_weights(folder+".h5")
    return model
        
        
print("begun")       
epochs = 25
iterations = 2
timeSteps = 8
numPreds = 30
vecSize = 67
noiseFactor = 0.15

modelPath = "../models/"
phase = "phase1"
       
IDtoPom = getIDtoPom()       
pom = pd.read_csv("../data/kenpom.csv")
pom["Team"] = pom["Team"].apply(lambda x: x.strip())
pom["Seed"] = pom["Seed"].apply(lambda x: replaceMissingSeed(x))
pom = pom.set_index(["Team","Year"])
np.random.seed(0)      

games = pd.read_csv("../data/"+phase+"/RegularSeasonDetailedResults.csv")
games["Lteam_name"] = games[["Season","Lteam"]].apply(lambda x: getPomName(x["Lteam"]),axis=1)
games["Wteam_name"] = games[["Season","Wteam"]].apply(lambda x: getPomName(x["Wteam"]),axis=1)
games["Wresult"] = games["Wscore"] - games["Lscore"]
games["Lresult"] = games["Lscore"] - games["Wscore"]

#join on team stats, subset to games where we have data for both teams involved
games = pd.merge(games,pom,left_on=["Wteam_name","Season"],right_index=True,suffixes=('','_WT'))
games = pd.merge(games,pom,left_on=["Lteam_name","Season"],right_index=True,suffixes=('','_LT'))


trainTourney = pd.read_csv("../data/"+phase+"/TourneyDetailedResults.csv")
sub = pd.read_csv("../data/"+phase+"/SampleSubmission.csv")
testTourney = submissionToTourney(sub)
predictionSeasons = testTourney["Season"].unique()

allTeams    = {}
allRMSEs    = []

teamIDs     = []   
#
#if len(sys.argv) == 1:
#    model       = defineModel()
#
#    Xtrain, ytrain, seasonInds = getData(allTeams,trainTourney)
#    print("data done")
#
#    print("getting stats")
#    tempX           = np.reshape(Xtrain,(Xtrain.shape[0]*Xtrain.shape[1],Xtrain.shape[2]))
#    means           = np.ones((timeSteps,tempX.shape[1]))*np.mean(tempX,axis=0)
#    stds            = np.ones((timeSteps,tempX.shape[1]))*np.std(tempX,axis=0)
#
#    Xtrain -= means
#    Xtrain /= stds+np.ones((stds.shape))*0.000001       
#    print("fitting")    
#    
#    for predSeason in predictionSeasons:
#        #each season we're predicting for we use the prior 2 tournaments as CV
#        cvSeasons = [ int(predSeason)-1, int(predSeason)-2]
#        holdOutSeasons = cvSeasons+[int(predSeason)]
#        #get a mask for the training folds, the CV fold, and the predictions
#        cvFoldInds = seasonInds["year"].apply(lambda x: x in cvSeasons)
#        cvFoldInds = cvFoldInds[cvFoldInds==True].index.tolist()
#        predFoldInds = seasonInds["year"].apply(lambda x: x == predSeason)
#        predFoldInds = predFoldInds[predFoldInds==True].index.tolist()
#        trainFoldInds = seasonInds["year"].apply(lambda x: not x in holdOutSeasons)
#        trainFoldInds = trainFoldInds[trainFoldInds==True].index.tolist()
#        
#        noise = np.random.rand(Xtrain.shape[0],Xtrain.shape[1],Xtrain.shape[2])*noiseFactor
#        
#        
#        Xtraintemp = Xtrain[trainFoldInds]+noise[trainFoldInds]
#        ytraintemp = ytrain[trainFoldInds]
#        Xcvtemp = Xtrain[cvFoldInds]+noise[cvFoldInds]
#        ycvtemp = ytrain[cvFoldInds]
#        Xtesttemp = Xtrain[predFoldInds]
#        ytesttemp = ytrain[predFoldInds]
#        
#        callbacks = [
#            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
#            ModelCheckpoint(modelPath+'cv_'+predSeason+'_rnn', 
#                            monitor='val_loss', save_best_only=True, verbose=1),
#        ]        
#        
#        model.fit(Xtraintemp,ytraintemp,nb_epoch=epochs,validation_data=(Xcvtemp, ycvtemp),callbacks=callbacks)
#
##    jsonstring  = model.to_json()
##    with open("../model/mmRNN.json",'wb') as f:
##        f.write(jsonstring)
##    model.save_weights("../model/mmRNN.h5",overwrite=True)
#
#else:
#    model   = loadThatModel("../model/mmRNN")
#
#
#
#output  = np.zeros((tourney.shape[0],numPreds))
#
#Xtest,dummy     = getData(allTeams,tourney,predict=True)
#Xtest           -=means
#Xtest           /=stds+np.ones((stds.shape))*0.0000001
#
#print(tourney.shape,Xtest.shape)
#preds   = model.predict(Xtest)
#print(preds.shape)
#preds   = np.reshape(preds,output.shape)
#print(preds.shape)
#output  = preds
#
##np.save("../output.nd",output)    
##np.save("../Xtest",Xtest)
##np.save("../means",means)
##np.save()
#
#    
#means   = np.mean(output,axis=1)
#stds    = np.std(output,axis=1)
#
#uncert  = 3
#towrite = []
#for game in range(0,means.shape[0]):
#    ID      = sub.loc[game]["Id"]
#    print(ID, means[game],stds[game])
#    st      = stds[game]*uncert + 0.0000001
#    prob    = norm.cdf(-0.01,loc=means[game],scale=st)
#    towrite.append([ID,prob])
#
#with open("../predictions3.csv",'wb') as f:
#    wr  = csv.writer(f)
#    for row in towrite:
#        wr.writerow(row)