from __future__ import print_function
from keras.layers import Input, LSTM, Dense, merge, Dropout
from keras.models import Model
import numpy as np
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
from keras.utils.visualize_util import plot
from keras.models import load_model
from sklearn import metrics



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
#        WcolNames.pop(WcolNames.index("Conference"))
        LcolNames = ["Daynum"] + [i for n,i in enumerate(games.columns) if (n >= gc.index("Lfgm")) and (n <= gc.index("Lpf"))]
        LcolNames = LcolNames + [i for n,i in enumerate(games.columns) if (n >= gc.index("Wins")) and (n <= gc.index("NCSOS Pyth Rank"))]
        LcolNames = LcolNames + ["Lresult"]
#        LcolNames.pop(LcolNames.index("Conference"))    
        
        gy  = games[games["Season"] == self.year]
        Ws  = gy[gy["Wteam"] == self.ID]
        Ls  = gy[gy["Lteam"] == self.ID]
        WGames = Ws[WcolNames]
        LGames = Ls[LcolNames]
        WG = np.array(WGames.as_matrix(),dtype=np.float32)
        LG = np.array(LGames.as_matrix(),dtype=np.float32)
        gs  = np.append(WG,LG,axis=0)
        gs  = gs[gs[:,0].argsort()]
        return gs
 
    def getStats(self):
        temp = pom.loc[(self.name,self.year)].drop("Conference")
        seed = temp["Seed"]
        self.seed = np.max([np.isnan(seed)*20,seed])
        return np.array(temp.as_matrix(),dtype=np.float32)

        
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
        IDtoPom[int(s[0])] = s[1].strip()
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
            return 1.0
        else:
            return 0.0
    else:
        return score1 - score2

def getData(allTeams,tourney,predict=False,resultType="categorical"):
    Xseason1train = []
    Xseason2train = []
    Xstats1train = []
    Xstats2train = []
    ytrain = []
    seasonInds = []
    gameInfo = []
    for rownum, row in enumerate(tourney.as_matrix()):
        year    = row[0]
        wteamID = row[2]
        wScore  = row[3]
        lteamID = row[4]
        lScore  = row[5]
        check   = np.random.rand()
        if predict:
            check = 1
        if check > 0.5:
            t1 = str(wteamID)+"_"+str(year)
            t2 = str(lteamID)+"_"+str(year)
            if t1 in allTeams:
                team1 = allTeams[t1]
            else:                
                team1   = Team(wteamID,year)
            if t2 in allTeams:
                team2 = allTeams[t2]
            else:
                team2   = Team(lteamID,year)
            result  = getResult(wScore,lScore,resultType=resultType)
        else:
            t1 = str(lteamID)+"_"+str(year)
            t2 = str(wteamID)+"_"+str(year)
            if t1 in allTeams:
                team1 = allTeams[t1]
            else:                
                team1   = Team(lteamID,year)
            if t2 in allTeams:
                team2 = allTeams[t2]
            else:
                team2   = Team(wteamID,year)
            result  = getResult(lScore,wScore,resultType=resultType)
        
        if t1 not in allTeams:
            allTeams[t1] = team1
        if t2 not in allTeams:
            allTeams[t2] = team2
        
        if predict:
            nTop = numPreds
        else:
            nTop = iterations
        
#        print("t1",t1)
#        print("t2",t2)
        for ind in range(0,nTop):
            season1 = team1.sampleFromSeason(numGames=timeSteps)
            season2 = team2.sampleFromSeason(numGames=timeSteps)
            
            Xseason1train.append(season1)
            Xseason2train.append(season2)
            Xstats1train.append(team1.stats)
            Xstats2train.append(team2.stats)
            ytrain.append(result)
            seasonInds.append(year)
            gameInfo.append([year,t1,t2])
#            
#            if len(Xtrain) > 1:
#                print("X[-1]", Xtrain[-1][-1])
#                print("X[-2]", Xtrain[-2][-1])
#            stop=raw_input("")
        
    Xseason1train  = np.array(Xseason1train)
    Xseason2train = np.array(Xseason2train)
    Xstats1train = np.array(Xstats1train)
    Xstats2train = np.array(Xstats2train)
    ytrain = np.array(ytrain).reshape((len(ytrain),1))
    seasonInds = pd.DataFrame(seasonInds,columns=["year"])
    gameInfo = np.array(gameInfo)
    print()
    return Xseason1train,Xseason2train,Xstats1train,Xstats2train,ytrain,seasonInds,gameInfo


def defineModel(timeSteps,gameVecSize,pomVecSize):
    #define the MM multi-input, multi-output RNN
    #define the season inputs
    seasonInput1 = Input(shape=(timeSteps,gameVecSize),name="season1Input")
    seasonInput2 = Input(shape=(timeSteps,gameVecSize),name="season2Input")
    
    #this LSTM processes the season vector for each team
    sharedSeasonLSTM = LSTM(128,dropout_W=0.5,dropout_U=0.5,name='seasonLSTM')
    
    #hook the LSTM up to its inputs
    seasonVector1 = sharedSeasonLSTM(seasonInput1)
    seasonVector2 = sharedSeasonLSTM(seasonInput2)
    
    #merge the two seasonVectors
    mergedSeasons = merge([seasonVector1,seasonVector2],mode='concat',concat_axis=-1)
    #first output - based solely on the season-level info
    seasonPreds = Dense(1,activation='sigmoid',name='SeasonOnlyPrediction')(mergedSeasons)
    
    #team statistics inputs
    teamStatsInput1 = Input(shape=(pomVecSize,),name="teamStats1Input")
    teamStatsInput2 = Input(shape=(pomVecSize,),name="teamStats2Input")
    
    #this Dense layer processes the pom statistics for each team
    sharedStatsDense = Dense(64,activation='relu')
    
    #hook the Dense layer up to the season stats inputs
    teamStats1 = sharedStatsDense(teamStatsInput1)
    teamStats1 = Dropout(0.5)(teamStats1)
    teamStats2 = sharedStatsDense(teamStatsInput2)
    teamStats2 = Dropout(0.5)(teamStats2)
    
    #merge the two team statistics layers
    mergedStats = merge([teamStats1,teamStats2],mode='concat',concat_axis=-1)
    
    #merge all of the layers
    mergedAll = merge([mergedSeasons,mergedStats],mode='concat',concat_axis=-1)
    
    #put 1 Dense layers on top
    final = Dense(64,activation='relu')(mergedAll)
    final = Dropout(0.5)(final)
    final = Dense(32,activation='relu')(final)
    final = Dropout(0.5)(final)
    #final output predictions
    predictions = Dense(1,activation='sigmoid',name='Prediction')(final)
    
    model = Model(input=[seasonInput1,seasonInput2,teamStatsInput1,teamStatsInput2],
                  output=[seasonPreds,predictions])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  loss_weights=[0.5,1.0])
    
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
iterations = 4
timeSteps = 12
numPreds = 30
#gameVecSize = 35
runType = "noCV"
patience = 3

modelPath = "../models/"
phase = "phase2"
       
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
predsForSubmission = None

if runType == "CV":
#    times = [12,15,18]
#    noises1 = [0.05,0.1,0.25]
#    noises2 = [0.05,0.1,0.25]
#    itNums = [2,4]
    times = [18]
    noises1 = [0.15]
    noises2 = [0.1]
    itNums = [2]
else:
    times = [18]
    noises1 = [0.25]
    noises2 = [0.1]
    itNums = [2]
    numEpochs = 5
cvList = []
if len(sys.argv) == 1:
    for timeSteps in times:
        for iterations in itNums:
            Xseason1train,Xseason2train,Xstats1train,Xstats2train,ytrain,seasonInds,gameInfo = getData(allTeams,trainTourney)
    
            gameVecSize = Xseason1train.shape[-1]
            pomVecSize = Xstats1train.shape[-1]
        
        
            tempSeasonX = np.reshape(Xseason1train,(Xseason1train.shape[0]*Xseason1train.shape[1],Xseason1train.shape[2]))
            means = np.ones((timeSteps,tempSeasonX.shape[1]))*np.mean(tempSeasonX,axis=0)
            stds = np.ones((timeSteps,tempSeasonX.shape[1]))*np.std(tempSeasonX,axis=0)
        
            statmeans = np.mean(np.append(Xstats1train,Xstats2train,axis=0),axis=0)
            statstds = np.std(np.append(Xstats1train,Xstats2train,axis=0),axis=0)
        
            Xseason1train -= means
            Xseason2train -= means
            Xseason1train /= stds+np.ones((stds.shape))*0.1
            Xseason2train /= stds+np.ones((stds.shape))*0.1
            
            Xstats1train -= statmeans
            Xstats2train -= statmeans
            Xstats1train /= statstds+np.ones((statstds.shape))*0.1
            Xstats2train /= statstds+np.ones((statstds.shape))*0.1
        
            
            print("fitting")    
          
            for predSeason in predictionSeasons:
                for noiseFactor1 in noises1:
                    for noiseFactor2 in noises2:
                        #define the model
                        model = defineModel(timeSteps,gameVecSize,pomVecSize)            
                        
                        #each season we're predicting for we use the prior 2 tournaments as CV
                        cvSeasons = [ int(predSeason)-1, int(predSeason)-2]
                        if runType != "CV":
                            cvSeasons = []
                        holdOutSeasons = cvSeasons+[int(predSeason)]
                        #get a mask for the training folds, the CV fold, and the predictions
                        cvFoldInds = seasonInds["year"].apply(lambda x: x in cvSeasons)
                        cvFoldInds = cvFoldInds[cvFoldInds==True].index.tolist()
                        predFoldInds = seasonInds["year"].apply(lambda x: x == int(predSeason))
                        predFoldInds = predFoldInds[predFoldInds==True].index.tolist()
                        trainFoldInds = seasonInds["year"].apply(lambda x: not x in holdOutSeasons)
                        trainFoldInds = trainFoldInds[trainFoldInds==True].index.tolist()
                        
                        noiseSeason1 = np.random.rand(Xseason1train.shape[0],Xseason1train.shape[1],
                                                      Xseason1train.shape[2])*noiseFactor1
                        noiseSeason2 = np.random.rand(Xseason2train.shape[0],Xseason2train.shape[1],
                                                      Xseason2train.shape[2])*noiseFactor1
                        noiseStats1 = np.random.rand(Xstats1train.shape[0],Xstats1train.shape[1])*noiseFactor2
                        noiseStats2 = np.random.rand(Xstats1train.shape[0],Xstats1train.shape[1])*noiseFactor2
                        
                        #subset to training data and add noise
                        Xseason1traintemp = Xseason1train[trainFoldInds]+noiseSeason1[trainFoldInds]
                        Xseason2traintemp = Xseason2train[trainFoldInds]+noiseSeason2[trainFoldInds]
                        Xstats1traintemp = Xstats1train[trainFoldInds]+noiseStats1[trainFoldInds]
                        Xstats2traintemp = Xstats2train[trainFoldInds]+noiseStats2[trainFoldInds]
                        ytraintemp = ytrain[trainFoldInds]
                        #subset to cv data and add noise
                        Xseason1cvtemp = Xseason1train[cvFoldInds]+noiseSeason1[cvFoldInds]
                        Xseason2cvtemp = Xseason2train[cvFoldInds]+noiseSeason2[cvFoldInds]
                        Xstats1cvtemp = Xstats1train[cvFoldInds]+noiseStats1[cvFoldInds]
                        Xstats2cvtemp = Xstats2train[cvFoldInds]+noiseStats2[cvFoldInds]  
                        ycvtemp = ytrain[cvFoldInds]
                        #subset to test data 
                        Xseason1testtemp = Xseason1train[predFoldInds]
                        Xseason2testtemp = Xseason2train[predFoldInds]
                        Xstats1testtemp = Xstats1train[predFoldInds]
                        Xstats2testtemp = Xstats2train[predFoldInds]
                        ytesttemp = ytrain[predFoldInds]
                        gameInfoPreds = gameInfo[predFoldInds]
                        
                        
                
                        
                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                            ModelCheckpoint(modelPath+'cv_'+predSeason+'_rnn', 
                                            monitor='val_loss', save_best_only=True, verbose=0),
                        ]        
                        trainInput = [Xseason1traintemp, Xseason2traintemp, Xstats1traintemp,Xstats2traintemp]
                        cvInput = [Xseason1cvtemp,Xseason2cvtemp,Xstats1cvtemp,Xstats2cvtemp]
                        trainOutput = [ytraintemp,ytraintemp]
                        cvOutput = [ycvtemp,ycvtemp]
                        testInput = [Xseason1testtemp, Xseason2testtemp, Xstats1testtemp,Xstats2testtemp]
                        testOutput = []
                        
                        if runType == "CV":
                            cbs = model.fit(trainInput,trainOutput,nb_epoch=epochs,validation_data=(cvInput, cvOutput),callbacks=callbacks,verbose=0)
                            model = load_model(modelPath+'cv_'+predSeason+'_rnn')
                        else:
                            cbs = model.fit(trainInput,trainOutput,nb_epoch=numEpochs)
                
                        
                        if runType == "CV":
                            if phase == "phase1":
                                p1 = model.predict(testInput)[0]
                                score = metrics.log_loss(ytesttemp,p1)
                            elif phase == "phase2":
                                score = min(cbs.history['val_Prediction_loss'])
                                numEpochs = len(cbs.history['val_Prediction_loss'])-patience
                            toappend = [timeSteps,iterations,noiseFactor1,noiseFactor2,predSeason,numEpochs,score]
                            print(toappend)
                            cvList.append(toappend)
                        
                        #prep the data for submission
                        else:
                            submissionTourney = testTourney[testTourney["Season"]==predSeason]  
                            Xs1sub,Xs2sub,Xst1sub,Xst2sub,ysub,sIsub,gIsub = getData(allTeams,submissionTourney,predict=True)
                            Xs1sub -= means
                            Xs2sub -= means
                            Xs1sub /= stds+np.ones((stds.shape))*0.1
                            Xs2sub /= stds+np.ones((stds.shape))*0.1
                            
                            Xst1sub -= statmeans
                            Xst2sub -= statmeans
                            Xst1sub /= statstds+np.ones((statstds.shape))*0.1
                            Xst2sub /= statstds+np.ones((statstds.shape))*0.1        
                            subInput = [Xs1sub,Xs2sub,Xst1sub,Xst2sub]
                            subOutput = [ysub,ysub]
                            
                            
                            preds = model.predict(subInput)[1]
                            gIsub = pd.DataFrame(gIsub,columns=["Year","t1","t2"])
                            gIsub["pred"] = preds
                            gIsub["team1"] = gIsub["t1"].apply(lambda x: x.split("_")[0])
                            gIsub["team2"] = gIsub["t2"].apply(lambda x: x.split("_")[0])
                            gIsub["id"] = gIsub.apply(lambda x: x.loc["Year"]+"_"+x.loc["team1"]+"_"+x.loc["team2"],axis=1)
                            groupedPreds = gIsub.groupby(["id"]).mean()["pred"]
                            if predsForSubmission is None:
                                predsForSubmission = groupedPreds
                            else:
                                predsForSubmission = predsForSubmission.append(groupedPreds)

else:
    model   = loadThatModel("../model/mmRNN")
    
    
if runType != "CV":
    pfs = pd.DataFrame(predsForSubmission).reset_index() 
    pfs["pred2"] = pfs["pred"].apply(lambda x: np.max(np.min(0.98,x),0.02))
    pfs.to_csv("../preds/"+phase+"/predictions_auto.csv",index=False)
    pfs["year"] = pfs["id"].apply(lambda x: int(x.split('_')[0]))
    pfs["t1"] = pfs["id"].apply(lambda x:IDtoPom[int(x.split('_')[1])])
    pfs["t2"] = pfs["id"].apply(lambda x:IDtoPom[int(x.split('_')[2])])
    fin = pd.merge(pfs,pom[["Rank"]],left_on=["t1","year"],right_index=True,suffixes=('','_t1'))
    fin = pd.merge(fin,pom[["Rank"]],left_on=["t2","year"],right_index=True,suffixes=('','_t2'))
    fin.to_csv("../preds/"+phase+"/predictions_inspection_auto.csv",index=False)
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