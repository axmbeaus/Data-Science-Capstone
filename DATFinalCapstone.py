# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:39:18 2023

@author: ab553103
"""
import numpy as np
import pandas as pd
from sklearn import linear_model
import math
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg

 

def linRegPtsEst(reg, df, away):
    size = len(df.index)
    if(away):
        dfcoef = df[['Away goals','Away S','Away ST', 'Away F', 'Away C', 'Away Y', 'Away R']]
    else:
        dfcoef = df[['Home goals','Home S','Home ST', 'Home F', 'Home C', 'Home Y', 'Home R']]
    est = reg.intercept_
    for x in range(reg.coef_.size):
        mean = (sum(dfcoef.iloc[:,x])/size)
        est += (reg.coef_[x]*mean)
    return est
    

def multiLineReg(team, df, dfpred):
    regr = linear_model.LinearRegression()
    dfxh = df[['Home goals','Home S','Home ST', 'Home F', 'Home C', 'Home Y', 'Home R']]
    dfyh = df['Home Pts']
    homeReg = regr.fit(dfxh, dfyh)
    hEst = linRegPtsEst(homeReg, dfpred[dfpred.team == team], False)
    dfxa = df[['Away goals','Away S','Away ST', 'Away F', 'Away C', 'Away Y', 'Away R']]
    dfya = df['Away Pts']
    awayReg = regr.fit(dfxa, dfya)
    aEst = linRegPtsEst(awayReg, dfpred[dfpred.team == team], True)
    return aEst+hEst
    
def forestPtsEst(forest, df, away):
    if(away):
        data = df[['Away goals','Away S','Away ST', 'Away F', 'Away C', 'Away Y', 'Away R']]
    else:
        data = df[['Home goals','Home S','Home ST', 'Home F', 'Home C', 'Home Y', 'Home R']]
    results = sum(forest.predict(data))
    return results

def forestReg(team, df, dfpred):
    forReg = RandomForestRegressor(n_estimators=100, random_state=0)
    homeX = df[['Home goals','Home S','Home ST', 'Home F', 'Home C', 'Home Y', 'Home R']]
    homeY = df['Home Pts']
    forReg.fit(homeX, homeY)
    hest = forestPtsEst(forReg, dfpred[dfpred.team == team], False)
    awayX = df[['Away goals','Away S','Away ST', 'Away F', 'Away C', 'Away Y', 'Away R']]
    awayY = df['Away Pts']
    forReg.fit(awayX, awayY)
    aest = forestPtsEst(forReg, dfpred[dfpred.team == team], True)
    return hest+aest

def xgbPtsEst(xgb, df, away):
    if(away):
        data = df[['Away goals','Away S','Away ST', 'Away F', 'Away C', 'Away Y', 'Away R']]
    else:
        data = df[['Home goals','Home S','Home ST', 'Home F', 'Home C', 'Home Y', 'Home R']]
    results = sum(xgb.predict(data))
    return results

def xgboostReg(team, df, dfpred):
    xgb_r = xg.XGBRegressor(objective ='reg:linear', n_estimators = 10, seed = 123)
    homeX = df[['Home goals','Home S','Home ST', 'Home F', 'Home C', 'Home Y', 'Home R']]
    homeY = df['Home Pts']
    xgb_r.fit(homeX, homeY)
    hest = xgbPtsEst(xgb_r, dfpred[dfpred.team == team], False)
    awayX = df[['Away goals','Away S','Away ST', 'Away F', 'Away C', 'Away Y', 'Away R']]
    awayY = df['Away Pts']
    xgb_r.fit(awayX, awayY)
    aest = xgbPtsEst(xgb_r, dfpred[dfpred.team == team], True)
    return hest+aest

def errCalulator(dftrain, dftest, dfnames):
    pts = [0]*dfnames.size
    accPts = [0]*dfnames.size
    err = 0
    for x in range(len(dfnames)):
        pts[x] = math.floor((multiLineReg(dfnames.iloc[x,0], dftrain, dftest)+xgboostReg(dfnames.iloc[x,0], dftrain, dftest)+forestReg(dfnames.iloc[x,0], dftrain, dftest))/3)
    for x in range(len(dfnames)):
        accPts[x] = dftest.iloc[x,17] + dftest.iloc[x,18]
    for x in range(len(dfnames)):
        err = err + abs(pts[x] - accPts[x])
    return err

def resComparison(dftrain, dftest, dfnames):
    points = [0]*20
    for x in range(20):
        points[x] = math.floor((multiLineReg(dfnames.iloc[x,0], dftrain, dftest)+xgboostReg(dfnames.iloc[x,0], dftrain, dftest)+forestReg(dfnames.iloc[x,0], dftrain, dftest))/3)
    points = pd.DataFrame(points, columns=['Points'])
    prediction = pd.concat([dfnames, points], axis = 1)
    prediction = prediction.sort_values(by=['Points'], ascending = False)
    prediction = prediction.reset_index()
    prediction = prediction.drop(columns = ['index'])
    print(prediction)
    acc = [0]*20
    for x in range(20):
        acc[x] = dftest.iloc[x,17] + dftest.iloc[x,18]
    acc = pd.DataFrame(acc, columns = ['Points'])
    accRes = pd.concat([dfnames, acc], axis = 1)
    accRes = accRes.sort_values(by = ['Points'], ascending = False)
    accRes = accRes.reset_index()
    accRes = accRes.drop(columns = ['index'])
    print(accRes)

def main():
    #dataSet = pd.read_csv('SeasonData.csv', sep=',', na_filter = "false")
    season10 = pd.read_csv('9-10.csv', sep=',', na_filter = "false")
    season11 = pd.read_csv('10-11.csv', sep=',', na_filter = "false")
    dataNames1 = pd.DataFrame(season11['team'].unique(), columns= ['Team'])
    season12 = pd.read_csv('11-12.csv', sep=',', na_filter = "false")
    dataNames2 = pd.DataFrame(season12['team'].unique(), columns= ['Team'])
    season13 = pd.read_csv('12-13.csv', sep=',', na_filter = "false")
    dataNames3 = pd.DataFrame(season13['team'].unique(), columns= ['Team'])
    season14 = pd.read_csv('13-14.csv', sep=',', na_filter = "false")
    dataNames4 = pd.DataFrame(season14['team'].unique(), columns= ['Team'])
    season15 = pd.read_csv('14-15.csv', sep=',', na_filter = "false")
    dataNames5 = pd.DataFrame(season15['team'].unique(), columns= ['Team'])
    season16 = pd.read_csv('15-16.csv', sep=',', na_filter = "false")
    dataNames6 = pd.DataFrame(season16['team'].unique(), columns= ['Team'])
    season17 = pd.read_csv('16-17.csv', sep=',', na_filter = "false")
    dataNames7 = pd.DataFrame(season17['team'].unique(), columns= ['Team'])
    season18 = pd.read_csv('17-18.csv', sep=',', na_filter = "false")
    dataNames8 = pd.DataFrame(season18['team'].unique(), columns= ['Team'])
    season19 = pd.read_csv('18-19.csv', sep=',', na_filter = "false")
    dataNames9 = pd.DataFrame(season19['team'].unique(), columns= ['Team'])
    dfnames = [dataNames1, dataNames2, dataNames3, dataNames4, dataNames5, dataNames6, dataNames7, dataNames8, dataNames9]
    seasons = [season10, season11, season12, season13, season14, season15, season16, season17, season18, season19]
    error = 0
    size = len(dfnames)
    for x in range(size):
        error = error + errCalulator(seasons[x], seasons[x+1], dfnames[x])
    print("Average Error per model: ")
    avgErr = error/size
    print(avgErr)
    print("Average Error per team: ")
    teamErr = avgErr/20
    print(teamErr)
    
    print("Comparison of preds vs acctual")
    resComparison(season10, season11, dataNames1)
    resComparison(season15, season16, dataNames6)
    

    
if __name__ == "__main__":
    main()