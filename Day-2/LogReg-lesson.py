import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

df_1y = pd.read_csv("/Volumes/HOLOCRON/FRESH-START-DATA/STATS_PRES/data/FirstYear-post.csv", skipinitialspace=True)
df_Uy = pd.read_csv("/Volumes/HOLOCRON/FRESH-START-DATA/STATS_PRES/data/Uy-mixed-effects.csv", skipinitialspace=True)


#####
##FUNCTION
#####
def final_data(dfk, cols_toreplace):

	for col in cols_toreplace:
		for it in list(set(dfk[col].values.tolist())):
			a=it
			dfk[col] = dfk[col].replace([a], list(set(dfk[col].values.tolist())).index(a))

	return dfk

def final_dataF1LONG(cols, dfk):
        data=[]

        counter=0
        for it in dfk[cols].values.tolist():
                data.append([counter, it[0], it[1]])
                data.append([counter, it[0], it[2]])
                counter+=0

        return pd.DataFrame(np.array(data).reshape(-1, 3), columns=['ID', 'rate', 'response'])


def long_stats(dfk, columns):
	x=final_dataLONG(columns, dfk)
	est=smf.mixedlm('rate ~ response', x, groups=x['ID'])
	est2=est.fit()
	print(est2.summary())

#####
##IMPLEMENTATION
#####
lr=LogReg(solver='newton-cg', multi_class='multinomial')

df2=df_1y.replace([np.nan], 'NA')
df3=df_Uy.replace([np.nan], 'NA')

df1y=final_data(df2, list(df2)[1:])
dfUy=final_data(df3, list(df3)[1:])

###
#Q5 RESULTS
###

#1y Students
print('===1y Q5===')
x=df1y[['Q5B', 'Q5C']]
y=df1y['Q5A']
x2=sm.add_constant(x)
est=smf.GLM(y, x2)
est2=est.fit()
print(est2.summary())

#Uy students
print('===Uy Q5===')
est=smf.mixedlm('Q5A ~ Q5B + Q5C', dfUy, groups=dfUy['AY'])
est2=est.fit()
print(est2.summary(), '\n')


###
#Q6 RESULTS
###

#1y Students
print('===1y Q6===')
x=df1y[['Q6B', 'Q6C']]
y=df1y['Q6A']
x2=sm.add_constant(x)
est=sm.GLM(y, x2)
est2=est.fit()
print(est2.summary(), '\n')

#Uy students
print('===Uy Q6===')
est=smf.mixedlm('Q6A ~ Q6B + Q6C', dfUy, groups=dfUy['AY'])
est2=est.fit()
print(est2.summary(), '\n')

###
#Q7 RESULTS
###

#1y Students
print('===1y Q7===')
x=df1y[['Q7B', 'Q7C']]
y=df1y['Q7A']
x2=sm.add_constant(x)
est=sm.GLM(y, x2)
est2=est.fit()
print(est2.summary(), '\n')

#Uy students
print('===Uy Q7===')
est=smf.mixedlm('Q7A ~ Q7B + Q7C', dfUy, groups=dfUy['AY'])
est2=est.fit()
print(est2.summary(), '\n')
