#importing necessary library
import numpy as np
import pandas as pd
import csv
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template,flash, request
from wtforms import Form, StringField, validators, SubmitField, SelectField
from bioinfokit.analys import get_data, stat
#Importing SQLAlcheny
from flask_sqlalchemy import SQLAlchemy
#from custom_validators import height_validator, weight_validator
from myModules import results_summary_to_dataframe, result_one,results_two, main_fun, reg_metric, linerity,normality,homoscedasticity,multicollinearity
#input are just variables - not as in the short form [ Y, X ]

from sqlalchemy import create_engine
import pymysql
desired_width=800
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns', 40)

#db_connection_str = 'mysql+pymysql://b5f3d54f1ae019:2577be25@us-cdbr-east-05.cleardb.net/heroku_8b3d35cf0aa7c8f'
#db_connection = create_engine(db_connection_str)
'''
df=pd.read_csv("Dataset/cleaning.csv")
print(df.head())
X=df[['informational_use','SIU','entertainment_use','gender','age','education','income','location']]
y=df['political_efficacy']
print(y)
X = sm.add_constant(X)
mlr_results = sm.OLS(y, X).fit()
print(mlr_results.summary())

import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("Dataset/cleaning.csv")
width_of_panel = 4
height_of_panel = 3
d = 1000
plt.figure(figsize=(width_of_panel, height_of_panel), dpi=d)
x = df['political_efficacy']
y = df['OEP']
x, y = np.meshgrid(x, y)
z = df['OFEP']

plt.contourf(x, y, z)'''
'''
df=pd.read_csv("Dataset/cleaning.csv")
import matplotlib.pyplot as plt
plt.style.use ('seaborn-white')
import numpy as np




x = df['informational_use']
y = df['SIU']

X,Y=np.meshgrid(x,y)
z=df['political_efficacy']
plt.contour(X,Y,z,colors='black')


import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("Dataset/cleaning.csv")
x = df['informational_use']
y = df['SIU']
Z = df['political_efficacy']'''
df=pd.read_csv("Dataset/new_data.csv")

import seaborn as sns
#sns.displot(df, x="SIU", y="political_efficacy", kind="")
#lt.show()

#sns.displot(df, x="SIU", y="informational_use", hue="political_efficacy", kind="kde")

#plt.show()

print(df['political_efficacy'].describe())

#sns.displot(df, x="SIU", y="informational_use", hue="political_efficacy", kind="kde")

#plt.show()
print(df['informational_use'].describe())
df['information_use2']=df['informational_use']/5
print(df.head())

#sns.displot(df, x="SIU", y="information_use2", hue="political_efficacy", kind="kde")

#plt.show()


#################################################################################################
'''import pandas as pd
from sklearn import datasets
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML
df = pd.read_csv("Dataset/data_cleaning.csv")
X1 = df[['political_efficacy','gender','age','education','income','location']]
y1 = df['OEP']
X1 = sm.add_constant(X1)
mlr_results1 = sm.OLS(y1, X1).fit()

X2 = df[['political_efficacy','gender','age','education','income','location']]
y2 = df['OFEP']
X2 = sm.add_constant(X2)
mlr_results2 = sm.OLS(y2, X2).fit()

stargazer = Stargazer([mlr_results1,mlr_results2])
with open('templates/summary2.html', 'w') as f:
    f.write(stargazer.render_html())


###################################################################
df = pd.read_csv("Dataset/data_cleaning.csv")
plt.scatter(df['informational_use'], df['OEP'], color='black')
plt.xlabel('informational_use')
plt.ylabel('OEP')
#plt.show()


plt.scatter(df['informational_use'],  df['OEP'],  color='black')
plt.plot(df['informational_use'],  df['OEP'], color='blue', linewidth=3)

plt.xlabel('informational_use')
plt.ylabel('OEP')
##plt.show()'''

############################################
#Correlation Metric

'''import seaborn as sns
import matplotlib.pyplot as plt
df=df[['informational_use','OEP','entertainment_use']]
correlation_matrix = df.corr().round(2)
plt.figure(figsize=(14, 12))
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()'''


def line(x, y):
    """Fit a line in a scatter
    based on slope and intercept"""

    slope, intercept = np.polyfit(x, y, 1)
    line_values = [slope * i + intercept for i in x]
    plt.plot(x, line_values, 'b')

df=pd.read_csv("Dataset/data_cleaningcsv")
plt.scatter(df['informational_use'],df['OEP'],
            color='lightsalmon')

plt.title('Test Scores by Student Teacher Ratio')
plt.xlabel('Student Teacher Ratio')
plt.ylabel('Average Test Scores')
line(df['informational_use'], df['OEP'])

#plt.savefig('scat4.png')
plt.show()