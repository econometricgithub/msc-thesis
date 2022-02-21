#importing necessary library
import numpy as np
import pandas as pd
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
import csv
desired_width=800
pd.set_option('display.width',desired_width)
pd.set_option('display.max_columns', 40)
'''

db_connection_str = 'mysql+pymysql://b5f3d54f1ae019:2577be25@us-cdbr-east-05.cleardb.net/heroku_8b3d35cf0aa7c8f'
db_connection = create_engine(db_connection_str)
df = pd.read_sql('SELECT * FROM answers', con=db_connection)'''





df=pd.read_csv("Dataset/raw.csv")

df["d_informational_use"]=0
df["d_communication"]=0
df["d_new_friends"]=0
df["d_share"]=0
df["d_study"]=0
df["d_entertainment"]=0

'''for i in df['reasoning_using']:
    str =list(i.split(","))
    for list1 in str:
        path_csv = "Dataset/MyCsv.csv"
        with open(path_csv, mode='a', newline='') as f:
            fields = ['Name', 'Branch', 'Year', 'CGPA', 'dd', 'cc']
            csv_writer = csv.writer(f)
            csv_writer.writerow(fields)


df = pd.read_csv("D:/Kyaw Thet/Thesis/MainProject/Dataset_2/raw_data.csv")
print(df.head())

#To generate the online expressive political participation
df["OEP"]=((df["OEP_1"]/5)+(df["OEP_2"]/5)+(df["OEP_3"]/5)+(df["OEP_4"]/5))/4

#To generate the offline expressive political participation
df["OFEP"]=(df["OFEP_1"]+ df["OFEP_2"]+ df["OFEP_3"] +df["OFEP_4"])/4

#To generate the Social Interactional Usage
df["SIU"]=(df["SIU_1"]/5+(df["SIU_2"]/5)+(df["SIU_3"]/5)+(df["SIU_INGroup"]/4))/4

#To generate the three usage
df["three_usage"]=((df["informational_use"]/5)+ df["SIU"]+ (df["entertainment_use"]/5))/3

#print(df.dropna())
#print(df.isnull())

df.to_csv("D:/Kyaw Thet/Thesis/MainProject/Dataset/data_cleaning.csv")
print(df.head())'''

df = pd.read_csv("Dataset/cleaning.csv")
#Normalization political Efficacy
#data['political_efficacy']=data['political_efficacy']/data['political_efficacy'].max()

#Normalization informational use
df['informational_use']=df['informational_use']/df['informational_use'].max()
#Normalization Entertainmetn use
df['entertainment_use']=df['entertainment_use']/df['entertainment_use'].max()
print(df)
df.to_csv('Dataset/data_cleaning.csv')