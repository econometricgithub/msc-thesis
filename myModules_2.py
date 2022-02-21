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

#a,b,c,d=model_1(OEP="OEP",SIU="SIU")
#e,f,g,h=model_1(OFEP="OFEP",SIU="political_efficacy")


def strager_2():
    import pandas as pd
    from sklearn import datasets
    import statsmodels.api as sm
    from stargazer.stargazer import Stargazer
    from IPython.core.display import HTML

    diabetes = datasets.load_diabetes()
    df = pd.DataFrame(diabetes.data)
    print(df)
    df.columns = ['Age', 'Sex', 'BMI', 'ABP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    df['target'] = diabetes.target

    est = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:4]])).fit()
    est2 = sm.OLS(endog=df['target'], exog=sm.add_constant(df[df.columns[0:6]])).fit()

    stargazer = Stargazer([est, est2])
    stargazer=stargazer.render_html()
    return stargazer

def stargazer(dv,idv=[]):
    import pandas as pd
    from sklearn import datasets
    import statsmodels.api as sm
    from stargazer.stargazer import Stargazer
    from IPython.core.display import HTML
    df = pd.read_csv("Dataset/data_cleaning.csv")
    X = df[idv]
    y = df[dv]
    X = sm.add_constant(X)
    mlr_results = sm.OLS(y, X).fit()
    stargazer = Stargazer([mlr_results])
    with open('templates/summary.html', 'w') as f:
        f.write(stargazer.render_html())


#Coefficients Plots
def coeff_2(dv,idv=[]):
    import statsmodels.api as sm
    df = pd.read_csv("Dataset/data_cleaning.csv")
    idv=df[idv]
    dv=df[dv]
    X = sm.add_constant(idv)
    y = dv
    X = sm.add_constant(X)
    mlr_results = sm.OLS(y, X).fit()
    # Extract the coefficients from the model
    df_coef = mlr_results.params.to_frame().rename(columns={0: 'coef'})
    # Visualize the coefficients
    ax = df_coef.plot.barh(figsize=(14, 7))
    ax.axvline(0, color='black', lw=1)
    plt.savefig("static/coeff_slr")

def coeff(x,y):
    import statsmodels.api as sm
    df = pd.read_csv("Dataset/data_cleaning.csv")
    X = x
    y = y
    X = sm.add_constant(X)
    mlr_results = sm.OLS(y, X).fit()
    # Extract the coefficients from the model
    df_coef = mlr_results.params.to_frame().rename(columns={0: 'coef'})
    # Visualize the coefficients
    ax = df_coef.plot.barh(figsize=(14, 7))
    ax.axvline(0, color='black', lw=1)
    plt.savefig("static/coeff.png")


