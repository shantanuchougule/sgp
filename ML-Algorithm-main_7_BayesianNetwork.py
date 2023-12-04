#program for implementation of bayesian network

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
import matplotlib.pyplot as plt

#load the data
data=pd.read_csv("heart.csv")
data=data.sort_values(by="age").reset_index().drop(columns=["index"])

#model
model=BayesianNetwork([('age','trestbps'),('trestbps','chol'),('chol','restecg'),('restecg','result')])

#fit the model
model.fit(data,estimator=MaximumLikelihoodEstimator)

#inference
infer=VariableElimination(model)

#query
q=infer.query(variables=['result'],evidence={'age':34})

#print the result
print(q)


