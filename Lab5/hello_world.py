import matplotlib.pyplot as plt

import numpy as np

import pymc3 as pm
import arviz as az

import pandas as pd
import re
import random
import math
import pgmpy

from scipy import stats
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#pica - inima neagra
# Defining a model
# un As
# 2 regi
# 2 regine

model = BayesianModel([('A', 'Q'), ('R', 'Q'), ('A', 'R'), ('R', 'R'), ('Q', 'Q')])
cpd_a = TabularCPD('A', 2, values=[[0.2], [0.8]])
cpd_r = TabularCPD('R', 2, values=[[0.4], [0.6]])
cpd_q = TabularCPD('Q', 2, values=[[0.9, 0.2], [0.1, 0.8]],
                  evidence=['R'], evidence_card=[2])

model.add_cpds(cpd_a, cpd_r, cpd_q)

model.get_cpds()

infer = VariableElimination(model)
result = infer.query(['A'], evidence={'A': 0, 'R': 1})
print(result)

