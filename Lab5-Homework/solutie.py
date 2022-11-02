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


joc_model = BayesianNetwork(
    [
        ("player1_mana", "player2_mana"),
        ("player1_mana", "Tura1"),
        ("player1_mana", "Tura3"),
        ("player2_mana", "Tura2"),
        ("Tura1", "Tura2"),
        ("Tura1", "Tura3"),
        ("Tura2", "Tura3"),
    ]
)

# facem cpd pentru prima carte
cpd_player_1_mana = TabularCPD(
    variable = "player1_mana",
    variable_card=5,
    values=[[0.2], [0.2], [0.2], [0.2], [0.2]],
)

# facem cpd pentru a doua carte
cpd_player_2_mana = TabularCPD(
    variable = "player2_mana",
    variable_card=5,
    values=[[0.2], [0.2], [0.2], [0.2], [0.2]],
)

# facem cpd petru Tura1
# primul jucator poate alege daca sa parieze sau sa astepte
cpd_Tura_1 = TabularCPD(
    variable = "Tura1",
    variable_card=2,
    values=[
        [0.4, 0.1, 0.5, 0.1, 0.8],
        [0.6, 0.9, 0.5, 0.9, 0.2]
    ],
    evidence= ["player1_mana"],
    evidence_card =[5]
)

# facem cpd pentru Tura2
cpd_Tura_2 = TabularCPD(
    variable = "Tura2",
    variable_card=2,
    values=[
        [0.7, 0.8, 0.7, 0.4, 0.6, 0.9, 0.5, 0.2, 0.3, 0.7],
        [0.3, 0.2, 0.3, 0.6, 0.4, 0.1, 0.5, 0.8, 0.7, 0.3]
    ],
    evidence= ["player2_mana", "Round1"],
    evidence_card =[5, 2]
)

# facem cpd pentru Tura3
cpd_Tura_3 = TabularCPD(
    variable = "Tura3",
    variable_card=2,
    values=[
        [0.3, 0.2, 0.6, 0.8, 0.5, 0.9, 0.1, 0.3, 0.2, 1, 0.6, 0.4, 0.1, 0.3, 0.1, 0.2],
        [0.7, 0.8, 0.4, 0.2, 0.5, 0.1, 0.9, 0.7, 0.8, 0, 0.4, 0.6, 0.9, 0.7, 0.9, 0.2]
    ],
    evidence= ["Tura1", "Tura2", "player1_mana"],
    evidence_card =[2, 2, 5]
)

joc_model.add_cpds(
    cpd_player_1_mana, cpd_player_2_mana, cpd_Tura_1, cpd_Tura_2, cpd_Tura_3
)

joc_model.check_model()

# Subpunct 2 -Laborator
# Interogarea modelului pentru a raspunde la intrebari:
infer = VariableElimination(joc_model)
print(infer.query(["Tura1"], evidence={"player1_mana": 1}))
print(infer.query(["Tura2"], evidence= {"player2_mana": 2, "Tura1": 0}))


# Subpunct 3 - Laborator
# Create a new game model
joc_model2 = BayesianNetwork(
    [
        ("mana1", "mana2"),
        ("mana1", "mana2", "Tura1"),
        ("mana1", "mana2", "Tura3"),
        ("mana2", "mana1", "Tura2"),
        ("Tura1", "Tura2"),
        ("Tura1", "Tura3"),
        ("Tura2", "Tura3"),
    ]
)

# facem cpd pentru prima carte
cpd_player_1_mana = TabularCPD(
    variable = "mana1",
    variable_card=5,
    values=[[0.2], [0.2], [0.2], [0.2], [0.2]],
)

# facem cpd pentru a doua carte
cpd_player_2_mana = TabularCPD(
    variable = "mana2",
    variable_card=5,
    values=[[0.2], [0.2], [0.2], [0.2], [0.2]],
)

# facem cpd petru Tura1
cpd_Tura_1 = TabularCPD(
    variable = "Tura1",
    variable_card=2,
    values=[
            [0.2, 0.3, 0.1, 0.1, 0],
            [0.9, 0.3, 0.7, 0, 0.6],
            [0.5, 0.1, 0.9, 0.7, 0.4],
            [0.6, 0.4, 0.7, 1, 0.8],
            [0.3, 1, 0.8, 0.2, 0.1]
    ],
    evidence= ["mana1", "mana2"],
    evidence_card =[5, 5]
)

# facem cpd petru Tura2
cpd_Tura_2 = TabularCPD(
    variable = "Tura2",
    variable_card=2,
    values=[
            [1, 0.6, 0.2, 0.7, 0.2, 0.6, 0.1, 0.2, 0, 0.3],
            [0.9, 0.8, 1, 0.3, 0.8, 0.1, 0.1, 0.3, 0.5, 0.9],
            [0.5, 0.6, 1, 0.5, 1, 0.6, 0.4, 0, 1, 0.3],
            [0.7, 0.3, 0.8, 0.3, 0.6, 0.9, 0.4, 1, 0.6, 0.4],
            [0.7, 0.6, 0.2, 0.1, 0.2, 0.4, 0.7, 0, 0.1, 0.9]
    ],
    evidence= ["Carte1", "mana2", "mana"],
    evidence_card =[5, 5, 2]
)

cpd_Tura_3 = TabularCPD(
    variable = "Tura3",
    variable_card=2,
    values=[
            [1, 1, 0.6, 0.5, 0.2, 0.2, 0.8, 0.2, 0.5, 0.4, 0.1, 1, 0.7, 0.1, 0.9],
            [0.5, 0.7, 0.2, 0.2, 0.7, 0.2, 1, 0.1, 0.7, 0.5, 0.7, 0.8, 0.9, 0.7, 0.7],
            [0.1, 0.7, 0.7, 0.2, 0.6, 1, 0.3, 0.6, 0.2, 0.2, 0.3, 0.5, 0.3, 0.9, 1],
            [0.8, 0.4, 1, 0.8, 0.6, 1, 1, 0.6, 0.5, 0.5, 0.1, 0.6, 0.5, 0.3, 0.5],
            [0.7, 0.4, 0.2, 0.8, 0.3, 0.6, 0.7, 0.2, 0.4, 0.8, 0.5, 0.3, 1, 0.9, 0.2]
    ],
    evidence= ["Tura1", "Tura2", "mana1", "mana2"],
    evidence_card =[2, 2, 5, 5]
)

joc_model2.add_cpds(
    cpd_player_1_mana, cpd_player_2_mana, cpd_Tura_1, cpd_Tura_2, cpd_Tura_3
)

joc_model2.check_model()
