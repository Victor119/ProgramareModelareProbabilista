import matplotlib.pyplot as plt

import numpy as np

import pymc3 as pm
import arviz as az

import random
import math

from scipy import stats

#heads probabilitatea de a fi al doilea muncitor

#problema 1
def unbiased_coin_toss():
    # Generate a random number from 0 to 1
    x = random.uniform(0, 1)

    if x >= 0.6:
        # Heads for True
        return True
    else:
        # Tails for False
        return False

def mediePrimulSauAlDoileaMuncitor():
    prob = []
    prob2 = []

    listaMuncitor = []
    dataSet = []
    dataSet1 = []
    # dataSet1 este setul de date de intrare pentru al doilea muncitor
    dataSet2 = []
    # dataSet2 este setul de date de intrare pentru primul muncitor

    matrice = []

    for i in range(10000):
    # Fiecare experiment are 1 extragere
        N = 1
        results = []
        results2 = []

    # Facem un experiment si il salvam in lista

        result = unbiased_coin_toss()
        result2 = not(result)
        value = i

        results.append(result)
        dataSet.append(result)
        dataSet1.append(value)

        results2.append(result2)
        dataSet.append(result2)
        dataSet2.append(value)

        n_heads = sum(results)
        p_heads = n_heads / N
        prob.append(p_heads)

        nHeads = sum(results2)
        pHeads = nHeads / N
        prob2.append(pHeads)


    # p_heads_MC are media celui de al doilea muncitor
    p_heads_MC = sum(prob) / 10000

    # prob are media primului muncitor
    pHeadsMc = sum(prob2) / 10000

    listaMuncitor.append(p_heads_MC)
    listaMuncitor.append(pHeadsMc)
    listaMuncitor.append(dataSet)
    matrice.append(listaMuncitor)
    matrice.append(dataSet1)
    matrice.append(dataSet2)
    return matrice

def sumDiferente(set, medie):
    summ = 0
    for i in range(0, len(set)):
        dif = int(set[i]) - medie
        summ = summ + pow(dif, 2)
    return summ

def problema1():
    medieMuncitori = mediePrimulSauAlDoileaMuncitor()

    dataSetMuncitor2 = medieMuncitori[1]

    dataSetMuncitor1 = medieMuncitori[2]

    medieMuncitor1 = medieMuncitori[0][0]
    medieMuncitor2 = medieMuncitori[0][1]
    print("Media probabilitatilor sa fie primul muncitor {:.3f}".format(medieMuncitor1))
    print("Media probabilitatilor sa fie al doilea muncitor {:.3f}".format(medieMuncitor2))

    #dataSetMuncitor1 e sample data set caci e submultime din dataSet-ul total deci avem impartit la n-1
    #dataSetMuncitor2 e la fel ca la dataSetMuncitor1
    numberDataPoint = len(dataSetMuncitor1) - 1
    variantaMuncitor1 = sumDiferente(dataSetMuncitor1, medieMuncitor1) / numberDataPoint
    deviatiaMuncitor1 = math.sqrt(variantaMuncitor1)


    numberDataPoint = len(dataSetMuncitor2) - 1
    variantaMuncitor2 = sumDiferente(dataSetMuncitor2, medieMuncitor2) / numberDataPoint
    deviatiaMuncitor2 = math.sqrt(variantaMuncitor2)


    print("Deviatia primului muncitor {:.3f}".format(deviatiaMuncitor1))
    print("Deviatia celui de al doilea muncitor {:.3f}".format(deviatiaMuncitor2))

    x = stats.norm.rvs(medieMuncitor1, deviatiaMuncitor1, size=10000) # Distributie normala cu media=medieMuncitor1 si deviatie standard = deviatiaMuncitor1 , 10000 samples
    y = stats.uniform.rvs(-1, 2, size=10000) # Distributie uniforma intre -1 si 1, 10000 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
    z = x+y # Compunerea prin insumare a celor 2 distributii

    az.plot_posterior({'x':x,'y':y,'z':z}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
    plt.show()

    x = stats.norm.rvs(medieMuncitor2, deviatiaMuncitor2, size=10000)  # Distributie normala cu media=medieMuncitor2 si deviatie standard = deviatiaMuncitor1 , 10000 samples
    y = stats.uniform.rvs(-1, 2, size=10000)  # Distributie uniforma intre -1 si 1, 10000 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
    z = x + y  # Compunerea prin insumare a celor 2 distributii

    az.plot_posterior({'x': x, 'y': y,'z': z})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
    plt.show()

#Problema 2
#Moneda1 este nemasluita
#vom considera valoarea True pentru stema iar False pentru cealalta
def coinTossMoneda1():
    # Generate a random number from 0 to 1
    x = random.uniform(0, 1)

    if x >= 0.5:
        # Stema for True
        return True
    else:
        # Ban for False
        return False

def cointTossMoneda2():
    # Generate a random number from 0 to 1
    x = random.uniform(0, 1)

    if x <= 0.3:
        # Stema for True
        return True
    else:
        # Ban for False
        return False


def numarRezultatePosibile():
    listMedii = []
    #lista cu toate aruncarile de monezi si salveaza in valori True sau False
    dataSet = []

    matrice = []

    #aici salvam de cate ori a aparut in 10 aruncari
    numarAparitiiSS = []
    numarAparitiiSB = []
    numarAparitiiBS = []
    numarAparitiiBB = []

    for i in range(100):

        resultSS = []
        resultSB = []
        resultBS = []
        resultBB = []

        N = 10

        for j in range(10):
            result = coinTossMoneda1()
            result2 = cointTossMoneda2()
            value = i

            if(result == True and result2 == True):
                resultSS.append(value)

            if (result == True and result2 == False):
                resultSB.append(value)

            if (result == False and result2 == True):
                resultBS.append(value)

            if (result == False and result2 == False):
                resultBB.append(value)


        nHeadsSet1 = len(resultSS)
        pHeads1 = nHeadsSet1 / N
        numarAparitiiSS.append(pHeads1)

        nHeadsSet2 = len(resultSB)
        pHeads2 = nHeadsSet2 / N
        numarAparitiiSB.append(pHeads2)

        nHeadsSet3 = len(resultBS)
        pHeads3 = nHeadsSet3 / N
        numarAparitiiBS.append(pHeads3)

        nHeadsSet4 = len(resultBB)
        pHeads4 = nHeadsSet4 / N
        numarAparitiiBB.append(pHeads4)

    #media aparitiilor SS in 100 aruncari
    mediaSS = sum(numarAparitiiSS) / 100

    #media aparitiilor SB in 100 aruncari
    mediaSB = sum(numarAparitiiSB) / 100

    # media aparitiilor BS in 100 aruncari
    mediaBS = sum(numarAparitiiBS) / 100

    # media aparitiilor BB in 100 aruncari
    mediaBB = sum(numarAparitiiBB) / 100

    listMedii.append(mediaSS)
    listMedii.append(mediaSB)
    listMedii.append(mediaBS)
    listMedii.append(mediaBB)
    matrice.append(listMedii)
    matrice.append(numarAparitiiSS)
    matrice.append(numarAparitiiSB)
    matrice.append(numarAparitiiBS)
    matrice.append(numarAparitiiBB)
    return matrice

def problema3():
    matrice = numarRezultatePosibile()

    dataSetSS = matrice[1]
    dataSetSB = matrice[2]
    dataSetBS = matrice[3]
    dataSetBB = matrice[4]

    medieSS = matrice[0][0]
    medieSB = matrice[0][1]
    medieBS = matrice[0][2]
    medieBB = matrice[0][3]

    print("Media probabilitatilor sa fie SS {:.3f}".format(medieSS))
    print("Media probabilitatilor sa fie SB {:.3f}".format(medieSB))
    print("Media probabilitatilor sa fie BS {:.3f}".format(medieBS))
    print("Media probabilitatilor sa fie BB {:.3f}".format(medieBB))

    numberDataPoint = len(dataSetSS)
    variatiaSS = sumDiferente(dataSetSS, medieSS) / numberDataPoint
    deviatiaSS = math.sqrt(variatiaSS)


    numberDataPoint = len(dataSetSB)
    variatiaSB = sumDiferente(dataSetSB, medieSB) / numberDataPoint
    deviatiaSB = math.sqrt(variatiaSB)

    numberDataPoint = len(dataSetSB)
    variatiaBS = sumDiferente(dataSetBS, medieBS) / numberDataPoint
    deviatiaBS = math.sqrt(variatiaBS)

    numberDataPoint = len(dataSetBB)
    variatiaBB = sumDiferente(dataSetBB, medieBB) / numberDataPoint
    deviatiaBB = math.sqrt(variatiaBB)


    print("Deviatia esantionului SS {:.3f}".format(deviatiaSS))
    print("Deviatia esantionului SB {:.3f}".format(deviatiaSB))
    print("Deviatia esantionului BS {:.3f}".format(deviatiaBS))
    print("Deviatia esantionului BB {:.3f}".format(deviatiaBB))

    x = stats.norm.rvs(medieSS, deviatiaSS, size=100) # Distributie normala cu media=medieSS si deviatie standard = deviatiaSS , 100 samples
    y = stats.uniform.rvs(-1, 2, size=100) # Distributie uniforma intre -1 si 1, 100 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
    z = x+y # Compunerea prin insumare a celor 2 distributii

    az.plot_posterior({'x':x,'y':y,'z':z}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
    plt.show()

    x = stats.norm.rvs(medieSB, deviatiaSB, size=100)  # Distributie normala cu media=medieSB si deviatie standard = deviatiaSB , 100 samples
    y = stats.uniform.rvs(-1, 2, size=100)  # Distributie uniforma intre -1 si 1, 100 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
    z = x + y  # Compunerea prin insumare a celor 2 distributii

    az.plot_posterior({'x': x, 'y': y,'z': z})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
    plt.show()

    x = stats.norm.rvs(medieBS, deviatiaBS, size=100)  # Distributie normala cu media=medieBS si deviatie standard = deviatiaBS , 100 samples
    y = stats.uniform.rvs(-1, 2, size=100)  # Distributie uniforma intre -1 si 1, 100 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
    z = x + y  # Compunerea prin insumare a celor 2 distributii

    az.plot_posterior({'x': x, 'y': y,'z': z})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
    plt.show()

    x = stats.norm.rvs(medieBB, deviatiaBB, size=100)  # Distributie normala cu media=medieBB si deviatie standard = deviatiaBB , 100 samples
    y = stats.uniform.rvs(-1, 2, size=100)  # Distributie uniforma intre -1 si 1, 100 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
    z = x + y  # Compunerea prin insumare a celor 2 distributii

    az.plot_posterior({'x': x, 'y': y,'z': z})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
    plt.show()




#problema2
# O simulare consta in generarea unei valori uniforme standard U(0,1) apoi in functie de aceasta valoare a unei valori
# care urmeaza o distributie exponentiala 4.3 sau 4.2 sau 5.2 sau 5.3 rezultatul este adaugat unei valori distribuite
# cu valoarea 0.25

# valoare = 4.3 daca 0 <= u < 0.25
# valoare = 4.2 daca 0.25 <= u <0.5
# valoare = 5.2 daca 0.5 <= u < 0.8
# valoare = 5.3 daca 0.8 <= u < 1
# T = X + valoare

def functiaU(u):
    timpY = 4.3
    timpW = 4.2
    timpZ = 5.2
    timpV = 5.3

    if u >= 0 and u < 0.25:
        return timpY
    if u >=0.25 and u < 0.5:
        return timpW
    if u >=0.5 and u < 0.8:
        return timpZ
    if u >= 0.8 and u <1:
        return timpV


def problema2():
    timpX = 0.25
    result = []

    #facem 100 de teste
    n = 100

    for i in range(n):
        # Generate a random number from 0 to 1
        x = random.uniform(0, 1)
        valoare = timpX + functiaU(x)
        result.append(valoare)

    #medieSir are timpul mediu de asteptare
    medieSir = sum(result) / n

    numberDataPoint = len(result)
    variatia = sumDiferente(result, medieSir) / numberDataPoint
    deviatia = math.sqrt(variatia)

    print("media timpului in milisecunde =", medieSir)
    print("deviatia standard dupa calcularea timpului mediu de asteptare =", deviatia)

    x = stats.norm.rvs(medieSir, deviatia, size=100)  # Distributie normala cu media=medieSir si deviatie standard = deviatia , 100 samples
    y = stats.uniform.rvs(-1, 2, size=100)  # Distributie uniforma intre -1 si 1, 100 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
    z = x + y  # Compunerea prin insumare a celor 2 distributii

    az.plot_posterior({'x': x, 'y': y,'z': z})  # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
    plt.show()



    # !!! probabilitatea va fi mereu mai mare de 3 milisecunde caci numai latenta dintre server si client are o distributie de procesare
    # exponentiala de 0.25 milisecunde la care se adauga dupa aceea procesarea oricarei cecereri dureaza mai mult de 4 milisecunde

#Ordine grafice: primul este pentru muncitor1 dupa muncitor2
problema1()

problema2()

#Ordine grafice: primul este pentru SS dupa SB, BS, BB
problema3()
