import operator
from collections import Counter

import numpy as np
import pandas as  pd
import random
import math

def construir_arbol(instancias, etiquetas, profundidad_actual, profundidad_max, criterion):
    # ALGORITMO RECURSIVO para construcción de un árbol de decisión binario. 
    # Suponemos que estamos parados en la raiz del árbol y tenemos que decidir cómo construirlo. 
    ganancia, pregunta = encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterion)

    # Criterio de corte: ¿Hay ganancia? ¿llegamos a la profundidad máxima?
    if ganancia < 0.05 or profundidad_actual == profundidad_max:
        #  Si no hay ganancia en separar, no separamos. 
        return Hoja(etiquetas)
    else:
        profundidad_actual = profundidad_actual + 1

        # Si hay ganancia en partir el conjunto en 2
        instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta,
                                                                                                          instancias,
                                                                                                          etiquetas)
        # partir devuelve instancias y etiquetas que caen en cada rama (izquierda y derecha)

        # Paso recursivo (consultar con el computador más cercano)
        sub_arbol_izquierdo = construir_arbol(instancias_cumplen, etiquetas_cumplen, profundidad_actual,
                                              profundidad_max, criterion)
        sub_arbol_derecho = construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen, profundidad_actual,
                                            profundidad_max, criterion)
        # los pasos anteriores crean todo lo que necesitemos de sub-árbol izquierdo y sub-árbol derecho

        # sólo falta conectarlos con un nodo de decisión:
        return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)


# Definición de la estructura del árbol. 
class Hoja:
    #  Contiene las cuentas para cada clase (en forma de diccionario)
    def __init__(self, etiquetas):
        self.cuentas = dict(Counter(etiquetas))


class Nodo_De_Decision:
    # Un Nodo de Decisión contiene preguntas y una referencia al sub-árbol izquierdo y al sub-árbol derecha
    def __init__(self, pregunta, sub_arbol_izquierdo, sub_arbol_derecho):
        self.pregunta = pregunta
        self.sub_arbol_izquierdo = sub_arbol_izquierdo
        self.sub_arbol_derecho = sub_arbol_derecho


# Definición de la clase "Pregunta"
class Pregunta:
    def __init__(self, atributo, valor):
        self.atributo = atributo
        self.valor = valor

    def cumple(self, instancia):
        # Devuelve verdadero si la instancia cumple con la pregunta
        return instancia[self.atributo] < self.valor

    def __repr__(self):
        return "¿Es el valor para {} menor a {}?".format(self.atributo, self.valor)


def gini(etiquetas):
    pNo, pSi = 0, 0

    if 1 in etiquetas:
        pSi = np.sum(etiquetas == 1) / len(etiquetas)
    if 0 in etiquetas:
        pNo = np.sum(etiquetas == 0) / len(etiquetas)
        
    #if not (pNo + pSi == 1):
    #    print(etiquetas)
    #    print(pSi, pNo)

    impureza = 1 - (pSi * pSi) - (pNo * pNo)
    return impureza


def entropy(etiquetas):
    p = 0
    if 0 in etiquetas:
        p = np.sum(etiquetas == 0) / len(etiquetas)
    return (-1) * p * np.log2(p) - (1 - p) * np.log2(1 - p)


def ganancia_gini(etiquetas_rama_izquierda, etiquetas_rama_derecha):
    giniD = gini(etiquetas_rama_derecha)
    giniI = gini(etiquetas_rama_izquierda)

    lenD = len(etiquetas_rama_derecha)
    lenI = len(etiquetas_rama_izquierda)
    n    = lenD + lenI

    giniAtributo = (lenD * giniD + lenI * giniI) / n
    giniOriginal = gini(np.append(etiquetas_rama_izquierda, etiquetas_rama_derecha))

    return giniOriginal - giniAtributo


def ganancia_entropy(etiquetas_rama_izquierda, etiquetas_rama_derecha):
    entropyOriginal = entropy(np.append(etiquetas_rama_izquierda, etiquetas_rama_derecha))

    entropyD = entropy(etiquetas_rama_derecha)
    entropyI = entropy(etiquetas_rama_izquierda)

    lenD = len(etiquetas_rama_derecha)
    lenI = len(etiquetas_rama_izquierda)
    n = lenD + lenI

    entropyAtributo = (lenD * entropyD + lenI * entropyI) / n

    return entropyOriginal - entropyAtributo


def partir_segun(pregunta, instancias, etiquetas):
    ind_cumplen    = instancias[pregunta.atributo] <  pregunta.valor
    ind_no_cumplen = instancias[pregunta.atributo] >= pregunta.valor

    instancias_cumplen    = instancias[ind_cumplen]
    instancias_no_cumplen = instancias[ind_no_cumplen]
    etiquetas_cumplen     = np.array(etiquetas)[ind_cumplen]
    etiquetas_no_cumplen  = np.array(etiquetas)[ind_no_cumplen]

    return instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen


def encontrar_mejor_atributo_y_corte(instancias, etiquetas, criterion):
    max_ganancia = -1
    mejor_pregunta = None

    for columna in instancias.columns:
        valores = set(instancias[columna])
        #valores = range(min(instancias[columna]), max(instancias[columna]), int(round(len(instancias[columna])/2)))
        #for valor in valores:
        # for valor in random.sample(list(instancias[columna]), math.ceil(len(list(instancias[columna]))*0.1)):
        lim_inf  = min(instancias[columna])
        lim_sup  = max(instancias[columna])
        cantidad = math.ceil(len(list(instancias[columna]))*0.4)
        for valor in np.random.uniform(lim_inf, lim_sup, cantidad):
            # Probando corte para atributo y valor
            pregunta = Pregunta(columna, valor)
            _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
            
            # Si una rama me queda vacía, no me interesa usar este corte
            if (len(etiquetas_rama_izquierda) == 0) or (len(etiquetas_rama_derecha) == 0):
                continue

            if   criterion == "gini":
                ganancia = ganancia_gini(etiquetas_rama_izquierda, etiquetas_rama_derecha)
            elif criterion == "entropy":
                ganancia = ganancia_entropy(etiquetas_rama_izquierda, etiquetas_rama_derecha)

            if ganancia > max_ganancia:
                max_ganancia   = ganancia
                mejor_pregunta = pregunta

    return max_ganancia, mejor_pregunta


def predecir(arbol, x_t):
    if type(arbol).__name__ == 'Hoja':
        maxLabel   = max(arbol.cuentas.items(), key=operator.itemgetter(1))
        prediccion = maxLabel[0]
    else:
        if x_t[arbol.pregunta.atributo] < arbol.pregunta.valor:
            prediccion = predecir(arbol.sub_arbol_izquierdo, x_t)
        else:
            prediccion = predecir(arbol.sub_arbol_derecho, x_t)

    return prediccion


def predecir_prob(arbol, x_t):
    if type(arbol).__name__ == 'Hoja':
        total = sum([v for v in arbol.cuentas.values()])
        prediccion = arbol.cuentas.get(1,0)/total
    else:
        if x_t[arbol.pregunta.atributo] < arbol.pregunta.valor:
            prediccion = predecir_prob(arbol.sub_arbol_izquierdo, x_t)
        else:
            prediccion = predecir_prob(arbol.sub_arbol_derecho, x_t)
    return prediccion


class MiClasificadorArbol():
    def __init__(self, X_columns, criterion="gini", profundidad_max=3):
        self.arbol = None
        self.columnas = X_columns
        self.criterion = criterion
        self.profundidad_max = profundidad_max

    def fit(self, X_train, y_train):
        self.arbol = construir_arbol(pd.DataFrame(X_train, columns=self.columnas), y_train, 0, self.profundidad_max,
                                     self.criterion)
        return self

    def predict(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t], columns=self.columnas).iloc[0]
            prediction = predecir(self.arbol, x_t_df)
            predictions.append(prediction)
        return predictions

    def predict_proba(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t], columns=self.columnas).iloc[0]
            prediction = predecir_prob(self.arbol, x_t_df)
            predictions.append(prediction)
        return predictions

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)
        return accuracy
