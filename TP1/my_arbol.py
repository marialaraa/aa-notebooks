import numpy as np
from collections import Counter
import operator
import pandas as  pd

def construir_arbol(instancias, etiquetas, profundidad_actual, profundidad_max):
    # ALGORITMO RECURSIVO para construcción de un árbol de decisión binario. 
    # Suponemos que estamos parados en la raiz del árbol y tenemos que decidir cómo construirlo. 
    ganancia, pregunta = encontrar_mejor_atributo_y_corte(instancias, etiquetas)
    
    # Criterio de corte: ¿Hay ganancia? ¿llegamos a la profundidad máxima?
    if ganancia == 0 or profundidad_actual == profundidad_max :
        #  Si no hay ganancia en separar, no separamos. 
        return Hoja(etiquetas)
    else: 
    	profundidad_actual += 1
        # Si hay ganancia en partir el conjunto en 2
        instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen = partir_segun(pregunta, instancias, etiquetas)
        # partir devuelve instancias y etiquetas que caen en cada rama (izquierda y derecha)

        # Paso recursivo (consultar con el computador más cercano)
        sub_arbol_izquierdo = construir_arbol(instancias_cumplen, etiquetas_cumplen, profundidad_actual, profundidad_max)
        sub_arbol_derecho   = construir_arbol(instancias_no_cumplen, etiquetas_no_cumplen, profundidad_actual, profundidad_max)
        # los pasos anteriores crean todo lo que necesitemos de sub-árbol izquierdo y sub-árbol derecho
        
        # sólo falta conectarlos con un nodo de decisión:
        return Nodo_De_Decision(pregunta, sub_arbol_izquierdo, sub_arbol_derecho)
    
# Definición de la estructura del árbol. 
class Hoja:
    #  Contiene las cuentas para cada clase (en forma de diccionario)
    #  Por ejemplo, {'Si': 2, 'No': 2}
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
    pSi = np.mean([x == 'Si' for x in etiquetas])    
    pNo = np.mean([x == 'No' for x in etiquetas])    
    
    impureza = 1 - pSi*pSi - pNo*pNo
    return impureza

def ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha):
    giniD = gini(etiquetas_rama_derecha)
    giniI = gini(etiquetas_rama_izquierda)
    
    lenD = len(etiquetas_rama_derecha)
    lenI = len(etiquetas_rama_izquierda)
    n = lenD + lenI
    
    giniAtributo = (lenD * giniD + lenI * giniI)/n
    giniOriginal = gini(np.append(etiquetas_rama_izquierda, etiquetas_rama_derecha))
    
    ganancia_gini = giniOriginal - giniAtributo
    return ganancia_gini


def partir_segun(pregunta, instancias, etiquetas):
    # Esta función debe separar instancias y etiquetas según si cada instancia cumple o no con la pregunta (ver método 'cumple')
    
    ind = instancias[pregunta.atributo] < pregunta.valor

    instancias_cumplen   = instancias[ind]
    etiquetas_cumplen    = np.array(etiquetas)[ind]    
    instancias_no_cumplen= instancias[-ind]
    etiquetas_no_cumplen = np.array(etiquetas)[-ind]    
    
    return instancias_cumplen, etiquetas_cumplen, instancias_no_cumplen, etiquetas_no_cumplen

def encontrar_mejor_atributo_y_corte(instancias, etiquetas):
    max_ganancia = 0
    mejor_pregunta = None
    for columna in instancias.columns:
        for valor in set(instancias[columna]):
            # Probando corte para atributo y valor
            pregunta = Pregunta(columna, valor)
            _, etiquetas_rama_izquierda, _, etiquetas_rama_derecha = partir_segun(pregunta, instancias, etiquetas)
   
            ganancia = ganancia_gini(instancias, etiquetas_rama_izquierda, etiquetas_rama_derecha)
            
            if ganancia > max_ganancia:
                max_ganancia = ganancia
                mejor_pregunta = pregunta            
    return max_ganancia, mejor_pregunta

def predecir(arbol, x_t):    
    if type(arbol).__name__ == 'Hoja':
        maxLabel = max(arbol.cuentas.items(), key=operator.itemgetter(1))
        prediccion = maxLabel[0]
    else:
        if x_t[arbol.pregunta.atributo] == arbol.pregunta.valor:
            prediccion = predecir(arbol.sub_arbol_izquierdo, x_t)
        else: 
            prediccion = predecir(arbol.sub_arbol_derecho, x_t)
    
    return prediccion
        
class MiClasificadorArbol(): 
    def __init__(self, X_columns):
        self.arbol = None
        self.columnas = X_columns
    
    def fit(self, X_train, y_train, profundidad_max):
        self.arbol = construir_arbol(pd.DataFrame(X_train, columns=self.columnas), y_train, 0, profundidad_max)
        return self
    
    def predict(self, X_test):
        predictions = []
        for x_t in X_test:
            x_t_df = pd.DataFrame([x_t], columns=self.columnas).iloc[0]
            prediction = predecir(self.arbol, x_t_df) 
            predictions.append(prediction)
        return predictions
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = sum(y_i == y_j for (y_i, y_j) in zip(y_pred, y_test)) / len(y_test)
        return accuracy


