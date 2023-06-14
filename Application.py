import altair
import streamlit as st
import pandas as pd
import pickle

import sklearn as sk 

# Chargement des models


# Pour k=3
with open('./Model/model.pickle', 'rb') as file:
    model_3 = pickle.load(file)


with open('./Model/pca.pickle', 'rb') as file:
    pca_3 = pickle.load(file)

with open('./Model/classes.pickle', 'rb') as file:
    classes_3 = pickle.load(file)

with open('./Model/columns.pickle', 'rb') as file:
    columns = pickle.load(file)

model , pca , classes = model_3 ,pca_3,classes_3

st.title("Systeme de recommandation de maladies à partir des symptomes")

# Ajoutez votre code Streamlit ici

st.sidebar.header("Choix des symptômes")

resultat = {value: 0 for value in list(columns)}


liste_K = [3,6]

selected_options = st.sidebar.multiselect('Sélectionnez les symptomes de la maladie', columns)



if len(selected_options)>0 : 
    for d in selected_options :
        resultat[d] = 1



if st.sidebar.button('Valider'):
    df = pd.DataFrame(resultat, index=[0],columns=columns)
    df = df.fillna(0)
 
    pca_value = pca.transform(df)
    dim_labels  = [f'PC{i+1}' for i in range(3)]
    Pca_test = pd.DataFrame(pca_value, index=(df).index, columns=dim_labels)
    predict_row = Pca_test.iloc[0:1,]
    result = model.predict(pd.DataFrame(predict_row))

    # print(result[0])

    st.header("Listes des maladies possibles")


    if result[0] ==0:
        data = pd.DataFrame({'Maladies': classes["class_0"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))
    if result[0] ==1:
        data = pd.DataFrame({'Maladies': classes["class_1"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))
    if result[0] ==2:
        data = pd.DataFrame({'Maladies': classes["class_2"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))
   
st.sidebar.divider()

st.sidebar.write("Auteurs :  KALMOGO Lucien & OUEDRAOGO Ousmane")






