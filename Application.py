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


## Pour k=6

with open('./Model/model_6.pickle', 'rb') as file:
    model_6 = pickle.load(file)


with open('./Model/pca_6.pickle', 'rb') as file:
    pca_6 = pickle.load(file)

with open('./Model/classes_6.pickle', 'rb') as file:
    classes_6 = pickle.load(file)

with open('./Model/columns.pickle', 'rb') as file:
    columns = pickle.load(file)

model , pca , classes = model_3 ,pca_3,classes_3

# print(classes)
st.title("Application de detection des maladies en fonction des symptomes")
# st.write("Cette application permet de predire une maladie en fonction des symptomes du patient")

# Ajoutez votre code Streamlit ici

st.sidebar.header("Application de prediction de maladie")


symptomes = {
  "douleur abdominale": "abdominal_pain",
  "menstruation anormale": "abnormal_menstruation",
  "acidité": "acidity",
  "insuffisance hépatique aiguë": "acute_liver_failure",
  "altération du sensorium": "altered_sensorium",
  "anxiété": "anxiety",
  "douleur au dos": "back_pain",
  "douleur à l'estomac": "belly_pain",
  "points noirs": "blackheads",
  "inconfort de la vessie": "bladder_discomfort",
  "cloque": "blister",
  "sang dans les crachats": "blood_in_sputum",
  "selles sanglantes": "bloody_stool",
  "vision floue et déformée": "blurred_and_distorted_vision",
  "essoufflement": "breathlessness",
  "ongles cassants": "brittle_nails",
  "ecchymoses": "bruising",
  "miction brûlante": "burning_micturition",
  "douleur thoracique": "chest_pain",
  "frissons": "chills",
  "mains et pieds froids": "cold_hands_and_feets",
  "coma": "coma",
  "congestion": "congestion",
  "constipation": "constipation",
  "sensation constante d'urine": "continuous_feel_of_urine",
  "éternuements continus": "continuous_sneezing",
  "toux": "cough",
  "crampes": "cramps",
  "urine sombre": "dark_urine",
  "déshydratation": "dehydration",
  "dépression": "depression",
  "diarrhée": "diarrhoea",
  "taches dischromiques": "dischromic_patches",
  "distension de l'abdomen": "distention_of_abdomen",
  "vertiges": "dizziness",
  "lèvres sèches et picotements": "drying_and_tingling_lips",
  "thyroïde hypertrophiée": "enlarged_thyroid",
  "faim excessive": "excessive_hunger",
  "contacts extraconjugaux": "extra_marital_contacts",
  "antécédents familiaux": "family_history",
  "rythme cardiaque rapide": "fast_heart_rate",
  "fatigue": "fatigue",
  "surcharge liquide": "fluid_overload",
  "odeur fétide de l'urine": "foul_smell_of_urine",
  "mal de tête": "headache",
  "forte fièvre": "high_fever",
  "douleur de l'articulation de la hanche": "hip_joint_pain",
  "antécédents de consommation d'alcool": "history_of_alcohol_consumption",
  "appétit accru": "increased_appetite",
  "indigestion": "indigestion",
  "ongles inflammatoires": "inflammatory_nails",
  "démangeaisons internes": "internal_itching",
  "taux de sucre irrégulier": "irregular_sugar_level",
  "irritabilité": "irritability",
  "irritation anale": "irritation_in_anus",
  "douleur articulaire": "joint_pain",
  "douleur au genou": "knee_pain",
  "manque de concentration": "lack_of_concentration",
  "léthargie": "lethargy",
  "perte d'appétit": "loss_of_appetite",
  "perte d'équilibre": "loss_of_balance",
  "perte d'odorat": "loss_of_smell",
  "malaise général": "malaise",
  "légère fièvre": "mild_fever",
  "changements d'humeur": "mood_swings",
  "raideur des mouvements": "movement_stiffness",
  "expectorations muqueuses": "mucoid_sputum",
  "douleur musculaire": "muscle_pain",
  "fonte musculaire": "muscle_wasting",
  "faiblesse musculaire": "muscle_weakness",
  "nausées": "nausea",
  "douleur au cou": "neck_pain",
  "éruptions cutanées nodulaires": "nodal_skin_eruptions",
  "obésité": "obesity",
  "douleur derrière les yeux": "pain_behind_the_eyes",
  "douleur pendant les selles": "pain_during_bowel_movements",
  "douleur dans la région anale": "pain_in_anal_region",
  "douleur lors de la marche": "painful_walking",
  "palpitations": "palpitations",
  "passage de gaz": "passage_of_gases",
  "taches dans la gorge": "patches_in_throat",
  "glaires": "phlegm",
  "polyurie": "polyuria",
  "veines saillantes sur le mollet": "prominent_veins_on_calf",
  "visage et yeux gonflés": "puffy_face_and_eyes",
  "boutons remplis de pus": "pus_filled_pimples",
  "transfusion sanguine": "receiving_blood_transfusion",
  "injections non stériles": "receiving_unsterile_injections",
  "plaie rouge autour du nez": "red_sore_around_nose",
  "taches rouges sur le corps": "red_spots_over_body",
  "rougeur des yeux": "redness_of_eyes",
  "agitation": "restlessness",
  "nez qui coule": "runny_nose",
  "expectorations rouillées": "rusty_sputum",
  "desquamation": "scurring",
  "frissons": "shivering",
  "dépôt argenté sur la peau": "silver_like_dusting",
  "pression des sinus": "sinus_pressure",
  "desquamation de la peau": "skin_peeling",
  "éruption cutanée": "skin_rash",
  "discours confus": "slurred_speech",
  "petites bosses sur les ongles": "small_dents_in_nails",
  "mouvements de rotation": "spinning_movements",
  "taches d'urine": "spotting_urination",
  "raideur du cou": "stiff_neck",
  "saignement de l'estomac": "stomach_bleeding",
  "douleur à l'estomac": "stomach_pain",
  "yeux enfoncés": "sunken_eyes",
  "transpiration": "sweating",
  "ganglions lymphatiques enflés": "swelled_lymph_nodes",
  "gonflement des articulations": "swelling_joints",
  "gonflement de l'estomac": "swelling_of_stomach",
  "vaisseaux sanguins gonflés": "swollen_blood_vessels",
  "extrémités enflées": "swollen_extremeties",
  "jambes enflées": "swollen_legs",
  "irritation de la gorge": "throat_irritation",
  "aspect toxique (typhoïde)": "toxic_look_(typhos)",
  "ulcères sur la langue": "ulcers_on_tongue",
  "instabilité": "unsteadiness",
  "troubles visuels": "visual_disturbances",
  "vomissements": "vomiting",
  "larmoiement": "watering_from_eyes",
  "faiblesse des membres": "weakness_in_limbs",
  "faiblesse d'un côté du corps": "weakness_of_one_body_side",
  "prise de poids": "weight_gain",
  "perte de poids": "weight_loss",
  "croûte jaune suintante": "yellow_crust_ooze",
  "urine jaune": "yellow_urine",
  "jaunissement des yeux": "yellowing_of_eyes",
  "peau jaunâtre": "yellowish_skin",
  "démangeaisons": "itching",
  "dischromic _patches" : "dischromic _patches",
  "spotting_ urination" : "spotting_ urination"
}


options = list(symptomes.keys())
# print(options)

resultat = {value: 0 for value in list(columns)}

# print(len(symptomes))

liste_K = [3,6]

selected_k_value = st.sidebar.selectbox('Sélectionnez le nombre de centre', liste_K)
selected_options = st.sidebar.multiselect('Sélectionnez les symptomes de la maladie', columns)

if selected_k_value !=3:
    model,pca,classes = model_6,pca_6,classes_6

if len(selected_options)>0 : 
    for d in selected_options :
        resultat[d] = 1



if st.sidebar.button('Valider'):
    df = pd.DataFrame(resultat, index=[0],columns=columns)
    df = df.fillna(0)
 
    pca_value = pca.transform(df)
    dim_labels  = [f'PC{i+1}' for i in range(selected_k_value)]
    Pca_test = pd.DataFrame(pca_value, index=(df).index, columns=dim_labels)
    predict_row = Pca_test.iloc[0:1,]
    result = model.predict(pd.DataFrame(predict_row))
    # st.write("La liste des maladies possibles")

    st.header("Listes des maladies possibles")

    print(result)

    if result[0] ==0:
        data = pd.DataFrame({'Maladies': classes["class_0"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))
    if result[0] ==1:
        data = pd.DataFrame({'Maladies': classes["class_1"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))
    if result[0] ==2:
        data = pd.DataFrame({'Maladies': classes["class_2"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))
    if result[0] ==3:
        data = pd.DataFrame({'Maladies': classes["class_3"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))
    if result[0] ==4:
        data = pd.DataFrame({'Maladies': classes["class_2"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))
    if result[0] ==5:
        data = pd.DataFrame({'Maladies': classes["class_5"]})
        st.table(data.style.set_properties(**{'max-height': '200px', 'overflow-y': 'auto'}))


st.sidebar.write('Listes des symptomes  :', selected_options)






