# ==========================================================
# TABLEAU DE BORD IA - ANALYSE D’UN COFFEE SHOP
# Projet Data Science / Machine Learning
# ==========================================================

# ==========================================================
# IMPORT DES LIBRAIRIES
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from prophet import Prophet


# ==========================================================
# CONFIGURATION PAGE
# ==========================================================

st.set_page_config(
    page_title="Intelligence IA Coffee Shop",
    layout="wide",
    page_icon="☕"
)

st.title(" Dashboard Intelligent - Analyse Coffee Shop")


# ==========================================================
# STYLE DU DASHBOARD
# ==========================================================

st.markdown("""
<style>

body {
background-color:#0e1117;
}

.metric-container {
background:#111827;
padding:15px;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# CHARGEMENT DES DONNÉES
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_excel("coffee_sales.xlsx")

    return df


df = load_data()


# ==========================================================
# NETTOYAGE DES DONNÉES
# ==========================================================

# Conversion du prix unitaire
df["unit_price"] = df["unit_price"].astype(str).str.replace(",", ".").astype(float)

# Conversion date
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

# Conversion heure
df["transaction_time"] = df["transaction_time"].astype(str)

df["hour"] = pd.to_datetime(
    df["transaction_time"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour

# Calcul revenu
df["revenue"] = df["transaction_qty"] * df["unit_price"]


# ==========================================================
# SIDEBAR FILTRES
# ==========================================================

st.sidebar.header("Filtres")

store = st.sidebar.multiselect(
    "Magasin",
    df["store_location"].dropna().unique(),
    default=df["store_location"].dropna().unique()
)

df = df[df["store_location"].isin(store)]


# ==========================================================
# KPI PRINCIPAUX
# ==========================================================

total_revenue = df["revenue"].sum()

transactions = df["transaction_id"].count()

avg_basket = total_revenue / transactions if transactions > 0 else 0

top_product = df.groupby("product_detail")["revenue"].sum().idxmax()

c1,c2,c3,c4 = st.columns(4)

c1.metric(" Chiffre d'affaires",f"${total_revenue:,.0f}")
c2.metric(" Transactions",transactions)
c3.metric(" Panier moyen",f"${avg_basket:.2f}")
c4.metric(" Produit top",top_product)

st.divider()


# ==========================================================
# ANALYSE VENTES PAR HEURE
# ==========================================================

st.subheader(" Évolution des ventes par heure")

hour_sales = df.groupby("hour")["revenue"].sum().reset_index()

fig = px.line(
    hour_sales,
    x="hour",
    y="revenue",
    markers=True
)

st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# HEATMAP DES VENTES
# ==========================================================

st.subheader(" Heatmap des ventes")

pivot = df.pivot_table(
    values="revenue",
    index="hour",
    columns="store_location",
    aggfunc="sum"
).fillna(0)

fig, ax = plt.subplots()

sns.heatmap(pivot,cmap="YlOrRd",annot=True)

st.pyplot(fig)


# ==========================================================
# CLUSTERING PRODUITS
# ==========================================================

st.subheader(" Clustering des produits")

cluster = df.groupby("product_detail").agg({
    "revenue":"sum",
    "transaction_qty":"sum",
    "unit_price":"mean"
}).reset_index()

scaler = StandardScaler()

X = scaler.fit_transform(cluster[["revenue","transaction_qty","unit_price"]])

kmeans = KMeans(n_clusters=3,random_state=42)

cluster["cluster"] = kmeans.fit_predict(X)

fig = px.scatter(
    cluster,
    x="revenue",
    y="transaction_qty",
    color="cluster",
    hover_name="product_detail"
)

st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# DÉTECTION ANOMALIES
# ==========================================================

st.subheader(" Détection d'anomalies")

iso = IsolationForest(contamination=0.02,random_state=42)

df["anomaly"] = iso.fit_predict(df[["revenue","transaction_qty","unit_price"]])

fig = px.scatter(
    df,
    x="transaction_qty",
    y="revenue",
    color="anomaly",
    title="Transactions suspectes"
)

st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# PRÉVISION DES VENTES (PROPHET)
# ==========================================================

st.subheader(" Prévision des ventes (30 jours)")

forecast_df = df.groupby("transaction_date")["revenue"].sum().reset_index()

forecast_df.columns = ["ds","y"]

forecast_df = forecast_df.sort_values("ds")

model_prophet = Prophet()

model_prophet.fit(forecast_df)

future = model_prophet.make_future_dataframe(periods=30)

forecast = model_prophet.predict(future)

fig = px.line(
    forecast,
    x="ds",
    y="yhat",
    title="Prévision du chiffre d'affaires"
)

st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# RECOMMANDATION PRODUITS
# ==========================================================

# ==========================================================
# RECOMMANDATION PRODUITS (MARKET BASKET ANALYSIS)
# ==========================================================

st.subheader(" Recommandation de produits")

# Création des transactions (liste de produits par ticket)
transactions = (
    df.groupby("transaction_id")["product_detail"]
    .apply(list)
    .values.tolist()
)

# Encodage des transactions
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket_df = pd.DataFrame(te_array, columns=te.columns_)

# Calcul des itemsets fréquents
freq_items = apriori(
    basket_df,
    min_support=0.02,
    use_colnames=True
)

# Génération des règles d'association
rules = association_rules(
    freq_items,
    metric="lift",
    min_threshold=1.1
)

# Tri des meilleures recommandations
rules = rules.sort_values(by="lift", ascending=False)

# Conversion propre des frozenset en texte
rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

# Affichage des règles
st.write("###  Produits souvent achetés ensemble")

if len(rules) > 0:

    st.dataframe(
        rules[
            [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]
        ].head(10)
    )

else:

    st.warning("Pas assez de données pour générer des recommandations.")
# ==========================================================
# SEGMENTATION CLIENTS
# ==========================================================

st.subheader(" Segmentation clients")

customer = df.groupby("transaction_id").agg({
    "revenue":"sum",
    "transaction_qty":"sum"
})

scaler = StandardScaler()

X = scaler.fit_transform(customer)

kmeans = KMeans(n_clusters=3,random_state=42)

customer["segment"] = kmeans.fit_predict(X)

fig = px.scatter(
    customer,
    x="transaction_qty",
    y="revenue",
    color="segment"
)

st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# MODÈLE MACHINE LEARNING
# ==========================================================

st.subheader(" Modèle Machine Learning")

ml = pd.get_dummies(
    df[["transaction_qty","unit_price","hour","product_category"]],
    drop_first=True
)

target = df["revenue"]

X_train,X_test,y_train,y_test = train_test_split(
    ml,target,test_size=0.2,random_state=42
)

model = RandomForestRegressor(n_estimators=200)

model.fit(X_train,y_train)

prediction = model.predict(X_test)

score = model.score(X_test,y_test)

st.success(f"Précision du modèle (R²) : {round(score,2)}")


# comparaison réel vs prédit
result = pd.DataFrame({
    "Revenu réel":y_test,
    "Revenu prédit":prediction
})

fig = px.scatter(
    result,
    x="Revenu réel",
    y="Revenu prédit",
    trendline="ols",
    title="Comparaison Réel vs Prédit"
)

st.plotly_chart(fig,use_container_width=True)

# ==========================================================
# COMPARAISON DE MODÈLES MACHINE LEARNING
# ==========================================================

st.subheader(" Comparaison de plusieurs modèles Machine Learning")

# Variables explicatives
ml = pd.get_dummies(
    df[["transaction_qty","unit_price","hour","product_category"]],
    drop_first=True
)

# Variable cible
target = df["revenue"]

# Séparation train/test
X_train,X_test,y_train,y_test = train_test_split(
    ml,target,test_size=0.2,random_state=42
)

# ==========================================================
# IMPORT DES MODÈLES
# ==========================================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================================
# MODELES
# ==========================================================

models = {

"Linear Regression": LinearRegression(),

"Decision Tree": DecisionTreeRegressor(max_depth=10),

"Random Forest": RandomForestRegressor(n_estimators=200),

"Gradient Boosting": GradientBoostingRegressor()

}

results = []

# ==========================================================
# ENTRAINEMENT
# ==========================================================

for name, model in models.items():

    model.fit(X_train,y_train)

    score = model.score(X_test,y_test)

    results.append({
        "Modèle":name,
        "Score R²":round(score,3)
    })

results_df = pd.DataFrame(results)

# ==========================================================
# RESULTATS
# ==========================================================

st.write("### Performance des modèles")

st.dataframe(results_df)

# ==========================================================
# GRAPHIQUE COMPARATIF
# ==========================================================

fig = px.bar(
    results_df,
    x="Modèle",
    y="Score R²",
    title="Comparaison des modèles ML"
)

st.plotly_chart(fig,use_container_width=True)
# ==========================================================
# DATASET
# ==========================================================

st.subheader("Jeu de données")

st.dataframe(df)