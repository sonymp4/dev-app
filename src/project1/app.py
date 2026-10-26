from __future__ import annotations

from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import streamlit as st

from project1.analysis import filter_data, prepare_data

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
ORIGIN_LABELS = {"usa": "États-Unis", "europe": "Europe", "japan": "Japon"}
ORIGIN_COLORS = {"usa": "#ef443b", "europe": "#2867c7", "japan": "#8dc4f5"}

st.set_page_config(page_title="Exploration automobile", page_icon="🚗", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    with urlopen(DATA_URL, timeout=20) as response:
        return pd.read_csv(response, na_values=["?"])


st.title("🚗 Exploration des données automobiles")
st.caption("Dataset historique Auto MPG — analyse descriptive, sans modèle IA ni prédiction.")
st.info(
    "Une valeur mpg élevée indique une meilleure efficacité. "
    "Une consommation en L/100 km faible signifie une consommation moindre."
)

try:
    cars = prepare_data(load_data())
except (URLError, TimeoutError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Impossible de charger ou de lire le jeu de données Auto MPG : {error}")
    st.stop()

with st.sidebar:
    st.header("Filtres")
    origin_options = sorted(cars["origin"].dropna().unique().tolist())
    selected_origin_label = st.selectbox(
        "Origine",
        ["Toutes"] + [ORIGIN_LABELS.get(value, value.title()) for value in origin_options],
    )
    selected_origin = (
        None
        if selected_origin_label == "Toutes"
        else next(
            value
            for value in origin_options
            if ORIGIN_LABELS.get(value, value.title()) == selected_origin_label
        )
    )

    cylinder_options = sorted(cars["cylinders"].astype(int).unique().tolist())
    selected_cylinders = st.pills(
        "Nombre de cylindres",
        cylinder_options,
        default=cylinder_options,
        format_func=lambda value: f"{value} cylindres",
        selection_mode="multi",
    )

    year_min = int(cars["year"].min())
    year_max = int(cars["year"].max())
    selected_years = st.slider(
        "Année du modèle",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )
    minimum_mpg = st.slider(
        "Efficacité minimale (mpg)",
        min_value=float(cars["mpg"].min()),
        max_value=float(cars["mpg"].max()),
        value=float(cars["mpg"].min()),
        step=1.0,
        help="N’afficher que les véhicules dont l’efficacité est au moins égale à ce seuil.",
    )

filtered = filter_data(
    cars,
    selected_origin,
    selected_years,
    selected_cylinders,
    minimum_mpg,
)
st.header("1 — Indicateurs et données filtrées")
st.write(f"**{len(filtered)}** observations sur {len(cars)}")

metric_columns = st.columns(3)
if filtered.empty:
    for column, label in zip(
        metric_columns,
        ("Efficacité moyenne", "Consommation moyenne", "Puissance moyenne"),
    ):
        column.metric(label, "—")
else:
    metric_columns[0].metric("Efficacité moyenne", f"{filtered['mpg'].mean():.1f} mpg")
    metric_columns[1].metric(
        "Consommation moyenne",
        f"{filtered['consumption_l_100km'].mean():.1f} L/100 km",
    )
    metric_columns[2].metric(
        "Puissance moyenne",
        f"{filtered['horsepower'].mean():.1f} hp",
    )
    st.caption(
        "Les moyennes ignorent les valeurs manquantes. "
        "La consommation moyenne est calculée à partir des conversions individuelles de mpg."
    )

display_columns = [
    "name",
    "origin",
    "year",
    "cylinders",
    "mpg",
    "consumption_l_100km",
    "horsepower",
    "weight_kg",
    "acceleration",
]
available_display_columns = [column for column in display_columns if column in filtered]
st.dataframe(
    filtered[available_display_columns].rename(
        columns={
            "name": "Véhicule",
            "origin": "Origine",
            "year": "Année",
            "cylinders": "Cylindres",
            "mpg": "Efficacité (mpg)",
            "consumption_l_100km": "Consommation (L/100 km)",
            "horsepower": "Puissance (hp)",
            "weight_kg": "Poids (kg)",
            "acceleration": "Accélération",
        }
    ),
    width="stretch",
    hide_index=True,
)

st.header("2 — Choisir un véhicule")
if filtered.empty:
    st.info("Aucun véhicule ne correspond aux filtres sélectionnés.")
else:
    vehicle_options = filtered.reset_index(drop=True)
    selected_vehicle_index = st.selectbox(
        "Véhicule",
        options=vehicle_options.index.tolist(),
        format_func=lambda index: (
            f"{vehicle_options.loc[index, 'name']} — "
            f"{int(vehicle_options.loc[index, 'year'])}"
        ),
    )
    selected_vehicle = vehicle_options.loc[[selected_vehicle_index], available_display_columns]
    st.dataframe(
        selected_vehicle.rename(
            columns={
                "name": "Véhicule",
                "origin": "Origine",
                "year": "Année",
                "cylinders": "Cylindres",
                "mpg": "Efficacité (mpg)",
                "consumption_l_100km": "Consommation (L/100 km)",
                "horsepower": "Puissance (hp)",
                "weight_kg": "Poids (kg)",
                "acceleration": "Accélération",
            }
        ),
        width="stretch",
        hide_index=True,
    )

st.header("3 — Comparaisons graphiques")
chart_columns = st.columns(2)
if filtered.empty:
    st.info("Les graphiques apparaîtront dès qu’au moins un véhicule correspondra aux filtres.")
else:
    weight_chart = px.scatter(
        filtered,
        x="weight_kg",
        y="consumption_l_100km",
        color="origin",
        color_discrete_map=ORIGIN_COLORS,
        hover_name="name",
        hover_data={"year": True, "horsepower": True, "weight_kg": ":.0f"},
        labels={
            "weight_kg": "Poids (kg)",
            "consumption_l_100km": "Consommation (L/100 km)",
            "origin": "Origine",
            "year": "Année",
            "horsepower": "Puissance (hp)",
        },
        title="Relation entre le poids et la consommation",
    )
    chart_columns[0].plotly_chart(weight_chart, width="stretch")

    origin_summary = (
        filtered.groupby("origin", as_index=False)
        .agg(
            vehicles=("name", "count"),
            average_consumption=("consumption_l_100km", "mean"),
            average_horsepower=("horsepower", "mean"),
        )
        .sort_values("origin")
    )
    origin_summary["origin_label"] = origin_summary["origin"].map(
        lambda value: ORIGIN_LABELS.get(value, value.title())
    )
    origin_chart = px.bar(
        origin_summary,
        x="origin_label",
        y="vehicles",
        color="origin",
        color_discrete_map=ORIGIN_COLORS,
        hover_data={
            "average_consumption": ":.1f",
            "average_horsepower": ":.1f",
            "origin": False,
        },
        labels={
            "origin_label": "Origine",
            "vehicles": "Nombre de véhicules",
            "average_consumption": "Consommation moyenne (L/100 km)",
            "average_horsepower": "Puissance moyenne (hp)",
            "origin": "Origine",
        },
        title="Nombre de véhicules par origine",
    )
    chart_columns[1].plotly_chart(origin_chart, width="stretch")
    st.caption(
        "Survolez les barres pour comparer également la consommation et la puissance moyennes "
        "de chaque origine."
    )
