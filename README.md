# Exploration des données automobiles

Tableau de bord descriptif du jeu de données historique Auto MPG. Il permet de
filtrer les véhicules par origine, année et nombre de cylindres, d'afficher les
moyennes d'efficacité, de consommation et de puissance, et de comparer poids,
consommation et origines. Aucun modèle ML ou prédiction n'est utilisé.

## Lancer l'application

Depuis la racine du projet, installez le projet et ses dépendances, puis lancez :

```console
python -m pip install .
streamlit run src/project1/app.py
```

L'application télécharge le CSV Auto MPG public depuis le dépôt
[seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/mpg.csv)
au premier chargement ; une connexion Internet est nécessaire.

## Tester les transformations

```console
python -m unittest discover -s tests
```