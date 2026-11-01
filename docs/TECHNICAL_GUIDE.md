# Comprendre et expliquer le projet

## Pourquoi Random Forest ?

Une baseline tabulaire rapide sur CPU, capable de relations non linéaires, sans normalisation.
Elle ne modélise pas explicitement les séquences comme un LSTM. Cette différence fournit
une piste de comparaison future, pas une preuve de supériorité.

## Pourquoi séparer les moteurs ?

Un découpage par ligne pourrait répartir deux cycles voisins du même moteur entre apprentissage
et validation. Le modèle serait évalué sur une trajectoire partiellement vue. Le découpage groupé
garde chaque moteur dans une seule partition.

## Pourquoi F2 ?

Le rappel mesure les pannes ciblées détectées. Le F2 accorde plus de poids au rappel qu’à la
précision. C’est un compromis pédagogique ; un usage industriel exige une analyse des coûts.

## Lire les résultats

Sur 100 observations de test, 25 sont positives. La forêt en détecte 22, en manque 3 et déclenche
5 alertes à tort. Précision = 22/27 ; rappel = 22/25. Un score de 0,80 ne signifie pas qu’un moteur
réel a 80 % de chances de tomber en panne : les scores ne sont pas calibrés et les données sont simulées.

## Reproduire

1. Installer les dépendances dans un environnement virtuel Python 3.12.
2. Exécuter `python -m maintenance.train` depuis la racine.
3. Lire `reports/metrics.json` : versions, graine, partitions et empreintes.
4. Vérifier la matrice de confusion dans `reports/test_predictions.csv`.
5. Exécuter `python -m unittest discover -s tests -v`.
6. Ouvrir `python -m streamlit run app.py`.

## Points à maîtriser

- Construire la cible sans la mettre dans les variables d’entrée.
- Distinguer sélection sur validation et évaluation indépendante sur test.
- Distinguer importance de variable et causalité.
- Expliquer les limites d’une seule partition et les unités différentes de validation/test.
- Présenter clairement la provenance open source et le rôle de l’assistance IA.
