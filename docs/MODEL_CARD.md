# Fiche du modèle — baseline Random Forest

## Usage et données

Classification binaire pédagogique sur des trajectoires simulées de moteurs. Aucune décision
médicale, aéronautique ou de maintenance réelle ne doit reposer sur cette démonstration.

25 entrées : cycle, 3 paramètres de fonctionnement, 21 capteurs. `engine_id`, RUL et labels
sont exclus des variables. Le lecteur contrôle les valeurs finies, les 26 colonnes du fichier,
les identifiants/cycles entiers positifs et l’unicité des couples moteur/cycle.

Cible : durée restante <= 30 cycles. Sur l’entraînement, elle vient de la fin de chaque
trajectoire. Sur le test, le fichier de vérité donne la durée restante après la dernière
observation. Les IDs 1..N correspondent aux lignes de ce fichier ; cette hypothèse est contrôlée.

## Protocole

75 moteurs d’apprentissage, 25 moteurs de validation, graine 42. Tous les cycles d’un moteur
restent dans la même partition. Forêt : 200 arbres, profondeur maximale 12, au moins 3
observations par feuille, pondération `balanced_subsample`. Aucun ajustement systématique
des hyperparamètres n’a été réalisé ; les arbres n’exigent pas de normalisation.

Le seuil est choisi parmi 0,05 à 0,95 par pas de 0,05 en maximisant le F2 sur validation.
En cas d’égalité, le seuil le plus élevé est retenu. Après réentraînement sur tous les moteurs
d’entraînement, ce seuil reste inchangé. Le test porte sur 100 autres trajectoires ; les IDs
numériques réutilisés entre fichiers ne représentent pas une identité globale partagée.

## Mesures et reproductibilité

`reports/metrics.json` contient les mesures, les IDs de partition, les paramètres, versions et
empreintes des données. `test_predictions.csv` contient une ligne par moteur de test.
La baseline naïve est un `DummyClassifier(strategy="prior")`, évalué au même seuil.

Le score de validation a servi à sélectionner le seuil : ce n’est pas un score final indépendant.
Les résultats LSTM historiques ne sont pas comparables sans réexécution à protocole identique.

## Limites et prochaines expériences

- Une seule partition et une seule graine : répéter une validation groupée avant généralisation.
- Les longues trajectoires pèsent davantage dans la sélection du seuil par cycle.
- Les cycles de validation et les observations finales du test ont des distributions différentes.
- Les scores ne sont pas des probabilités calibrées ; étudier la calibration sur validation.
- L’importance par impureté peut favoriser des variables corrélées et n’établit pas de causalité.
- Aucun transfert vers des machines réelles ou des équipements de dialyse n’a été évalué.
- Comparer un LSTM et la forêt sur exactement les mêmes unités et partitions serait une suite utile.
- De futurs choix de modèle nécessitent un nouveau test intact : ne pas optimiser sur les
  résultats de test déjà observés dans ce dépôt.
