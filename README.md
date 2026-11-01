# Predictive Maintenance Lab

**Maintenance prédictive · Python · Random Forest · Évaluation reproductible · Streamlit**

Adaptation de portfolio maintenue par **Younes Radouani**, élève ingénieur IA à l’EMSI Casablanca.
Fork de [Predictive-Maintenance-using-LSTM](https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM)
d’**Umberto Griffo**, sous [licence MIT](LICENSE). Le code et l’historique d’origine sont conservés.

## Le problème

À partir de mesures de capteurs, identifier les moteurs simulés qui atteindront la panne
 dans les **30 prochains cycles**. Le tableau de bord permet d’explorer les alertes,
les erreurs du modèle et les variables importantes.

**Démonstration sur un benchmark public**, sans déploiement industriel ni validation médicale.
Ce dépôt est indépendant du projet de stage SmartDialysis IA.

## Apports de cette version

| Élément | Implémentation ajoutée |
| --- | --- |
| Pipeline ML | Validation des données, cible explicite, Random Forest et baseline naïve |
| Validation | Moteurs séparés entre apprentissage et validation ; identifiants et cible exclus des entrées |
| Seuil | Choisi sur validation selon le F2, puis figé avant le test externe |
| Interface | Streamlit en français : filtres, export CSV et matrice de confusion |
| Traçabilité | Paramètres, versions, identifiants de partition et SHA-256 des données |
| Qualité | 10 tests automatisés : pipeline, cas invalides, reproductibilité et interface |

Le nouveau module fonctionne sur CPU, sans compte cloud, clé API ou GPU.

## Démarrage rapide

Environnement vérifié : **Python 3.12**. Versions fixées dans `requirements.txt`.

```bash
git clone https://github.com/younesdebug/predictive-maintenance-lab.git
cd predictive-maintenance-lab
python -m venv .venv
```

Activation sur Linux/macOS : `source .venv/bin/activate`.
Sur Windows PowerShell : `.venv/Scripts/Activate.ps1`.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Les rapports de référence sont inclus : l’interface s’ouvre sans réentraînement.
Pour recalculer les résultats et lancer les tests :

```bash
python -m maintenance.train --seed 42 --horizon 30 --trees 200
python -m unittest discover -s tests -v
```

## Résultats mesurés

Exécution du **1er novembre 2026**, graine 42, 200 arbres, seuil **0,30**.
Test externe : **100 moteurs**, une observation finale par moteur, dont **25 positifs**.

| Mesure | Random Forest | Baseline naïve (fréquence de classe) |
| --- | ---: | ---: |
| Précision des alertes | 81,48 % | 0 %* |
| Rappel | 88,00 % | 0 % |
| F2 | 0,8661 | 0 |
| Average precision | 0,9599 | 0,25 |
| ROC AUC | 0,9851 | 0,50 |
| Vrais positifs / faux négatifs | 22 / 3 | 0 / 25 |
| Faux positifs / vrais négatifs | 5 / 70 | 0 / 75 |

*La baseline ne déclenche aucune alerte au seuil retenu ; la précision est définie à zéro par convention.*

[Rapport complet](reports/metrics.json) · [Prédictions individuelles](reports/test_predictions.csv)

Ces chiffres proviennent de la nouvelle baseline ; les résultats LSTM historiques **n’ont pas été réexécutés**.

## Protocole et limites

1. Construire la cible `RUL <= 30` sur les trajectoires d’entraînement complètes.
2. Réserver 25 % des moteurs à la validation avec `GroupShuffleSplit`.
3. Apprendre sur les autres moteurs et choisir le seuil maximisant le F2 sur validation.
4. Réentraîner sur l’ensemble d’entraînement en conservant ce seuil.
5. Évaluer la dernière observation de chaque moteur du test externe.

Le F2 privilégie le rappel. Le choix réel du seuil nécessiterait une analyse des coûts.
Les scores ne sont pas calibrés et une seule partition est présentée. La validation par cycle
et le test par moteur ont des distributions différentes : leurs métriques ne sont pas directement
comparables. L’importance des variables n’établit pas de causalité.

[Fiche du modèle](docs/MODEL_CARD.md) · [Guide technique pour l’entretien](docs/TECHNICAL_GUIDE.md)

## Organisation

| Chemin | Rôle |
| --- | --- |
| `maintenance/` | Nouveau module de préparation, entraînement et évaluation |
| `app.py` | Nouveau tableau de bord de lecture des résultats |
| `tests/` | Tests du module et de l’interface |
| `reports/` | Résultats réellement générés par la baseline |
| `Dataset/` | Données présentes dans le dépôt d’origine |
| `src/lstm/`, `Output/` | Code et sorties LSTM historiques, non réexécutés |
| `docs/UPSTREAM_README.md` | README original conservé |

Les nouvelles dépendances ne remettent pas à niveau l’ancien code Keras/TensorFlow.
L’application ne charge aucun des modèles `.h5` historiques.

## Sources et attribution

- Projet d’origine : **Umberto Griffo**, [Predictive-Maintenance-using-LSTM](https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM), MIT.
- Données citées par le projet d’origine : **A. Saxena et K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set**, NASA Ames Prognostics Data Repository. Les fichiers hérités sont conservés sans modification ; voir le [README original](docs/UPSTREAM_README.md).
- Adaptation de portfolio maintenue par **Younes Radouani**, développée avec assistance IA. Le périmètre est détaillé dans [PROVENANCE.md](docs/PROVENANCE.md).

[GitHub](https://github.com/younesdebug) · [LinkedIn](https://www.linkedin.com/in/younes-radouani-a41723327/)
