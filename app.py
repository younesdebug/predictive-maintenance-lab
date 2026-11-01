"""Read-only dashboard for benchmark predictions; no model deserialization."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Predictive Maintenance Lab", page_icon="⚙️", layout="wide")
st.caption("YOUNES RADOUANI · PORTFOLIO IA")
st.title("Predictive Maintenance Lab")
st.write("Explorer les alertes d’un modèle Random Forest sur des moteurs simulés.")
st.info("Démonstration pédagogique sur un benchmark public. Scores non calibrés : "
        "ils ne constituent ni une probabilité fiable de panne réelle ni un ordre d’intervention.")

try:
    report = json.loads((ROOT / "reports/metrics.json").read_text())
    predictions = pd.read_csv(ROOT / "reports/test_predictions.csv")
    importance = pd.read_csv(ROOT / "reports/feature_importance.csv")
except (OSError, ValueError) as exc:
    st.error("Rapports absents ou illisibles. Exécuter : python -m maintenance.train")
    st.stop()

with st.sidebar:
    st.header("Protocole")
    st.metric("Horizon", f"{report['horizon_cycles']} cycles")
    st.metric("Seuil fixé sur validation", f"{report['threshold']:.2f}")
    st.caption("Le jeu de test n’a pas servi à choisir ce seuil. Une ligne représente "
               "le dernier cycle disponible d’un moteur de test.")
    alerts_only = st.checkbox("Afficher uniquement les alertes")
    st.divider()
    st.markdown("[Code source et méthode](https://github.com/younesdebug/predictive-maintenance-lab)")
    st.markdown("[Projet d’origine · Umberto Griffo](https://github.com/umbertogriffo/Predictive-Maintenance-using-LSTM)")

result = report["test_random_forest"]
columns = st.columns(4)
for column, label, value in zip(columns, ["Moteurs de test", "Alertes", "Rappel", "Précision"],
                                [str(len(predictions)), str(int(predictions.alert.sum())),
                                 f"{result['recall']:.1%}", f"{result['precision']:.1%}"]):
    column.metric(label, value)

fleet_tab, evaluation_tab, method_tab = st.tabs(["Vue des moteurs", "Évaluation", "Méthode & limites"])
with fleet_tab:
    st.subheader("Moteurs classés par score")
    st.caption("Classement illustratif. Les labels et la durée de vie restante proviennent "
               "du benchmark et ne seraient pas connus en exploitation.")
    shown = predictions[predictions.alert == 1] if alerts_only else predictions
    shown = shown.sort_values("risk_score", ascending=False)
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "risk_score": st.column_config.ProgressColumn("Score du modèle", min_value=0, max_value=1, format="%.3f"),
        "engine_id": "Moteur", "cycle": "Dernier cycle", "rul": "Cycles restants (vérité)",
        "failure_within_horizon": "Panne dans l’horizon (vérité)", "alert": "Alerte simulée"})
    st.download_button("Télécharger la sélection (CSV)", shown.to_csv(index=False),
                       file_name="maintenance_selection.csv", mime="text/csv")
    engine = st.selectbox("Examiner un moteur", predictions.engine_id.tolist())
    selected = predictions[predictions.engine_id == engine].iloc[0]
    st.write(f"Moteur **{engine}** — score **{selected.risk_score:.3f}** ; "
             f"seuil **{report['threshold']:.2f}** ; "
             f"alerte simulée : **{'oui' if selected.alert else 'non'}**.")

with evaluation_tab:
    st.subheader("Résultats sur le test externe")
    rows = []
    for name, key in [("Random Forest", "test_random_forest"), ("Baseline : fréquence de classe", "test_dummy_prior")]:
        score = report[key]
        rows.append({"Modèle": name, **{k: score[k] for k in ["precision", "recall", "f2", "average_precision", "roc_auc"]}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.write("**Matrice de confusion**")
        tn, fp, fn, tp = result["confusion_matrix_tn_fp_fn_tp"]
        st.table(pd.DataFrame([[tn, fp], [fn, tp]], index=["Réel : pas de panne", "Réel : panne"],
                             columns=["Prédit : pas d’alerte", "Prédit : alerte"]))
    with right:
        st.write("**Importance des variables (impureté)**")
        st.bar_chart(importance.head(8).set_index("feature"), horizontal=True)
        st.caption("Cette importance peut favoriser certaines variables corrélées ; elle ne prouve pas une causalité.")

with method_tab:
    st.markdown("""
### Ce que cette version ajoute
- Baseline Random Forest et comparaison à un classifieur naïf.
- Séparation de la validation par moteur ; aucune cible dans les variables d’entrée.
- Seuil choisi sur validation en privilégiant le rappel (F2), puis figé avant le test.
- Exports de prédictions, empreintes des données, versions et tests automatisés.

### Limites
Une seule partition et une seule graine sont utilisées. La validation porte sur tous les
cycles de moteurs distincts, alors que le test porte sur leurs dernières observations :
les métriques ne sont pas directement comparables. Le modèle doit être validé sur d’autres
partitions et calibré avant toute réflexion opérationnelle. Aucune application médicale
ou industrielle réelle n’a été validée ici.

### Origine
Fork du projet **Predictive-Maintenance-using-LSTM** d’**Umberto Griffo** sous licence MIT.
Les modèles LSTM et leurs résultats historiques sont conservés séparément. Cette extension
de portfolio est maintenue par **Younes Radouani** avec assistance IA pour le développement.
""")
