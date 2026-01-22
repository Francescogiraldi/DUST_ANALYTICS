import io
import os
import zipfile
import calendar
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
import requests
import streamlit as st
from dateutil import parser as dateparser
import plotly.express as px

# ============================
# Configuration Streamlit
# ============================
st.set_page_config(page_title="Dust — ROI & Usage Dashboard", layout="wide")


# ============================
# Paramètres & constantes
# ============================
TABLES_DISPONIBLES = ["users", "assistant_messages", "assistants"]  # on ignore builders/feedback
MODES = ["month", "range"]

DEFAULT_BASE_URL = "https://dust.tt"


# ============================
# Utilitaires Secrets
# ============================
def secret_get(key: str) -> Optional[str]:
    try:
        if key in st.secrets:
            v = str(st.secrets[key]).strip()
            return v if v else None
    except Exception:
        pass
    return None


def require_secrets() -> Tuple[str, str, str]:
    """
    Récupère les secrets obligatoires depuis Streamlit Secrets.
    """
    api_key = secret_get("DUST_API_KEY")
    w_id = secret_get("DUST_WORKSPACE_ID")
    base_url = secret_get("DUST_BASE_URL") or DEFAULT_BASE_URL

    if not api_key or not w_id:
        st.error(
            "Secrets manquants. Ajoute **DUST_API_KEY** et **DUST_WORKSPACE_ID** dans "
            "Streamlit Cloud → Settings → Secrets."
        )
        st.stop()
    return api_key, w_id, base_url


# ============================
# Aides dates (YYYY-MM)
# ============================
def ym(d: date) -> str:
    return d.strftime("%Y-%m")


def count_months_inclusive(start_ym: str, end_ym: str) -> int:
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    return (ey - sy) * 12 + (em - sm) + 1


# ============================
# Coûts TTC (approximation)
# ============================
def cout_ttc(mensuel_ht: float, tva_pct: float) -> float:
    return float(mensuel_ht) * (1.0 + float(tva_pct) / 100.0)


def cout_ttc_periode(mode: str, start_ym: str, end_ym: Optional[str], mensuel_ht: float, tva_pct: float) -> float:
    """
    Hypothèse simple (utile pour ROI) :
      - month : 1 mois
      - range : nombre de mois inclusifs * coût mensuel TTC
    """
    mensuel = cout_ttc(mensuel_ht, tva_pct)
    if mode == "month":
        return mensuel
    if not end_ym:
        return mensuel
    n = count_months_inclusive(start_ym, end_ym)
    return mensuel * n


# ============================
# API Dust — workspace usage
# ============================
@dataclass
class UsageQuery:
    mode: str
    start: str          # YYYY-MM
    end: Optional[str]  # YYYY-MM si mode=range
    table: str          # users / assistant_messages / assistants
    accept: str = "text/csv"  # d'après la doc, CSV via header Accept :contentReference[oaicite:1]{index=1}

    def as_params(self) -> Dict[str, str]:
        params = {
            "start": self.start,
            "mode": self.mode,
            "table": self.table,
        }
        if self.mode == "range" and self.end:
            params["end"] = self.end
        return params


@st.cache_data(show_spinner=False, ttl=15 * 60)
def fetch_usage_table(base_url: str, w_id: str, api_key: str, q: UsageQuery) -> pd.DataFrame:
    """
    Appelle l'endpoint /api/v1/w/{wId}/workspace-usage et renvoie un DataFrame.
    Supporte CSV et (au cas où) ZIP de CSV.
    """
    url = f"{base_url.rstrip('/')}/api/v1/w/{w_id}/workspace-usage"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": q.accept,
    }

    resp = requests.get(url, headers=headers, params=q.as_params(), timeout=90)
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if resp.status_code == 403:
        raise PermissionError("Accès refusé (403). Vérifie les droits de la clé API / workspace.")
    if resp.status_code != 200:
        raise RuntimeError(f"Erreur API ({resp.status_code}) : {(resp.text or '')[:600]}")

    # ZIP (cas rare)
    if "application/zip" in ctype or resp.content[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        # on prend le premier CSV
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with z.open(csv_names[0]) as f:
            return pd.read_csv(f)

    # CSV
    if "text/csv" in ctype or resp.text[:100].count(",") > 0:
        return pd.read_csv(io.StringIO(resp.text))

    # fallback
    try:
        return pd.read_json(io.StringIO(resp.text))
    except Exception:
        return pd.DataFrame()


# ============================
# Préparation & classification (Agents vs LLM de base)
# ============================
def safe_dt(x: Any):
    try:
        return dateparser.parse(str(x))
    except Exception:
        return None


def provider_from_base_model(model_id: str) -> str:
    """
    Heuristique simple pour classer les modèles de base en familles.
    Ajuste si tu as des conventions internes.
    """
    s = (model_id or "").lower()
    if "claude" in s:
        return "Anthropic (Claude)"
    if "gpt" in s or s.startswith("o1") or s.startswith("o3") or "openai" in s:
        return "OpenAI (ChatGPT)"
    if "gemini" in s:
        return "Google (Gemini)"
    if "mistral" in s or "mixtral" in s:
        return "Mistral"
    if "llama" in s:
        return "Meta (Llama)"
    return "Autres"


def provider_label_from_provider_id(provider_id: Optional[str]) -> str:
    if not provider_id:
        return "Inconnu"
    m = {
        "anthropic": "Anthropic (Claude)",
        "openai": "OpenAI (ChatGPT)",
        "google": "Google (Gemini)",
        "mistral": "Mistral",
        "meta": "Meta (Llama)",
    }
    return m.get(provider_id.lower(), provider_id)


def prepare_data(users: pd.DataFrame, messages: pd.DataFrame, assistants: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    - Nettoie/renomme quelques colonnes
    - Joint users -> messages (pour afficher les noms, pas les emails)
    - Joint assistants -> messages (pour modèle/provider des agents)
    - Classe chaque message en:
        * "Agents personnalisés"
        * "LLM de base"
      + statut publication si agent
    """
    users_df = users.copy()
    msgs_df = messages.copy()
    as_df = assistants.copy()

    # ---- Users
    # Colonnes attendues (exemples): userId, userName, userEmail, messageCount, lastMessageSent, activeDaysCount, groups
    if "userId" in users_df.columns:
        users_df = users_df.rename(columns={"userId": "user_id"})
    if "userName" in users_df.columns:
        users_df = users_df.rename(columns={"userName": "user_name"})
    # On garde userEmail hors affichage
    for c in list(users_df.columns):
        if c.lower() in {"useremail", "email"}:
            # on laisse la colonne mais on ne l'affichera pas
            pass

    # ---- Messages
    # Colonnes attendues: createdAt, assistant_id, assistant_name, assistant_settings, conversation_id, user_id, user_email, source
    if "createdAt" in msgs_df.columns:
        msgs_df["created_at"] = msgs_df["createdAt"].map(safe_dt)
    else:
        # fallback
        for cand in ["created_at", "timestamp", "time", "date"]:
            if cand in msgs_df.columns:
                msgs_df["created_at"] = msgs_df[cand].map(safe_dt)
                break
    if "user_id" not in msgs_df.columns and "userId" in msgs_df.columns:
        msgs_df = msgs_df.rename(columns={"userId": "user_id"})

    # jointure users pour récupérer user_name
    if "user_id" in msgs_df.columns and "user_id" in users_df.columns and "user_name" in users_df.columns:
        msgs_df = msgs_df.merge(users_df[["user_id", "user_name", "messageCount", "activeDaysCount", "groups", "lastMessageSent"]]
                                .rename(columns={
                                    "messageCount": "user_message_count",
                                    "activeDaysCount": "user_active_days",
                                    "lastMessageSent": "user_last_message_sent"
                                }),
                                on="user_id", how="left")

    # Supprimer l'email utilisateur des messages (confidentialité)
    if "user_email" in msgs_df.columns:
        msgs_df = msgs_df.drop(columns=["user_email"])

    # ---- Assistants (catalogue des agents)
    # Colonnes attendues: name, settings, modelId, providerId, messages, distinctUsersReached, distinctConversations, lastEdit
    if "name" in as_df.columns:
        as_names = set(as_df["name"].dropna().astype(str))
    else:
        as_names = set()

    # jointure assistants pour enrichir les messages
    if "assistant_name" in msgs_df.columns and "name" in as_df.columns:
        msgs_df = msgs_df.merge(
            as_df[["name", "settings", "modelId", "providerId"]],
            left_on="assistant_name",
            right_on="name",
            how="left",
            suffixes=("", "_agent")
        )

    # ---- Classification Agents vs Base
    # Règle robuste (sur tes exports réels):
    # - Les agents ont assistant_settings = published/unpublished (souvent)
    # - Les modèles de base ont assistant_settings = unknown et assistant_id == assistant_name (souvent)
    def is_agent_row(r) -> bool:
        s = str(r.get("assistant_settings", "")).lower()
        an = str(r.get("assistant_name", "")).strip()
        if s in {"published", "unpublished"}:
            return True
        if an in as_names:
            return True
        # Si on a un match dans la jointure (providerId/modelId non nuls)
        if pd.notna(r.get("providerId")) or pd.notna(r.get("modelId")):
            return True
        return False

    msgs_df["type_usage"] = msgs_df.apply(lambda r: "Agents personnalisés" if is_agent_row(r) else "LLM de base", axis=1)

    # Statut publication
    def publication_statut(r) -> str:
        if r["type_usage"] != "Agents personnalisés":
            return "N/A"
        s = str(r.get("assistant_settings", "")).lower()
        if s in {"published", "unpublished"}:
            return s
        s2 = str(r.get("settings", "")).lower()
        if s2 in {"published", "unpublished"}:
            return s2
        return "inconnu"

    msgs_df["statut_publication"] = msgs_df.apply(publication_statut, axis=1)

    # LLM famille + modèle (base vs agent)
    def llm_famille(r) -> str:
        if r["type_usage"] == "LLM de base":
            base_model = str(r.get("assistant_id") or r.get("assistant_name") or "")
            return provider_from_base_model(base_model)
        return provider_label_from_provider_id(r.get("providerId"))

    def llm_modele(r) -> str:
        if r["type_usage"] == "LLM de base":
            return str(r.get("assistant_id") or r.get("assistant_name") or "")
        mid = r.get("modelId")
        return str(mid) if pd.notna(mid) else "Inconnu"

    msgs_df["llm_famille"] = msgs_df.apply(llm_famille, axis=1)
    msgs_df["llm_modele"] = msgs_df.apply(llm_modele, axis=1)

    # Clés temps
    if "created_at" in msgs_df.columns:
        msgs_df = msgs_df.dropna(subset=["created_at"])
        msgs_df["jour"] = msgs_df["created_at"].dt.date

    return users_df, msgs_df, as_df


# ============================
# KPIs ROI-oriented
# ============================
def compute_kpis(
    users_df: pd.DataFrame,
    msgs_df: pd.DataFrame,
    as_df: pd.DataFrame,
    membres_total: int,
    seuil_actif_messages: int,
    cout_periode_ttc: float
) -> Dict[str, Any]:
    # Utilisateurs actifs/inactifs (table users => inclut les zéros)
    active_users = 0
    inactive_users = 0
    total_users = 0

    if "messageCount" in users_df.columns:
        mc = pd.to_numeric(users_df["messageCount"], errors="coerce").fillna(0)
        total_users = int(len(users_df))
        active_users = int((mc >= seuil_actif_messages).sum())
        inactive_users = int((mc == 0).sum())

    # Usage global
    total_messages = int(len(msgs_df))
    total_conversations = int(msgs_df["conversation_id"].nunique()) if "conversation_id" in msgs_df.columns else None
    actifs_via_logs = int(msgs_df["user_id"].nunique()) if "user_id" in msgs_df.columns else None

    # Split agents vs base
    split = msgs_df["type_usage"].value_counts().to_dict() if "type_usage" in msgs_df.columns else {}
    agents_messages = int(split.get("Agents personnalisés", 0))
    base_messages = int(split.get("LLM de base", 0))

    # Coûts unitaires
    cout_par_membre = cout_periode_ttc / max(1, int(membres_total))
    cout_inactifs = cout_par_membre * inactive_users if inactive_users is not None else None
    cout_par_actif = (cout_periode_ttc / max(1, int(active_users))) if active_users else None
    cout_par_message = (cout_periode_ttc / max(1, int(total_messages))) if total_messages else None
    cout_par_conversation = (cout_periode_ttc / max(1, int(total_conversations))) if total_conversations else None

    taux_activation = (active_users / max(1, membres_total)) * 100.0

    return {
        "membres_total": int(membres_total),
        "users_total": total_users,
        "users_actifs": active_users,
        "users_inactifs_zero": inactive_users,
        "taux_activation_pct": taux_activation,
        "messages_total": total_messages,
        "conversations_total": total_conversations,
        "actifs_via_logs": actifs_via_logs,
        "messages_agents": agents_messages,
        "messages_base": base_messages,
        "cout_periode_ttc": cout_periode_ttc,
        "cout_par_membre": cout_par_membre,
        "cout_inactifs": cout_inactifs,
        "cout_par_actif": cout_par_actif,
        "cout_par_message": cout_par_message,
        "cout_par_conversation": cout_par_conversation,
    }


# ============================
# UI / Dashboard
# ============================
def main():
    st.title("Dust — Tableau de bord ROI & Usage (LLM de base vs Agents)")
    st.caption(
        "Objectif : analyser l’usage réel (utilisateurs, agents, modèles) et produire des KPI orientés ROI "
        "(coût TTC par actif, par message, et identification des utilisateurs à 0)."
    )

    api_key, w_id, base_url_secret = require_secrets()

    # ---- Sidebar
    with st.sidebar:
        st.header("Paramètres")

        # Base URL (optionnel)
        base_url = st.selectbox(
            "Région / Base URL",
            options=[base_url_secret, "https://dust.tt", "https://eu.dust.tt"],
            index=0 if base_url_secret in [base_url_secret, "https://dust.tt", "https://eu.dust.tt"] else 0
        )

        st.markdown("**Workspace** (depuis Secrets)")
        st.code(w_id)

        st.divider()
        st.subheader("Période")
        mode = st.selectbox("Mode", options=MODES, index=0, help="month = un mois, range = plusieurs mois (YYYY-MM).")

        if mode == "month":
            d = st.date_input("Mois", value=date.today().replace(day=1))
            start_ym = ym(d)
            end_ym = None
        else:
            d0 = st.date_input("Début (mois)", value=date.today().replace(day=1))
            d1 = st.date_input("Fin (mois)", value=date.today().replace(day=1))
            start_ym = ym(min(d0, d1))
            end_ym = ym(max(d0, d1))

        st.divider()
        st.subheader("Utilisateurs")
        seuil_actif = st.slider(
            "Seuil 'client actif' (messages sur la période)",
            min_value=1, max_value=50, value=1, step=1,
            help="Un utilisateur est 'actif' si messageCount >= ce seuil."
        )

        st.divider()
        st.subheader("Coûts (TTC)")
        mensuel_ht = st.number_input("Abonnement mensuel HT (€)", value=3683.0, step=50.0)
        tva_pct = st.number_input("TVA (%)", value=22.0, step=1.0)
        membres_total = st.number_input("Nombre d'abonnés (membres)", value=127, step=1)

        st.divider()
        colA, colB = st.columns(2)
        with colA:
            run = st.button("🚀 Charger", use_container_width=True)
        with colB:
            if st.button("🧹 Vider cache", use_container_width=True):
                st.cache_data.clear()
                st.toast("Cache vidé.")

        st.caption("Clé API & Workspace ID sont lus depuis **Streamlit Secrets**.")

    if not run:
        st.info("Configure la période et clique sur **Charger**.")
        return

    # ---- Chargement données
    with st.spinner("Chargement des données via l'API Dust..."):
        try:
            q_users = UsageQuery(mode=mode, start=start_ym, end=end_ym, table="users")
            q_msgs = UsageQuery(mode=mode, start=start_ym, end=end_ym, table="assistant_messages")
            q_as = UsageQuery(mode=mode, start=start_ym, end=end_ym, table="assistants")

            users_df = fetch_usage_table(base_url, w_id, api_key, q_users)
            msgs_df = fetch_usage_table(base_url, w_id, api_key, q_msgs)
            as_df = fetch_usage_table(base_url, w_id, api_key, q_as)

        except PermissionError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            st.stop()

    # ---- Préparation & enrichissement
    users_df, msgs_df, as_df = prepare_data(users_df, msgs_df, as_df)

    # ---- Coûts sur période
    cout_period = cout_ttc_periode(mode, start_ym, end_ym, mensuel_ht, tva_pct)

    # ---- KPIs
    kpis = compute_kpis(
        users_df=users_df,
        msgs_df=msgs_df,
        as_df=as_df,
        membres_total=int(membres_total),
        seuil_actif_messages=int(seuil_actif),
        cout_periode_ttc=float(cout_period),
    )

    # ============================
    # Onglets
    # ============================
    t_resume, t_agents, t_llm, t_users, t_data = st.tabs([
        "📌 Résumé ROI",
        "🤖 Agents (publiés / non publiés)",
        "🧠 LLM de base (ChatGPT / Claude / …)",
        "👤 Utilisateurs (zéro & segmentation)",
        "🗂 Données & export"
    ])

    # ----------------------------
    # Résumé ROI
    # ----------------------------
    with t_resume:
        st.subheader("KPI principaux (période sélectionnée)")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Membres (plan)", f"{kpis['membres_total']:,}")
        c2.metric("Utilisateurs actifs", f"{kpis['users_actifs']:,}")
        c3.metric("Utilisateurs à 0", f"{kpis['users_inactifs_zero']:,}")
        c4.metric("Taux d’activation", f"{kpis['taux_activation_pct']:.1f}%")
        c5.metric("Messages (assistant)", f"{kpis['messages_total']:,}")

        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Coût TTC période", f"{kpis['cout_periode_ttc']:,.2f} €")
        c7.metric("Coût TTC / membre", f"{kpis['cout_par_membre']:,.2f} €")
        c8.metric("Coût TTC / actif", f"{kpis['cout_par_actif']:,.2f} €" if kpis["cout_par_actif"] else "N/A")
        c9.metric("Coût TTC / message", f"{kpis['cout_par_message']:,.4f} €" if kpis["cout_par_message"] else "N/A")
        c10.metric("Coût TTC des inactifs", f"{kpis['cout_inactifs']:,.2f} €" if kpis["cout_inactifs"] is not None else "N/A")

        st.caption(
            "Interprétation ROI : **Coût / actif** = coût du plan réparti uniquement sur les utilisateurs réellement actifs. "
            "La ligne **Coût des inactifs** aide à estimer l’enjeu de réactivation/désactivation."
        )

        st.divider()
        st.subheader("Tendance d’usage (messages / jour)")

        if "jour" in msgs_df.columns:
            daily = msgs_df.groupby(["jour", "type_usage"]).size().reset_index(name="messages")
            fig = px.area(daily, x="jour", y="messages", color="type_usage", title="Messages par jour — Agents vs LLM de base")
            st.plotly_chart(fig, use_container_width=True)

            # Actifs par jour (via logs)
            if "user_id" in msgs_df.columns:
                dau = msgs_df.groupby("jour")["user_id"].nunique().reset_index(name="utilisateurs_actifs_jour")
                fig2 = px.line(dau, x="jour", y="utilisateurs_actifs_jour", title="Utilisateurs actifs par jour (via logs)")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Colonne temporelle absente ou non parsable : impossible de tracer la tendance journalière.")

        st.divider()
        st.subheader("Répartition globale")

        split_df = (
            msgs_df["type_usage"].value_counts()
            .reset_index()
            .rename(columns={"index": "type_usage", "type_usage": "messages"})
        )
        fig3 = px.pie(split_df, names="type_usage", values="messages", title="Part d’usage — Agents vs LLM de base", hole=0.45)
        st.plotly_chart(fig3, use_container_width=True)

        # Répartition par famille LLM (tous usages)
        fam = msgs_df["llm_famille"].value_counts().reset_index()
        fam.columns = ["llm_famille", "messages"]
        fig4 = px.bar(fam, x="llm_famille", y="messages", title="Messages par famille LLM (tous usages)")
        st.plotly_chart(fig4, use_container_width=True)

    # ----------------------------
    # Agents
    # ----------------------------
    with t_agents:
        st.subheader("Adoption des agents (publiés vs non publiés)")

        agents_msgs = msgs_df[msgs_df["type_usage"] == "Agents personnalisés"].copy()

        # Part publié / non publié
        stat = agents_msgs["statut_publication"].value_counts().reset_index()
        stat.columns = ["statut_publication", "messages"]
        fig = px.bar(stat, x="statut_publication", y="messages", title="Messages — publié vs non publié (agents)")
        st.plotly_chart(fig, use_container_width=True)

        # Top agents (par messages) — depuis logs
        if "assistant_name" in agents_msgs.columns:
            top_agents = agents_msgs["assistant_name"].value_counts().head(25).reset_index()
            top_agents.columns = ["agent", "messages"]
            fig2 = px.bar(top_agents, x="agent", y="messages", title="Top 25 agents (messages)")
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Catalogue agents (table 'assistants') — usage agrégé sur la période")

        # Table assistants : on garde l’essentiel ROI
        cols_keep = [c for c in ["name", "settings", "providerId", "modelId", "messages", "distinctUsersReached", "distinctConversations", "lastEdit"]
                     if c in as_df.columns]
        as_show = as_df[cols_keep].copy() if cols_keep else as_df.copy()

        # Labels provider
        if "providerId" in as_show.columns:
            as_show["provider"] = as_show["providerId"].map(provider_label_from_provider_id)

        st.dataframe(as_show.sort_values(by="messages", ascending=False) if "messages" in as_show.columns else as_show,
                     use_container_width=True, height=520)

        # Scatter adoption : messages vs users atteints
        if {"messages", "distinctUsersReached"}.issubset(set(as_df.columns)):
            sc = as_df.copy()
            sc["provider"] = sc["providerId"].map(provider_label_from_provider_id) if "providerId" in sc.columns else "Inconnu"
            fig3 = px.scatter(
                sc,
                x="distinctUsersReached",
                y="messages",
                color="settings" if "settings" in sc.columns else None,
                size="distinctConversations" if "distinctConversations" in sc.columns else None,
                hover_name="name" if "name" in sc.columns else None,
                title="Adoption agents : utilisateurs atteints vs volume (messages)"
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # LLM de base
    # ----------------------------
    with t_llm:
        st.subheader("LLM de base (ChatGPT / Claude / …) — usage direct vs via agents")

        base_msgs = msgs_df[msgs_df["type_usage"] == "LLM de base"].copy()

        if base_msgs.empty:
            st.info("Aucun usage détecté pour les LLM de base sur la période.")
        else:
            # Top modèles de base
            top_models = base_msgs["llm_modele"].value_counts().head(30).reset_index()
            top_models.columns = ["modèle", "messages"]
            fig = px.bar(top_models, x="modèle", y="messages", title="Top 30 modèles de base (messages)")
            st.plotly_chart(fig, use_container_width=True)

            # Par famille/provider
            fam = base_msgs["llm_famille"].value_counts().reset_index()
            fam.columns = ["famille", "messages"]
            fig2 = px.pie(fam, names="famille", values="messages", title="Répartition par famille (LLM de base)", hole=0.45)
            st.plotly_chart(fig2, use_container_width=True)

            # Dans le temps (familles)
            if "jour" in base_msgs.columns:
                daily = base_msgs.groupby(["jour", "llm_famille"]).size().reset_index(name="messages")
                fig3 = px.area(daily, x="jour", y="messages", color="llm_famille", title="LLM de base — messages par jour et par famille")
                st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        st.subheader("Comparatif : LLM via agents (modelId/providerId)")

        agents_msgs = msgs_df[msgs_df["type_usage"] == "Agents personnalisés"].copy()
        if agents_msgs.empty:
            st.info("Aucun usage agent détecté sur la période.")
        else:
            # Par modèle sous-jacent (agents)
            top_agent_models = agents_msgs["llm_modele"].value_counts().head(30).reset_index()
            top_agent_models.columns = ["modèle (agents)", "messages"]
            fig4 = px.bar(top_agent_models, x="modèle (agents)", y="messages", title="Top 30 modèles (via agents)")
            st.plotly_chart(fig4, use_container_width=True)

            fam2 = agents_msgs["llm_famille"].value_counts().reset_index()
            fam2.columns = ["famille", "messages"]
            fig5 = px.pie(fam2, names="famille", values="messages", title="Répartition par famille (via agents)", hole=0.45)
            st.plotly_chart(fig5, use_container_width=True)

    # ----------------------------
    # Utilisateurs
    # ----------------------------
    with t_users:
        st.subheader("Utilisateurs : inactifs (0) & segmentation (réactivation / désactivation)")

        if "messageCount" not in users_df.columns:
            st.info("La table users ne contient pas 'messageCount' : impossible d’identifier les utilisateurs à 0.")
        else:
            users_local = users_df.copy()
            users_local["messageCount"] = pd.to_numeric(users_local["messageCount"], errors="coerce").fillna(0).astype(int)

            # Inactifs = 0
            inactifs = users_local[users_local["messageCount"] == 0].copy()
            st.markdown(f"### Utilisateurs à **0** sur la période : **{len(inactifs):,}**")

            cols = [c for c in ["user_id", "user_name", "messageCount", "activeDaysCount", "lastMessageSent", "groups"] if c in inactifs.columns]
            st.dataframe(inactifs[cols].sort_values(by=["groups", "user_name"], na_position="last") if cols else inactifs,
                         use_container_width=True, height=480)

            # Export inactifs
            if cols:
                csv_bytes = inactifs[cols].to_csv(index=False).encode("utf-8")
                st.download_button("Télécharger la liste (CSV) — utilisateurs à 0", data=csv_bytes,
                                   file_name="utilisateurs_zero.csv", mime="text/csv")

            st.divider()
            st.markdown("### Distribution d’activité (messageCount)")

            # Distribution (log-friendly)
            dist = users_local["messageCount"].value_counts().reset_index()
            dist.columns = ["messageCount", "utilisateurs"]
            dist = dist.sort_values("messageCount")

            fig = px.bar(dist.head(60), x="messageCount", y="utilisateurs", title="Distribution (zoom sur les 60 premiers niveaux)")
            st.plotly_chart(fig, use_container_width=True)

            # Segmentations utiles
            st.divider()
            st.markdown("### Segmentation simple")
            low = users_local[(users_local["messageCount"] >= 1) & (users_local["messageCount"] <= 3)].copy()
            mid = users_local[(users_local["messageCount"] >= 4) & (users_local["messageCount"] <= 20)].copy()
            high = users_local[users_local["messageCount"] > 20].copy()

            a, b, c = st.columns(3)
            a.metric("Faible (1–3)", f"{len(low):,}")
            b.metric("Moyen (4–20)", f"{len(mid):,}")
            c.metric("Fort (>20)", f"{len(high):,}")

            st.caption("Astuce ROI : combine 'zéro' + 'faible' pour une campagne de réactivation ciblée.")

    # ----------------------------
    # Données & export
    # ----------------------------
    with t_data:
        st.subheader("Données nettoyées (sans emails) & exports")

        st.markdown("### assistant_messages (échantillon)")
        show_cols = [c for c in [
            "created_at", "jour", "conversation_id", "user_id", "user_name",
            "assistant_name", "assistant_id", "type_usage", "statut_publication",
            "llm_famille", "llm_modele", "source"
        ] if c in msgs_df.columns]
        st.dataframe(msgs_df[show_cols].head(200), use_container_width=True, height=420)

        csv_msgs = msgs_df[show_cols].to_csv(index=False).encode("utf-8") if show_cols else msgs_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger assistant_messages (CSV — nettoyé)", data=csv_msgs, file_name="assistant_messages_clean.csv", mime="text/csv")

        st.divider()
        st.markdown("### users (sans email)")
        user_cols = [c for c in ["user_id", "user_name", "messageCount", "activeDaysCount", "lastMessageSent", "groups"] if c in users_df.columns]
        st.dataframe(users_df[user_cols].sort_values("messageCount", ascending=False) if "messageCount" in users_df.columns else users_df,
                     use_container_width=True, height=420)

        csv_users = users_df[user_cols].to_csv(index=False).encode("utf-8") if user_cols else users_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger users (CSV — sans email)", data=csv_users, file_name="users_clean.csv", mime="text/csv")

        st.divider()
        st.markdown("### assistants (catalogue)")
        as_cols = [c for c in ["name", "settings", "providerId", "modelId", "messages", "distinctUsersReached", "distinctConversations", "lastEdit"] if c in as_df.columns]
        st.dataframe(as_df[as_cols].sort_values("messages", ascending=False) if "messages" in as_df.columns else as_df,
                     use_container_width=True, height=420)

        csv_as = as_df[as_cols].to_csv(index=False).encode("utf-8") if as_cols else as_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger assistants (CSV)", data=csv_as, file_name="assistants.csv", mime="text/csv")


if __name__ == "__main__":
    main()
