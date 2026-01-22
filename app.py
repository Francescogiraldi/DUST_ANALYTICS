import io
import json
import zipfile
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# ============================
# Configuration Streamlit
# ============================
st.set_page_config(page_title="Dust — ROI & Usage Dashboard", layout="wide")


# ============================
# Références API (documentation)
# ============================
# Endpoint: GET https://dust.tt/api/v1/w/{wId}/workspace-usage (CSV ou JSON) :contentReference[oaicite:1]{index=1}


# ============================
# Secrets (Streamlit Cloud)
# ============================
def _secret(key: str) -> Optional[str]:
    try:
        if key in st.secrets:
            v = str(st.secrets[key]).strip()
            return v if v else None
    except Exception:
        return None
    return None


def get_required_secrets() -> Tuple[str, str, str]:
    api_key = _secret("DUST_API_KEY")
    w_id = _secret("DUST_WORKSPACE_ID")
    base_url = _secret("DUST_BASE_URL") or "https://dust.tt"
    if not api_key or not w_id:
        st.error(
            "Secrets manquants : ajoute **DUST_API_KEY** et **DUST_WORKSPACE_ID** dans "
            "Streamlit Cloud → Settings → Secrets."
        )
        st.stop()
    return api_key, w_id, base_url


# ============================
# Dates (YYYY-MM) & coûts TTC
# ============================
def ym(d: date) -> str:
    return d.strftime("%Y-%m")


def months_inclusive(start_ym: str, end_ym: str) -> int:
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    return (ey - sy) * 12 + (em - sm) + 1


def cout_ttc(mensuel_ht: float, tva_pct: float) -> float:
    return float(mensuel_ht) * (1.0 + float(tva_pct) / 100.0)


def cout_ttc_periode(mode: str, start_ym: str, end_ym: Optional[str], mensuel_ht: float, tva_pct: float) -> float:
    mensuel = cout_ttc(mensuel_ht, tva_pct)
    if mode == "month":
        return mensuel
    if not end_ym:
        return mensuel
    return mensuel * months_inclusive(start_ym, end_ym)


# ============================
# Mapping providers / modèles (robuste NaN)
# ============================
def is_na(x: Any) -> bool:
    try:
        return x is None or pd.isna(x)
    except Exception:
        return x is None


def provider_label_from_provider_id(provider_id: Any) -> str:
    # IMPORTANT : provider_id peut être NaN (float) après jointure -> on sécurise
    if is_na(provider_id):
        return "Inconnu"
    s = str(provider_id).strip()
    if not s or s.lower() == "nan":
        return "Inconnu"
    mapping = {
        "openai": "OpenAI (ChatGPT)",
        "anthropic": "Anthropic (Claude)",
        "google": "Google (Gemini)",
        "mistral": "Mistral",
        "meta": "Meta (Llama)",
    }
    return mapping.get(s.lower(), s)


def provider_from_base_model(model_id: Any) -> str:
    # IMPORTANT : model_id peut être NaN
    if is_na(model_id):
        return "Autres"
    s = str(model_id).strip().lower()
    if not s or s == "nan":
        return "Autres"
    if "claude" in s:
        return "Anthropic (Claude)"
    if "gpt" in s or s.startswith(("o1", "o3")) or "openai" in s:
        return "OpenAI (ChatGPT)"
    if "gemini" in s:
        return "Google (Gemini)"
    if "mistral" in s or "mixtral" in s:
        return "Mistral"
    if "llama" in s:
        return "Meta (Llama)"
    return "Autres"


# ============================
# Appel API Dust
# ============================
@dataclass(frozen=True)
class UsageQuery:
    mode: str           # "month" | "range"
    start: str          # "YYYY-MM"
    end: Optional[str]  # "YYYY-MM" si range
    table: str          # "users" | "assistant_messages" | "assistants"
    include_inactive: bool
    output_format: str  # "csv" | "json"

    def params(self) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "start": self.start,
            "mode": self.mode,
            "table": self.table,
            "includeInactive": str(self.include_inactive).lower(),
            "format": self.output_format,
        }
        if self.mode == "range" and self.end:
            p["end"] = self.end
        return p


def parse_api_response(resp: requests.Response) -> pd.DataFrame:
    ctype = (resp.headers.get("Content-Type") or "").lower()

    # ZIP de CSV
    if "application/zip" in ctype or resp.content[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with z.open(csv_names[0]) as f:
            return pd.read_csv(f)

    # CSV
    if "text/csv" in ctype:
        return pd.read_csv(io.StringIO(resp.text))

    # JSON (liste ou objet)
    if "application/json" in ctype:
        payload = resp.json()
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            # essaie de trouver une liste
            for v in payload.values():
                if isinstance(v, list):
                    return pd.DataFrame(v)
            # fallback
            try:
                return pd.DataFrame.from_dict(payload, orient="index")
            except Exception:
                return pd.DataFrame()

    # Fallback “best effort”
    try:
        return pd.read_csv(io.StringIO(resp.text))
    except Exception:
        try:
            return pd.read_json(io.StringIO(resp.text))
        except Exception:
            return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=15 * 60)
def fetch_usage_df(base_url: str, w_id: str, api_key: str, q: UsageQuery) -> pd.DataFrame:
    url = f"{base_url.rstrip('/')}/api/v1/w/{w_id}/workspace-usage"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, params=q.params(), timeout=90)

    if resp.status_code == 403:
        raise PermissionError("Accès refusé (403). Vérifie les droits de la clé API sur ce workspace.")
    if resp.status_code != 200:
        raise RuntimeError(f"Erreur API ({resp.status_code}) : {(resp.text or '')[:800]}")

    return parse_api_response(resp)


# ============================
# Préparation des données (sans emails)
# ============================
def normalize_users(users_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()

    # Renommages stables (tes CSV: userId, userName, userEmail, ...)
    if "userId" in df.columns:
        df = df.rename(columns={"userId": "user_id"})
    if "userName" in df.columns:
        df = df.rename(columns={"userName": "user_name"})

    # Conversions utiles
    if "messageCount" in df.columns:
        df["messageCount"] = pd.to_numeric(df["messageCount"], errors="coerce").fillna(0).astype(int)
    if "activeDaysCount" in df.columns:
        df["activeDaysCount"] = pd.to_numeric(df["activeDaysCount"], errors="coerce").fillna(0).astype(int)

    # On ne supprime pas “userEmail” ici (utile éventuellement pour export interne),
    # mais on ne l'affichera JAMAIS dans l’UI.
    return df


def normalize_assistants(as_df: pd.DataFrame) -> pd.DataFrame:
    df = as_df.copy()

    # Nettoyage privacy : on ne veut pas afficher authorEmails
    if "authorEmails" in df.columns:
        df = df.drop(columns=["authorEmails"])

    # Conversions utiles
    for c in ["messages", "distinctUsersReached", "distinctConversations"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "providerId" in df.columns:
        df["provider"] = df["providerId"].map(provider_label_from_provider_id)
    else:
        df["provider"] = "Inconnu"

    return df


def normalize_messages(msgs_df: pd.DataFrame) -> pd.DataFrame:
    df = msgs_df.copy()

    # createdAt -> datetime
    if "createdAt" in df.columns:
        df["created_at"] = pd.to_datetime(df["createdAt"], errors="coerce")
    else:
        df["created_at"] = pd.NaT

    df = df.dropna(subset=["created_at"])
    df["jour"] = df["created_at"].dt.date

    # Confidentialité : on supprime email si présent
    if "user_email" in df.columns:
        df = df.drop(columns=["user_email"])

    # Normaliser assistant_settings
    if "assistant_settings" in df.columns:
        df["assistant_settings"] = df["assistant_settings"].astype(str).str.lower()
    else:
        df["assistant_settings"] = "unknown"

    return df


def enrich_messages(users_df: pd.DataFrame, msgs_df: pd.DataFrame, as_df: pd.DataFrame) -> pd.DataFrame:
    """
    - Jointure users -> messages (pour user_name)
    - Jointure assistants -> messages (pour providerId/modelId/settings des agents)
    - Classification :
        * Agents personnalisés si assistant_settings in {published, unpublished}
          OU assistant_name match un agent du catalogue
        * Sinon : LLM de base
    - Ajout :
        * statut_publication (published/unpublished/inconnu)
        * llm_famille (OpenAI/Anthropic/…)
        * llm_modele (id du modèle)
    """
    df = msgs_df.copy()

    # 1) Jointure users (user_id -> user_name)
    if "user_id" in df.columns and "user_id" in users_df.columns:
        df = df.merge(
            users_df[["user_id", "user_name", "messageCount", "activeDaysCount", "lastMessageSent", "groups"]],
            on="user_id",
            how="left",
            suffixes=("", "_user")
        )

    # Label user (sans email)
    if "user_name" in df.columns:
        df["user_label"] = df["user_name"].fillna(df["user_id"].astype(str))
    else:
        df["user_label"] = df["user_id"].astype(str) if "user_id" in df.columns else "Inconnu"

    # 2) Jointure assistants (assistant_name -> assistants.name)
    # (sur tes CSV: assistants.name, assistant_messages.assistant_name)
    as_name_set = set(as_df["name"].dropna().astype(str)) if "name" in as_df.columns else set()

    if "assistant_name" in df.columns and "name" in as_df.columns:
        df = df.merge(
            as_df[["name", "settings", "modelId", "providerId", "provider"]],
            left_on="assistant_name",
            right_on="name",
            how="left",
            suffixes=("", "_agent")
        )

    # 3) Classification (vectorisée)
    published_mask = df["assistant_settings"].isin(["published", "unpublished"])
    name_mask = df["assistant_name"].astype(str).isin(as_name_set) if "assistant_name" in df.columns else False
    joined_mask = df.get("providerId").notna() if "providerId" in df.columns else False

    is_agent = published_mask | name_mask | joined_mask
    df["type_usage"] = "LLM de base"
    df.loc[is_agent, "type_usage"] = "Agents personnalisés"

    # 4) Statut publication
    df["statut_publication"] = "N/A"
    # priorité : assistant_settings (logs)
    df.loc[is_agent, "statut_publication"] = df.loc[is_agent, "assistant_settings"].replace(
        {"published": "published", "unpublished": "unpublished"}
    )
    # fallback : assistants.settings si assistant_settings est unknown
    if "settings" in df.columns:
        need = is_agent & df["statut_publication"].isin(["unknown", "N/A"])  # prudence
        df.loc[need, "statut_publication"] = df.loc[need, "settings"].astype(str).str.lower()

    # normalisation finale
    df.loc[is_agent, "statut_publication"] = df.loc[is_agent, "statut_publication"].replace(
        {"nan": "inconnu", "unknown": "inconnu", "": "inconnu"}
    ).fillna("inconnu")

    # 5) LLM famille / modèle (vectorisé)
    base_mask = df["type_usage"].eq("LLM de base")

    # modèle de base : assistant_id (tes logs) sinon assistant_name
    base_model = None
    if "assistant_id" in df.columns:
        base_model = df["assistant_id"]
    elif "assistant_name" in df.columns:
        base_model = df["assistant_name"]
    else:
        base_model = pd.Series([""] * len(df), index=df.index)

    base_model = base_model.fillna(df.get("assistant_name"))

    df["llm_modele"] = "Inconnu"
    df.loc[base_mask, "llm_modele"] = base_model.loc[base_mask].astype(str)

    # pour agents : modelId
    if "modelId" in df.columns:
        df.loc[~base_mask, "llm_modele"] = df.loc[~base_mask, "modelId"].astype(str).replace({"nan": "Inconnu"})
    else:
        df.loc[~base_mask, "llm_modele"] = "Inconnu"

    df["llm_famille"] = "Autres"
    df.loc[base_mask, "llm_famille"] = base_model.loc[base_mask].map(provider_from_base_model)

    # agents : provider (déjà labelisé)
    if "provider" in df.columns:
        df.loc[~base_mask, "llm_famille"] = df.loc[~base_mask, "provider"].fillna("Inconnu")
    elif "providerId" in df.columns:
        df.loc[~base_mask, "llm_famille"] = df.loc[~base_mask, "providerId"].map(provider_label_from_provider_id)
    else:
        df.loc[~base_mask, "llm_famille"] = "Inconnu"

    # 6) Nettoyage colonnes inutiles (privacy)
    # 'workspace_name' peut être conservé, mais pas nécessaire au ROI
    return df


# ============================
# KPIs orientés ROI
# ============================
def compute_kpis(
    users_df: pd.DataFrame,
    msgs_df: pd.DataFrame,
    membres_total: int,
    seuil_actif_messages: int,
    cout_periode_ttc: float
) -> Dict[str, Any]:
    users_total = int(len(users_df))

    # Utilisateurs à 0 et actifs basés sur users.messageCount (inclut les zéros)
    if "messageCount" in users_df.columns:
        zero_users = int((users_df["messageCount"] == 0).sum())
        active_users = int((users_df["messageCount"] >= seuil_actif_messages).sum())
    else:
        zero_users = 0
        active_users = int(msgs_df["user_id"].nunique()) if "user_id" in msgs_df.columns else 0

    taux_activation = 100.0 * active_users / max(1, int(membres_total))

    messages_total = int(len(msgs_df))
    conversations_total = int(msgs_df["conversation_id"].nunique()) if "conversation_id" in msgs_df.columns else 0

    # Split agents vs base
    split = msgs_df["type_usage"].value_counts().to_dict() if "type_usage" in msgs_df.columns else {}
    messages_agents = int(split.get("Agents personnalisés", 0))
    messages_base = int(split.get("LLM de base", 0))

    cout_par_membre = cout_periode_ttc / max(1, int(membres_total))
    cout_des_inactifs = cout_par_membre * zero_users

    cout_par_actif = (cout_periode_ttc / active_users) if active_users else None
    cout_par_message = (cout_periode_ttc / messages_total) if messages_total else None
    cout_par_conversation = (cout_periode_ttc / conversations_total) if conversations_total else None

    # Allocation indicative du coût (proxy : proportion de messages)
    cout_agents = cout_periode_ttc * (messages_agents / max(1, messages_total))
    cout_base = cout_periode_ttc * (messages_base / max(1, messages_total))

    return {
        "membres_total": int(membres_total),
        "users_total": users_total,
        "users_actifs": active_users,
        "users_zero": zero_users,
        "taux_activation_pct": taux_activation,
        "messages_total": messages_total,
        "conversations_total": conversations_total,
        "messages_agents": messages_agents,
        "messages_base": messages_base,
        "cout_periode_ttc": cout_periode_ttc,
        "cout_par_membre": cout_par_membre,
        "cout_par_actif": cout_par_actif,
        "cout_par_message": cout_par_message,
        "cout_par_conversation": cout_par_conversation,
        "cout_des_inactifs": cout_des_inactifs,
        "cout_agents_proxy": cout_agents,
        "cout_base_proxy": cout_base,
    }


# ============================
# UI : Dashboard
# ============================
def main() -> None:
    st.title("Dust — Tableau de bord ROI & Usage")
    st.caption(
        "Analyse ROI-oriented : utilisation réelle, adoption des agents (publiés/non publiés), "
        "mix LLM (ChatGPT/Claude/…), et utilisateurs à zéro pour réactivation/désactivation."
    )

    api_key, w_id, base_url_secret = get_required_secrets()

    # -------- Sidebar : paramètres
    with st.sidebar:
        st.header("Paramètres")

        base_url = st.selectbox(
            "Région / Base URL",
            options=list(dict.fromkeys([base_url_secret, "https://dust.tt", "https://eu.dust.tt"])),
            index=0,
        )

        st.markdown("**Workspace (Secrets)**")
        st.code(w_id)

        st.divider()
        st.subheader("Période (mois)")

        mode = st.selectbox("Mode", options=["month", "range"], index=0)

        if mode == "month":
            d0 = st.date_input("Mois", value=date.today().replace(day=1))
            start_ym = ym(d0)
            end_ym = None
        else:
            d_start = st.date_input("Début (mois)", value=date.today().replace(day=1))
            d_end = st.date_input("Fin (mois)", value=date.today().replace(day=1))
            start_ym = ym(min(d_start, d_end))
            end_ym = ym(max(d_start, d_end))

        st.divider()
        st.subheader("Extraction")
        include_inactive = st.checkbox(
            "Inclure les utilisateurs inactifs (messageCount = 0)",
            value=True,
            help="Recommandé pour identifier les comptes à réactiver/désactiver."
        )
        output_format = st.selectbox("Format", options=["csv", "json"], index=0)

        st.divider()
        st.subheader("Définition 'actif'")
        seuil_actif = st.slider(
            "Seuil messages sur la période",
            min_value=1, max_value=50, value=1, step=1
        )

        st.divider()
        st.subheader("Coûts (TTC)")
        mensuel_ht = st.number_input("Abonnement mensuel HT (€)", value=3683.0, step=50.0)
        tva_pct = st.number_input("TVA (%)", value=22.0, step=1.0)
        membres_total = st.number_input("Nombre de membres (abonnés)", value=127, step=1)

        cout_periode = cout_ttc_periode(mode, start_ym, end_ym, mensuel_ht, tva_pct)
        st.caption(f"Coût TTC période (approx.) : **{cout_periode:,.2f} €**")

        st.divider()
        colA, colB = st.columns(2)
        run = colA.button("Charger", use_container_width=True)
        if colB.button("Vider cache", use_container_width=True):
            st.cache_data.clear()
            st.toast("Cache vidé.")

    if not run:
        st.info("Configure la période puis clique sur **Charger**.")
        return

    # -------- Chargement données
    with st.spinner("Appel API Dust…"):
        q_users = UsageQuery(mode=mode, start=start_ym, end=end_ym, table="users",
                             include_inactive=include_inactive, output_format=output_format)
        q_msgs = UsageQuery(mode=mode, start=start_ym, end=end_ym, table="assistant_messages",
                            include_inactive=include_inactive, output_format=output_format)
        q_as = UsageQuery(mode=mode, start=start_ym, end=end_ym, table="assistants",
                          include_inactive=include_inactive, output_format=output_format)

        try:
            users_raw = fetch_usage_df(base_url, w_id, api_key, q_users)
            msgs_raw = fetch_usage_df(base_url, w_id, api_key, q_msgs)
            as_raw = fetch_usage_df(base_url, w_id, api_key, q_as)
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            st.stop()

    # -------- Normalisation
    users_df = normalize_users(users_raw)
    as_df = normalize_assistants(as_raw)
    msgs_df = normalize_messages(msgs_raw)

    # -------- Enrichissement
    msgs_enriched = enrich_messages(users_df, msgs_df, as_df)

    # -------- KPIs
    kpis = compute_kpis(
        users_df=users_df,
        msgs_df=msgs_enriched,
        membres_total=int(membres_total),
        seuil_actif_messages=int(seuil_actif),
        cout_periode_ttc=float(cout_periode)
    )

    # ============================
    # Tabs
    # ============================
    t_resume, t_agents, t_llm, t_users, t_data = st.tabs([
        "Résumé ROI",
        "Agents (publiés / non publiés)",
        "Modèles (LLM de base vs via agents)",
        "Utilisateurs (zéro & ciblage)",
        "Données & exports",
    ])

    # ----------------------------
    # Résumé ROI
    # ----------------------------
    with t_resume:
        st.subheader("KPI principaux")

        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Membres (plan)", f"{kpis['membres_total']:,}")
        a2.metric("Utilisateurs actifs", f"{kpis['users_actifs']:,}")
        a3.metric("Utilisateurs à 0", f"{kpis['users_zero']:,}")
        a4.metric("Taux d’activation", f"{kpis['taux_activation_pct']:.1f}%")
        a5.metric("Messages (assistant)", f"{kpis['messages_total']:,}")

        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Coût TTC période", f"{kpis['cout_periode_ttc']:,.2f} €")
        b2.metric("Coût TTC / membre", f"{kpis['cout_par_membre']:,.2f} €")
        b3.metric("Coût TTC / actif", f"{kpis['cout_par_actif']:,.2f} €" if kpis["cout_par_actif"] else "N/A")
        b4.metric("Coût TTC / message", f"{kpis['cout_par_message']:,.4f} €" if kpis["cout_par_message"] else "N/A")
        b5.metric("Coût des comptes à 0", f"{kpis['cout_des_inactifs']:,.2f} €")

        st.caption(
            "Lecture ROI : **Coût / actif** répartit le coût du plan sur les seuls utilisateurs réellement actifs. "
            "**Coût des comptes à 0** quantifie l’enjeu de réactivation/désactivation."
        )

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            split = msgs_enriched["type_usage"].value_counts().reset_index()
            split.columns = ["type_usage", "messages"]
            fig = px.pie(split, names="type_usage", values="messages", hole=0.45, title="Part d’usage — Agents vs LLM de base")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fam = msgs_enriched["llm_famille"].value_counts().reset_index()
            fam.columns = ["famille", "messages"]
            fig2 = px.bar(fam, x="famille", y="messages", title="Messages par famille LLM (global)")
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Tendances (par jour)")

        daily = msgs_enriched.groupby(["jour", "type_usage"]).size().reset_index(name="messages")
        fig3 = px.area(daily, x="jour", y="messages", color="type_usage", title="Messages/jour — Agents vs LLM de base")
        st.plotly_chart(fig3, use_container_width=True)

        if "user_id" in msgs_enriched.columns:
            dau = msgs_enriched.groupby("jour")["user_id"].nunique().reset_index(name="utilisateurs_actifs_jour")
            fig4 = px.line(dau, x="jour", y="utilisateurs_actifs_jour", title="Utilisateurs actifs/jour (via logs)")
            st.plotly_chart(fig4, use_container_width=True)

    # ----------------------------
    # Agents
    # ----------------------------
    with t_agents:
        st.subheader("Agents — usage & gouvernance (publiés / non publiés)")

        agents_msgs = msgs_enriched[msgs_enriched["type_usage"].eq("Agents personnalisés")].copy()
        if agents_msgs.empty:
            st.info("Aucun usage d’agent détecté sur la période.")
        else:
            s1, s2 = st.columns(2)

            with s1:
                pub = agents_msgs["statut_publication"].value_counts().reset_index()
                pub.columns = ["statut_publication", "messages"]
                fig = px.bar(pub, x="statut_publication", y="messages", title="Messages — publié vs non publié (agents)")
                st.plotly_chart(fig, use_container_width=True)

            with s2:
                top_agents = agents_msgs["assistant_name"].value_counts().head(25).reset_index()
                top_agents.columns = ["agent", "messages"]
                fig2 = px.bar(top_agents, x="agent", y="messages", title="Top 25 agents (messages)")
                st.plotly_chart(fig2, use_container_width=True)

            st.divider()
            st.subheader("Catalogue agents (assistants) — adoption")

            # Agents publiés mais non utilisés (messages == 0)
            if "messages" in as_df.columns and "settings" in as_df.columns:
                published_unused = as_df[(as_df["settings"].astype(str).str.lower() == "published") & (as_df["messages"] == 0)]
                st.markdown(f"**Agents publiés non utilisés** : {len(published_unused):,}")
                if len(published_unused) > 0:
                    st.dataframe(
                        published_unused[["name", "provider", "modelId", "messages", "distinctUsersReached", "distinctConversations", "lastEdit"]],
                        use_container_width=True,
                        height=260
                    )

            # Scatter adoption : users reached vs messages
            if {"distinctUsersReached", "messages"}.issubset(set(as_df.columns)):
                fig3 = px.scatter(
                    as_df,
                    x="distinctUsersReached",
                    y="messages",
                    color="settings" if "settings" in as_df.columns else None,
                    size="distinctConversations" if "distinctConversations" in as_df.columns else None,
                    hover_name="name" if "name" in as_df.columns else None,
                    title="Adoption agents : utilisateurs atteints vs volume (messages)"
                )
                st.plotly_chart(fig3, use_container_width=True)

    # ----------------------------
    # Modèles (base vs via agents)
    # ----------------------------
    with t_llm:
        st.subheader("Comparatif modèles — LLM de base vs via agents")

        base_msgs = msgs_enriched[msgs_enriched["type_usage"].eq("LLM de base")].copy()
        agents_msgs = msgs_enriched[msgs_enriched["type_usage"].eq("Agents personnalisés")].copy()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### LLM de base (usage direct)")
            if base_msgs.empty:
                st.info("Aucun usage de LLM de base sur la période.")
            else:
                fam = base_msgs["llm_famille"].value_counts().reset_index()
                fam.columns = ["famille", "messages"]
                fig = px.pie(fam, names="famille", values="messages", hole=0.45, title="Répartition familles — LLM de base")
                st.plotly_chart(fig, use_container_width=True)

                top = base_msgs["llm_modele"].value_counts().head(20).reset_index()
                top.columns = ["modèle", "messages"]
                fig2 = px.bar(top, x="modèle", y="messages", title="Top 20 modèles de base (messages)")
                st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("### Via agents (modèle sous-jacent)")
            if agents_msgs.empty:
                st.info("Aucun usage via agents sur la période.")
            else:
                fam2 = agents_msgs["llm_famille"].value_counts().reset_index()
                fam2.columns = ["famille", "messages"]
                fig3 = px.pie(fam2, names="famille", values="messages", hole=0.45, title="Répartition familles — via agents")
                st.plotly_chart(fig3, use_container_width=True)

                top2 = agents_msgs["llm_modele"].value_counts().head(20).reset_index()
                top2.columns = ["modèle", "messages"]
                fig4 = px.bar(top2, x="modèle", y="messages", title="Top 20 modèles via agents (messages)")
                st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.subheader("Tendance familles (par jour)")

        daily_fam = msgs_enriched.groupby(["jour", "llm_famille"]).size().reset_index(name="messages")
        fig5 = px.area(daily_fam, x="jour", y="messages", color="llm_famille", title="Messages/jour — par famille LLM")
        st.plotly_chart(fig5, use_container_width=True)

    # ----------------------------
    # Utilisateurs (zéro & ciblage)
    # ----------------------------
    with t_users:
        st.subheader("Utilisateurs à zéro & segmentation (réactivation / désactivation)")

        if "messageCount" not in users_df.columns:
            st.info("Colonne 'messageCount' absente dans la table users : impossible d’identifier les zéros.")
        else:
            users_local = users_df.copy()

            # Vue 'zéro'
            zeros = users_local[users_local["messageCount"] == 0].copy()
            st.markdown(f"### Utilisateurs à **0** sur la période : **{len(zeros):,}**")

            cols = [c for c in ["user_id", "user_name", "messageCount", "activeDaysCount", "lastMessageSent", "groups"] if c in zeros.columns]
            show_zeros = zeros[cols].copy() if cols else zeros

            # Option de tri par groupe
            if "groups" in show_zeros.columns:
                show_zeros = show_zeros.sort_values(by=["groups", "user_name"], na_position="last")
            elif "user_name" in show_zeros.columns:
                show_zeros = show_zeros.sort_values(by=["user_name"], na_position="last")

            st.dataframe(show_zeros, use_container_width=True, height=480)

            csv_zeros = show_zeros.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger la liste (CSV) — utilisateurs à 0",
                data=csv_zeros,
                file_name="utilisateurs_zero.csv",
                mime="text/csv"
            )

            st.divider()
            st.markdown("### Segmentation simple (messageCount)")

            low = users_local[(users_local["messageCount"] >= 1) & (users_local["messageCount"] <= 3)]
            mid = users_local[(users_local["messageCount"] >= 4) & (users_local["messageCount"] <= 20)]
            high = users_local[users_local["messageCount"] > 20]

            s1, s2, s3 = st.columns(3)
            s1.metric("Faible (1–3)", f"{len(low):,}")
            s2.metric("Moyen (4–20)", f"{len(mid):,}")
            s3.metric("Fort (>20)", f"{len(high):,}")

            dist = users_local["messageCount"].value_counts().reset_index()
            dist.columns = ["messageCount", "utilisateurs"]
            dist = dist.sort_values("messageCount").head(80)

            fig = px.bar(dist, x="messageCount", y="utilisateurs", title="Distribution (zoom sur 80 niveaux)")
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Données & exports
    # ----------------------------
    with t_data:
        st.subheader("Exports (sans emails)")

        st.markdown("### assistant_messages (nettoyé)")
        keep = [c for c in [
            "created_at", "jour", "conversation_id", "message_id",
            "user_id", "user_label",
            "assistant_name", "assistant_id",
            "assistant_settings", "type_usage", "statut_publication",
            "llm_famille", "llm_modele", "source"
        ] if c in msgs_enriched.columns]

        st.dataframe(msgs_enriched[keep].head(300), use_container_width=True, height=420)
        st.download_button(
            "Télécharger assistant_messages_clean.csv",
            data=msgs_enriched[keep].to_csv(index=False).encode("utf-8"),
            file_name="assistant_messages_clean.csv",
            mime="text/csv"
        )

        st.divider()
        st.markdown("### users (sans affichage email)")
        keep_u = [c for c in ["user_id", "user_name", "messageCount", "activeDaysCount", "lastMessageSent", "groups"] if c in users_df.columns]
        st.dataframe(
            users_df[keep_u].sort_values("messageCount", ascending=False) if "messageCount" in users_df.columns else users_df[keep_u],
            use_container_width=True, height=420
        )
        st.download_button(
            "Télécharger users_clean.csv",
            data=users_df[keep_u].to_csv(index=False).encode("utf-8"),
            file_name="users_clean.csv",
            mime="text/csv"
        )

        st.divider()
        st.markdown("### assistants (catalogue)")
        keep_a = [c for c in ["name", "description", "settings", "provider", "modelId", "messages", "distinctUsersReached", "distinctConversations", "lastEdit"] if c in as_df.columns]
        st.dataframe(
            as_df[keep_a].sort_values("messages", ascending=False) if "messages" in as_df.columns else as_df[keep_a],
            use_container_width=True, height=420
        )
        st.download_button(
            "Télécharger assistants.csv",
            data=as_df[keep_a].to_csv(index=False).encode("utf-8"),
            file_name="assistants.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
