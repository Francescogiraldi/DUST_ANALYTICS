import io
import zipfile
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ============================
# Configuration Streamlit
# ============================
st.set_page_config(page_title="Dust — ROI & Usage Dashboard", layout="wide")


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
# Dates & coûts TTC
# ============================
def ym(d: date) -> str:
    return d.strftime("%Y-%m")


def ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def months_inclusive(start_str: str, end_str: str) -> int:
    def _to_year_month(s: str) -> Tuple[int, int]:
        parts = s.split("-")
        return int(parts[0]), int(parts[1])

    sy, sm = _to_year_month(start_str)
    ey, em = _to_year_month(end_str)
    return (ey - sy) * 12 + (em - sm) + 1


def cout_ttc_membre(prix_unitaire_ht: float, tva_pct: float) -> float:
    return float(prix_unitaire_ht) * (1.0 + float(tva_pct) / 100.0)


def cout_ttc_periode_par_membres(
    mode: str,
    start_str: str,
    end_str: Optional[str],
    membres_total: int,
    prix_unitaire_ht: float,
    tva_pct: float,
) -> float:
    if mode == "month" or not end_str:
        nb_mois = 1
    else:
        nb_mois = months_inclusive(start_str, end_str)

    prix_unitaire_ttc = cout_ttc_membre(prix_unitaire_ht, tva_pct)
    return float(membres_total) * prix_unitaire_ttc * nb_mois


# ============================
# Mapping providers / modèles
# ============================
def is_na(x: Any) -> bool:
    try:
        return x is None or pd.isna(x)
    except Exception:
        return x is None


def provider_label_from_provider_id(provider_id: Any) -> str:
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
# Helpers agents / période
# ============================
def agent_status_from_settings(settings: Any) -> str:
    s = str(settings).strip().lower()
    if s == "published":
        return "Publié"
    if s == "unpublished":
        return "Non publié"
    return "Inconnu"


def get_period_bounds(mode: str, start_param: str, end_param: Optional[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if mode == "month":
        start = pd.to_datetime(f"{start_param}-01", errors="coerce").normalize()
        end = (start + pd.offsets.MonthEnd(0)).normalize()
    else:
        start = pd.to_datetime(start_param, errors="coerce").normalize()
        end = pd.to_datetime(end_param or start_param, errors="coerce").normalize()
    return start, end


# ============================
# Appel API Dust
# ============================
@dataclass(frozen=True)
class UsageQuery:
    mode: str
    start: str
    end: Optional[str]
    table: str
    include_inactive: bool

    def params(self) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "start": self.start,
            "mode": self.mode,
            "table": self.table,
            "includeInactive": str(self.include_inactive).lower(),
        }
        if self.mode == "range" and self.end:
            p["end"] = self.end
        return p


def parse_api_response(resp: requests.Response) -> pd.DataFrame:
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if "application/zip" in ctype or resp.content[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]

        if not csv_names:
            return pd.DataFrame()

        dfs = []
        for name in csv_names:
            with z.open(name) as f:
                dfs.append(pd.read_csv(f))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    if "text/csv" in ctype:
        return pd.read_csv(io.StringIO(resp.text))

    if "application/json" in ctype:
        payload = resp.json()
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, list):
                    return pd.DataFrame(v)
            try:
                return pd.DataFrame.from_dict(payload, orient="index")
            except Exception:
                return pd.DataFrame()

    try:
        return pd.read_csv(io.StringIO(resp.text))
    except Exception:
        try:
            return pd.read_json(io.StringIO(resp.text))
        except Exception:
            return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=15 * 60)
def fetch_usage_df(
    base_url: str,
    w_id: str,
    api_key: str,
    q: UsageQuery,
    output_format: str,
) -> pd.DataFrame:
    url = f"{base_url.rstrip('/')}/api/v1/w/{w_id}/workspace-usage"

    accept_map = {
        "json": "application/json",
        "csv": "text/csv",
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": accept_map.get(output_format, "application/json"),
    }

    resp = requests.get(url, headers=headers, params=q.params(), timeout=90)

    if resp.status_code == 403:
        raise PermissionError("Accès refusé (403). Vérifie les droits de la clé API sur ce workspace.")
    if resp.status_code != 200:
        raise RuntimeError(f"Erreur API ({resp.status_code}) : {(resp.text or '')[:800]}")

    return parse_api_response(resp)


# ============================
# Préparation des données
# ============================
def normalize_users(users_df: pd.DataFrame) -> pd.DataFrame:
    df = users_df.copy()

    if "userId" in df.columns:
        df = df.rename(columns={"userId": "user_id"})
    if "userName" in df.columns:
        df = df.rename(columns={"userName": "user_name"})

    if "messageCount" in df.columns:
        df["messageCount"] = pd.to_numeric(df["messageCount"], errors="coerce").fillna(0).astype(int)
    else:
        df["messageCount"] = 0

    if "activeDaysCount" in df.columns:
        df["activeDaysCount"] = pd.to_numeric(df["activeDaysCount"], errors="coerce").fillna(0).astype(int)
    else:
        df["activeDaysCount"] = 0

    return df


def normalize_assistants(as_df: pd.DataFrame) -> pd.DataFrame:
    df = as_df.copy()

    if "authorEmails" in df.columns:
        df = df.drop(columns=["authorEmails"])

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

    if "createdAt" in df.columns:
        df["created_at"] = pd.to_datetime(df["createdAt"], errors="coerce")
    else:
        df["created_at"] = pd.NaT

    df = df.dropna(subset=["created_at"]).copy()

    if not df.empty:
        df["jour"] = df["created_at"].dt.floor("D")
    else:
        df["jour"] = pd.Series(dtype="datetime64[ns]")

    if "user_email" in df.columns:
        df = df.drop(columns=["user_email"])

    if "assistant_settings" in df.columns:
        df["assistant_settings"] = df["assistant_settings"].astype(str).str.lower()
    else:
        df["assistant_settings"] = "unknown"

    return df


def enrich_messages(users_df: pd.DataFrame, msgs_df: pd.DataFrame, as_df: pd.DataFrame) -> pd.DataFrame:
    df = msgs_df.copy()

    if "user_id" in df.columns and "user_id" in users_df.columns:
        available_user_cols = [
            c
            for c in ["user_id", "user_name", "messageCount", "activeDaysCount", "lastMessageSent", "groups"]
            if c in users_df.columns
        ]
        df = df.merge(
            users_df[available_user_cols],
            on="user_id",
            how="left",
            suffixes=("", "_user"),
        )

    if "user_name" in df.columns and "user_id" in df.columns:
        df["user_label"] = df["user_name"].fillna(df["user_id"].astype(str))
    elif "user_id" in df.columns:
        df["user_label"] = df["user_id"].astype(str)
    else:
        df["user_label"] = "Inconnu"

    as_name_set = set(as_df["name"].dropna().astype(str)) if "name" in as_df.columns else set()

    if "assistant_name" in df.columns and "name" in as_df.columns:
        available_agent_cols = [c for c in ["name", "settings", "modelId", "providerId", "provider"] if c in as_df.columns]
        df = df.merge(
            as_df[available_agent_cols],
            left_on="assistant_name",
            right_on="name",
            how="left",
            suffixes=("", "_agent"),
        )

    published_mask = df["assistant_settings"].isin(["published", "unpublished"])
    name_mask = df["assistant_name"].astype(str).isin(as_name_set) if "assistant_name" in df.columns else False
    joined_mask = df.get("providerId").notna() if "providerId" in df.columns else False
    is_agent = published_mask | name_mask | joined_mask

    df["type_usage"] = "LLM de base"
    df.loc[is_agent, "type_usage"] = "Agents personnalisés"

    df["statut_publication"] = "N/A"
    df.loc[is_agent, "statut_publication"] = df.loc[is_agent, "assistant_settings"].replace(
        {"published": "published", "unpublished": "unpublished"}
    )

    if "settings" in df.columns:
        need = is_agent & df["statut_publication"].isin(["unknown", "N/A"])
        df.loc[need, "statut_publication"] = df.loc[need, "settings"].astype(str).str.lower()

    df.loc[is_agent, "statut_publication"] = (
        df.loc[is_agent, "statut_publication"]
        .replace({"nan": "inconnu", "unknown": "inconnu", "": "inconnu"})
        .fillna("inconnu")
    )

    base_mask = df["type_usage"].eq("LLM de base")

    if "assistant_id" in df.columns:
        base_model = df["assistant_id"]
    elif "assistant_name" in df.columns:
        base_model = df["assistant_name"]
    else:
        base_model = pd.Series([""] * len(df), index=df.index)

    if "assistant_name" in df.columns:
        base_model = base_model.fillna(df["assistant_name"])

    df["llm_modele"] = "Inconnu"
    df.loc[base_mask, "llm_modele"] = base_model.loc[base_mask].astype(str)

    if "modelId" in df.columns:
        df.loc[~base_mask, "llm_modele"] = (
            df.loc[~base_mask, "modelId"].astype(str).replace({"nan": "Inconnu"})
        )
    else:
        df.loc[~base_mask, "llm_modele"] = "Inconnu"

    df["llm_famille"] = "Autres"
    df.loc[base_mask, "llm_famille"] = base_model.loc[base_mask].map(provider_from_base_model)

    if "provider" in df.columns:
        df.loc[~base_mask, "llm_famille"] = df.loc[~base_mask, "provider"].fillna("Inconnu")
    elif "providerId" in df.columns:
        df.loc[~base_mask, "llm_famille"] = df.loc[~base_mask, "providerId"].map(provider_label_from_provider_id)
    else:
        df.loc[~base_mask, "llm_famille"] = "Inconnu"

    return df


# ============================
# Segmentation utilisateurs
# ============================
def assign_user_segment(message_count: int) -> str:
    if int(message_count) == 0:
        return "Inactifs (0)"
    if 1 <= int(message_count) <= 3:
        return "Faible (1–3)"
    if 4 <= int(message_count) <= 20:
        return "Moyen (4–20)"
    return "Fort (>20)"


def regularity_bucket_from_pct(pct: float) -> str:
    if pct <= 0:
        return "0 % des semaines"
    if pct <= 25:
        return "1 à 25 %"
    if pct <= 50:
        return "26 à 50 %"
    if pct <= 75:
        return "51 à 75 %"
    if pct < 100:
        return "76 à 99 %"
    return "100 % des semaines"


def compute_weekly_coverage_all_users(users_df: pd.DataFrame, msgs_df: pd.DataFrame) -> pd.DataFrame:
    users = users_df.copy()

    if "user_name" in users.columns and "user_id" in users.columns:
        users["user_label"] = users["user_name"].fillna(users["user_id"].astype(str))
    elif "user_id" in users.columns:
        users["user_label"] = users["user_id"].astype(str)
    else:
        users["user_label"] = "Inconnu"

    users["segment_usage"] = users["messageCount"].apply(assign_user_segment)

    if msgs_df.empty or "created_at" not in msgs_df.columns or "user_label" not in msgs_df.columns:
        users["semaines_actives"] = 0
        users["semaines_periode"] = 0
        users["taux_couverture_semaines_pct"] = 0.0
        users["regularite_periode"] = "0 % des semaines"
        return users

    df = msgs_df.copy().dropna(subset=["created_at"])
    if df.empty:
        users["semaines_actives"] = 0
        users["semaines_periode"] = 0
        users["taux_couverture_semaines_pct"] = 0.0
        users["regularite_periode"] = "0 % des semaines"
        return users

    iso = df["created_at"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)
    df["year_week"] = df["iso_year"].astype(str) + "-S" + df["iso_week"].astype(str).str.zfill(2)

    total_weeks = int(df["year_week"].nunique())

    weekly = (
        df.groupby("user_label")
        .agg(
            semaines_actives=("year_week", "nunique"),
            conversations_logs=("conversation_id", "nunique") if "conversation_id" in df.columns else ("user_label", "size"),
            dernier_message_logs=("jour", "max") if "jour" in df.columns else ("created_at", "max"),
            messages_logs=("user_label", "size"),
        )
        .reset_index()
    )

    users = users.merge(weekly, on="user_label", how="left")
    users["semaines_actives"] = users["semaines_actives"].fillna(0).astype(int)
    users["semaines_periode"] = total_weeks
    users["taux_couverture_semaines_pct"] = (
        100.0 * users["semaines_actives"] / max(1, total_weeks)
    ).round(1)
    users["regularite_periode"] = users["taux_couverture_semaines_pct"].apply(regularity_bucket_from_pct)

    if "lastMessageSent" in users.columns:
        users["dernier_message"] = users["dernier_message_logs"].fillna(users["lastMessageSent"])
    else:
        users["dernier_message"] = users["dernier_message_logs"]

    if "conversations_logs" in users.columns:
        users["conversations"] = users["conversations_logs"].fillna(0).astype(int)
    else:
        users["conversations"] = 0

    drop_cols = [c for c in ["conversations_logs", "dernier_message_logs", "messages_logs"] if c in users.columns]
    if drop_cols:
        users = users.drop(columns=drop_cols)

    return users


# ============================
# KPIs orientés ROI
# ============================
def compute_kpis(
    users_df: pd.DataFrame,
    msgs_df: pd.DataFrame,
    membres_total: int,
    seuil_actif_messages: int,
    cout_periode_ttc: float,
) -> Dict[str, Any]:
    users_total = int(len(users_df))

    if "messageCount" in users_df.columns:
        zero_users = int((users_df["messageCount"] == 0).sum())
        active_users = int((users_df["messageCount"] >= seuil_actif_messages).sum())
    else:
        zero_users = 0
        active_users = int(msgs_df["user_id"].nunique()) if "user_id" in msgs_df.columns else 0

    taux_activation = 100.0 * active_users / max(1, int(membres_total))
    messages_total = int(len(msgs_df))
    conversations_total = int(msgs_df["conversation_id"].nunique()) if "conversation_id" in msgs_df.columns else 0

    split = msgs_df["type_usage"].value_counts().to_dict() if "type_usage" in msgs_df.columns else {}
    messages_agents = int(split.get("Agents personnalisés", 0))
    messages_base = int(split.get("LLM de base", 0))

    cout_par_membre = cout_periode_ttc / max(1, int(membres_total))
    cout_des_inactifs = cout_par_membre * zero_users
    cout_par_actif = (cout_periode_ttc / active_users) if active_users else None
    cout_par_message = (cout_periode_ttc / messages_total) if messages_total else None
    cout_par_conversation = (cout_periode_ttc / conversations_total) if conversations_total else None

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
        "Analyse ROI-oriented : utilisation réelle, adoption des agents, mix LLM, "
        "segmentation utilisateurs et réactivation des comptes inactifs."
    )

    api_key, w_id, base_url_secret = get_required_secrets()

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
        st.subheader("Période")
        mode = st.selectbox("Mode", options=["month", "range"], index=0)

        if mode == "month":
            d0 = st.date_input("Mois", value=date.today().replace(day=1))
            start_param = ym(d0)
            end_param = None
        else:
            d_start = st.date_input("Début", value=date.today().replace(day=1))
            d_end = st.date_input("Fin", value=date.today())

            real_start = min(d_start, d_end)
            real_end = max(d_start, d_end)

            start_param = ymd(real_start)
            end_param = ymd(real_end)

        st.divider()
        st.subheader("Extraction")
        include_inactive = st.checkbox(
            "Inclure les utilisateurs inactifs (messageCount = 0)",
            value=True,
            help="Recommandé pour identifier les comptes à réactiver/désactiver.",
        )
        output_format = st.selectbox("Format", options=["json", "csv"], index=0)

        st.divider()
        st.subheader("Définition 'actif'")
        seuil_actif = st.slider(
            "Seuil messages sur la période",
            min_value=1,
            max_value=50,
            value=1,
            step=1,
        )

        st.divider()
        st.subheader("Coûts (calculés par membre)")
        prix_unitaire_ht = st.number_input("Prix mensuel HT par membre (€)", value=29.0, step=1.0)
        tva_pct = st.number_input("TVA (%)", value=22.0, step=1.0)

        st.divider()
        colA, colB = st.columns(2)
        run = colA.button("Charger", use_container_width=True)

        if colB.button("Vider cache", use_container_width=True):
            st.cache_data.clear()
            st.toast("Cache vidé.")

    if not run:
        st.info("Configure la période puis clique sur **Charger**.")
        return

    with st.spinner("Appel API Dust…"):
        q_users = UsageQuery(
            mode=mode,
            start=start_param,
            end=end_param,
            table="users",
            include_inactive=include_inactive,
        )
        q_msgs = UsageQuery(
            mode=mode,
            start=start_param,
            end=end_param,
            table="assistant_messages",
            include_inactive=include_inactive,
        )
        q_as = UsageQuery(
            mode=mode,
            start=start_param,
            end=end_param,
            table="assistants",
            include_inactive=include_inactive,
        )

        try:
            users_raw = fetch_usage_df(base_url, w_id, api_key, q_users, output_format)
            msgs_raw = fetch_usage_df(base_url, w_id, api_key, q_msgs, output_format)
            as_raw = fetch_usage_df(base_url, w_id, api_key, q_as, output_format)
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
            st.stop()

    users_df = normalize_users(users_raw)
    as_df = normalize_assistants(as_raw)
    msgs_df = normalize_messages(msgs_raw)
    msgs_enriched = enrich_messages(users_df, msgs_df, as_df)
    period_start_dt, period_end_dt = get_period_bounds(mode, start_param, end_param)

    membres_total = int(users_df["user_id"].nunique()) if "user_id" in users_df.columns else int(len(users_df))

    cout_periode = cout_ttc_periode_par_membres(
        mode=mode,
        start_str=start_param,
        end_str=end_param,
        membres_total=membres_total,
        prix_unitaire_ht=float(prix_unitaire_ht),
        tva_pct=float(tva_pct),
    )

    users_seg = compute_weekly_coverage_all_users(users_df, msgs_enriched)

    kpis = compute_kpis(
        users_df=users_df,
        msgs_df=msgs_enriched,
        membres_total=membres_total,
        seuil_actif_messages=int(seuil_actif),
        cout_periode_ttc=float(cout_periode),
    )

    st.caption(f"Coût TTC période calculé sur {membres_total} membres : **{cout_periode:,.2f} €**")

    t_resume, t_agents, t_agents_insights, t_llm, t_users, t_data = st.tabs(
        [
            "Résumé ROI",
            "Agents (publiés / non publiés)",
            "Agents Insights & Nouveaux Agents",
            "Modèles (LLM de base vs via agents)",
            "Utilisateurs & segmentation",
            "Données & exports",
        ]
    )

    # ----------------------------
    # Résumé ROI
    # ----------------------------
    with t_resume:
        st.subheader("KPI principaux")

        a1, a2, a3, a4, a5, a6 = st.columns(6)
        a1.metric("Membres (période)", f"{kpis['membres_total']:,}")
        a2.metric("Utilisateurs observés", f"{kpis['users_total']:,}")
        a3.metric("Utilisateurs actifs", f"{kpis['users_actifs']:,}")
        a4.metric("Utilisateurs à 0", f"{kpis['users_zero']:,}")
        a5.metric("Taux d’activation", f"{kpis['taux_activation_pct']:.1f}%")
        a6.metric("Messages (assistant)", f"{kpis['messages_total']:,}")

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
            if not msgs_enriched.empty and "type_usage" in msgs_enriched.columns:
                split = msgs_enriched["type_usage"].value_counts().reset_index()
                split.columns = ["type_usage", "messages"]
                fig = px.pie(
                    split,
                    names="type_usage",
                    values="messages",
                    hole=0.45,
                    title="Part d’usage — Agents vs LLM de base",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune donnée disponible pour la répartition Agents / LLM de base.")

        with c2:
            if not msgs_enriched.empty and "llm_famille" in msgs_enriched.columns:
                fam = msgs_enriched["llm_famille"].value_counts().reset_index()
                fam.columns = ["famille", "messages"]
                fig2 = px.bar(fam, x="famille", y="messages", title="Messages par famille LLM (global)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Aucune donnée disponible pour les familles LLM.")

        st.divider()
        st.subheader("Tendances (par jour)")

        if not msgs_enriched.empty and {"jour", "type_usage"}.issubset(msgs_enriched.columns):
            daily = (
                msgs_enriched.groupby(["jour", "type_usage"])
                .size()
                .reset_index(name="messages")
                .sort_values("jour")
            )
            fig3 = px.area(
                daily,
                x="jour",
                y="messages",
                color="type_usage",
                title="Messages/jour — Agents vs LLM de base",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Aucune tendance journalière disponible.")

        if not msgs_enriched.empty and {"jour", "user_id"}.issubset(msgs_enriched.columns):
            dau = (
                msgs_enriched.groupby("jour")["user_id"]
                .nunique()
                .reset_index(name="utilisateurs_actifs_jour")
                .sort_values("jour")
            )
            fig4 = px.line(
                dau,
                x="jour",
                y="utilisateurs_actifs_jour",
                title="Utilisateurs actifs/jour (via logs)",
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.subheader("Top 30 utilisateurs")

        if "user_label" in msgs_enriched.columns and not msgs_enriched.empty:
            agg_dict = {"messages": ("user_label", "size")}
            if "conversation_id" in msgs_enriched.columns:
                agg_dict["conversations"] = ("conversation_id", "nunique")
            if "jour" in msgs_enriched.columns:
                agg_dict["dernier_message"] = ("jour", "max")

            top_users = (
                msgs_enriched.groupby("user_label")
                .agg(**agg_dict)
                .reset_index()
                .sort_values("messages", ascending=False)
                .head(30)
            )

            top_users["part_messages_pct"] = (
                100.0 * top_users["messages"] / max(1, int(kpis["messages_total"]))
            ).round(1)

            top_cols_left, top_cols_right = st.columns([1.2, 1])

            with top_cols_left:
                display_cols = [c for c in ["user_label", "messages", "part_messages_pct", "conversations", "dernier_message"] if c in top_users.columns]
                st.dataframe(
                    top_users[display_cols].rename(
                        columns={
                            "user_label": "utilisateur",
                            "part_messages_pct": "% du total",
                        }
                    ),
                    use_container_width=True,
                    height=700,
                )

            with top_cols_right:
                fig_top_users = px.bar(
                    top_users.sort_values("messages", ascending=True),
                    x="messages",
                    y="user_label",
                    orientation="h",
                    title="Top 30 utilisateurs par nombre de messages",
                )
                st.plotly_chart(fig_top_users, use_container_width=True)
        else:
            st.info("Impossible de calculer le top utilisateurs : colonne user_label absente ou aucune donnée.")

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

        if "messages" in as_df.columns and "settings" in as_df.columns:
            published_unused = as_df[
                (as_df["settings"].astype(str).str.lower() == "published")
                & (as_df["messages"] == 0)
            ]

            st.markdown(f"**Agents publiés non utilisés** : {len(published_unused):,}")

            if len(published_unused) > 0:
                cols = [
                    c for c in [
                        "name",
                        "provider",
                        "modelId",
                        "messages",
                        "distinctUsersReached",
                        "distinctConversations",
                        "lastEdit",
                    ]
                    if c in published_unused.columns
                ]
                st.dataframe(
                    published_unused[cols],
                    use_container_width=True,
                    height=260,
                )

    # ----------------------------
    # Agents Insights & Nouveaux Agents
    # ----------------------------
    with t_agents_insights:
        st.subheader("Agents Insights & Nouveaux Agents")
        st.caption(
            "Vue portefeuille agents : adoption, usage, statut de publication, "
            "nouveaux agents et opportunités d’action."
        )

        agents_logs = msgs_enriched[msgs_enriched["type_usage"].eq("Agents personnalisés")].copy()
        agents_catalog = as_df.copy()

        if "name" not in agents_catalog.columns:
            agents_catalog["name"] = pd.Series(dtype="object")

        if "settings" not in agents_catalog.columns:
            agents_catalog["settings"] = "unknown"

        agents_catalog["statut_agent"] = agents_catalog["settings"].apply(agent_status_from_settings)

        if "lastEdit" in agents_catalog.columns:
            agents_catalog["lastEdit_dt"] = pd.to_datetime(agents_catalog["lastEdit"], errors="coerce")
        else:
            agents_catalog["lastEdit_dt"] = pd.NaT

        if "description" in agents_catalog.columns:
            agents_catalog["description_len"] = (
                agents_catalog["description"].fillna("").astype(str).str.len()
            )
        else:
            agents_catalog["description_len"] = 0

        if agents_logs.empty or "assistant_name" not in agents_logs.columns:
            st.info("Aucun usage d’agent détecté sur la période.")
        else:
            usage_agg = {
                "messages_periode": ("assistant_name", "size"),
                "premier_usage": ("created_at", "min"),
                "dernier_usage": ("created_at", "max"),
            }

            if "user_label" in agents_logs.columns:
                usage_agg["utilisateurs_touches"] = ("user_label", "nunique")
            else:
                usage_agg["utilisateurs_touches"] = ("assistant_name", "size")

            if "conversation_id" in agents_logs.columns:
                usage_agg["conversations"] = ("conversation_id", "nunique")
            else:
                usage_agg["conversations"] = ("assistant_name", "size")

            if "jour" in agents_logs.columns:
                usage_agg["jours_actifs"] = ("jour", "nunique")
            else:
                usage_agg["jours_actifs"] = ("assistant_name", "size")

            usage_agents = (
                agents_logs.groupby("assistant_name")
                .agg(**usage_agg)
                .reset_index()
                .rename(columns={"assistant_name": "name"})
            )

            if "statut_publication" in agents_logs.columns:
                status_mix = (
                    agents_logs.groupby(["assistant_name", "statut_publication"])
                    .size()
                    .unstack(fill_value=0)
                    .reset_index()
                    .rename(columns={"assistant_name": "name"})
                )
                usage_agents = usage_agents.merge(status_mix, on="name", how="left")

            agents_portfolio = agents_catalog.merge(usage_agents, on="name", how="outer")

            if "settings" not in agents_portfolio.columns:
                agents_portfolio["settings"] = "unknown"

            if "description_len" not in agents_portfolio.columns:
                agents_portfolio["description_len"] = 0

            if "lastEdit_dt" not in agents_portfolio.columns:
                if "lastEdit" in agents_portfolio.columns:
                    agents_portfolio["lastEdit_dt"] = pd.to_datetime(
                        agents_portfolio["lastEdit"], errors="coerce"
                    )
                else:
                    agents_portfolio["lastEdit_dt"] = pd.NaT

            agents_portfolio["statut_agent"] = agents_portfolio["settings"].apply(agent_status_from_settings)

            for c in [
                "messages",
                "distinctUsersReached",
                "distinctConversations",
                "messages_periode",
                "utilisateurs_touches",
                "conversations",
                "jours_actifs",
                "published",
                "unpublished",
                "description_len",
            ]:
                if c in agents_portfolio.columns:
                    agents_portfolio[c] = pd.to_numeric(
                        agents_portfolio[c], errors="coerce"
                    ).fillna(0)

            if "messages_periode" not in agents_portfolio.columns:
                agents_portfolio["messages_periode"] = 0
            if "utilisateurs_touches" not in agents_portfolio.columns:
                agents_portfolio["utilisateurs_touches"] = 0
            if "conversations" not in agents_portfolio.columns:
                agents_portfolio["conversations"] = 0
            if "jours_actifs" not in agents_portfolio.columns:
                agents_portfolio["jours_actifs"] = 0

            agents_portfolio["messages_par_utilisateur"] = (
                agents_portfolio["messages_periode"]
                / agents_portfolio["utilisateurs_touches"].replace(0, pd.NA)
            ).fillna(0).round(1)

            agents_portfolio["adoption_pct"] = (
                100.0 * agents_portfolio["utilisateurs_touches"] / max(1, int(kpis["users_total"]))
            ).round(1)

            agents_portfolio["insight"] = ""

            agents_portfolio.loc[
                (agents_portfolio["statut_agent"] == "Publié")
                & (agents_portfolio["messages_periode"] == 0),
                "insight",
            ] = "Publié sans usage"

            agents_portfolio.loc[
                (agents_portfolio["statut_agent"] == "Non publié")
                & (agents_portfolio["messages_periode"] > 0),
                "insight",
            ] = "Usage non publié"

            agents_portfolio.loc[
                (agents_portfolio["messages_periode"] >= 50)
                & (agents_portfolio["utilisateurs_touches"] <= 2),
                "insight",
            ] = "Usage concentré"

            agents_portfolio.loc[
                (agents_portfolio["description_len"] == 0)
                & agents_portfolio["insight"].eq(""),
                "insight",
            ] = "Description vide"

            published_used = int(
                (
                    (agents_portfolio["statut_agent"] == "Publié")
                    & (agents_portfolio["messages_periode"] > 0)
                ).sum()
            )
            published_unused = int(
                (
                    (agents_portfolio["statut_agent"] == "Publié")
                    & (agents_portfolio["messages_periode"] == 0)
                ).sum()
            )
            unpublished_used = int(
                (
                    (agents_portfolio["statut_agent"] == "Non publié")
                    & (agents_portfolio["messages_periode"] > 0)
                ).sum()
            )
            active_agents = int((agents_portfolio["messages_periode"] > 0).sum())

            total_agent_messages = float(agents_portfolio["messages_periode"].sum())
            top3_share = 0.0
            if total_agent_messages > 0:
                top3_share = round(
                    100.0
                    * agents_portfolio.sort_values("messages_periode", ascending=False)["messages_periode"]
                    .head(3)
                    .sum()
                    / total_agent_messages,
                    1,
                )

            new_agents = agents_portfolio[
                agents_portfolio["lastEdit_dt"].notna()
                & (agents_portfolio["lastEdit_dt"].dt.normalize() >= period_start_dt)
                & (agents_portfolio["lastEdit_dt"].dt.normalize() <= period_end_dt)
            ].copy()

            i1, i2, i3, i4, i5 = st.columns(5)
            i1.metric("Agents actifs", f"{active_agents:,}")
            i2.metric("Publiés utilisés", f"{published_used:,}")
            i3.metric("Publiés sans usage", f"{published_unused:,}")
            i4.metric("Non publiés utilisés", f"{unpublished_used:,}")
            i5.metric("Nouveaux / édités", f"{len(new_agents):,}")

            if published_unused > 0:
                st.warning(
                    f"{published_unused} agent(s) publiés n'ont généré aucun message sur la période."
                )
            if unpublished_used > 0:
                st.info(
                    f"{unpublished_used} agent(s) non publiés ont été utilisés sur la période."
                )
            if top3_share >= 60:
                st.info(
                    f"L’usage des agents est concentré : les 3 premiers représentent {top3_share}% des messages agents."
                )

            st.divider()

            c1, c2 = st.columns(2)

            with c1:
                status_summary = (
                    agents_portfolio.groupby("statut_agent", dropna=False)
                    .agg(
                        agents=("name", "nunique"),
                        messages=("messages_periode", "sum"),
                    )
                    .reset_index()
                )
                fig_status = px.bar(
                    status_summary,
                    x="statut_agent",
                    y="messages",
                    text="agents",
                    title="Messages agents par statut de publication",
                )
                st.plotly_chart(fig_status, use_container_width=True)

            with c2:
                active_portfolio = agents_portfolio[
                    agents_portfolio["messages_periode"] > 0
                ].copy()

                if not active_portfolio.empty:
                    fig_portfolio = px.scatter(
                        active_portfolio,
                        x="utilisateurs_touches",
                        y="messages_periode",
                        size="conversations",
                        color="statut_agent",
                        hover_name="name",
                        hover_data=[
                            "adoption_pct",
                            "messages_par_utilisateur",
                            "jours_actifs",
                            "insight",
                        ],
                        title="Portefeuille agents — adoption vs volume d’usage",
                    )
                    st.plotly_chart(fig_portfolio, use_container_width=True)
                else:
                    st.info("Pas assez de données pour afficher le portefeuille agents.")

            st.divider()

            g1, g2 = st.columns(2)

            with g1:
                top_agents_insight = (
                    agents_portfolio.sort_values("messages_periode", ascending=False)
                    .head(15)
                    .copy()
                )
                if not top_agents_insight.empty:
                    fig_top_agents = px.bar(
                        top_agents_insight.sort_values("messages_periode", ascending=True),
                        x="messages_periode",
                        y="name",
                        orientation="h",
                        color="statut_agent",
                        title="Top 15 agents par messages",
                    )
                    st.plotly_chart(fig_top_agents, use_container_width=True)

            with g2:
                top_agents_adoption = (
                    agents_portfolio.sort_values(
                        ["utilisateurs_touches", "messages_periode"], ascending=False
                    )
                    .head(15)
                    .copy()
                )
                if not top_agents_adoption.empty:
                    fig_top_adoption = px.bar(
                        top_agents_adoption.sort_values("utilisateurs_touches", ascending=True),
                        x="utilisateurs_touches",
                        y="name",
                        orientation="h",
                        color="statut_agent",
                        title="Top 15 agents par utilisateurs touchés",
                    )
                    st.plotly_chart(fig_top_adoption, use_container_width=True)

            st.divider()

            st.subheader("Insights actionnables")
            a1, a2, a3 = st.columns(3)

            with a1:
                st.markdown("### À rationaliser")
                published_no_usage_df = agents_portfolio[
                    (agents_portfolio["statut_agent"] == "Publié")
                    & (agents_portfolio["messages_periode"] == 0)
                ].copy()
                if published_no_usage_df.empty:
                    st.success("Aucun agent publié sans usage.")
                else:
                    cols = [
                        c
                        for c in [
                            "name",
                            "statut_agent",
                            "lastEdit_dt",
                            "distinctUsersReached",
                            "distinctConversations",
                            "insight",
                        ]
                        if c in published_no_usage_df.columns
                    ]
                    st.dataframe(
                        published_no_usage_df[cols].sort_values("lastEdit_dt", ascending=False),
                        use_container_width=True,
                        height=260,
                    )

            with a2:
                st.markdown("### À gouverner")
                unpublished_usage_df = agents_portfolio[
                    (agents_portfolio["statut_agent"] == "Non publié")
                    & (agents_portfolio["messages_periode"] > 0)
                ].copy()
                if unpublished_usage_df.empty:
                    st.success("Aucun agent non publié utilisé.")
                else:
                    cols = [
                        c
                        for c in [
                            "name",
                            "messages_periode",
                            "utilisateurs_touches",
                            "conversations",
                            "messages_par_utilisateur",
                            "insight",
                        ]
                        if c in unpublished_usage_df.columns
                    ]
                    st.dataframe(
                        unpublished_usage_df[cols].sort_values("messages_periode", ascending=False),
                        use_container_width=True,
                        height=260,
                    )

            with a3:
                st.markdown("### À pousser")
                high_potential_df = agents_portfolio[
                    (agents_portfolio["messages_periode"] > 0)
                    & (agents_portfolio["utilisateurs_touches"] >= 3)
                ].copy()
                if high_potential_df.empty:
                    st.info("Pas encore d’agent avec adoption large sur la période.")
                else:
                    cols = [
                        c
                        for c in [
                            "name",
                            "statut_agent",
                            "messages_periode",
                            "utilisateurs_touches",
                            "adoption_pct",
                            "messages_par_utilisateur",
                        ]
                        if c in high_potential_df.columns
                    ]
                    st.dataframe(
                        high_potential_df[cols].sort_values(
                            ["utilisateurs_touches", "messages_periode"], ascending=False
                        ),
                        use_container_width=True,
                        height=260,
                    )

            st.divider()

            st.subheader("Nouveaux Agents")
            st.caption(
                "Proxy utilisé : `lastEdit` dans la période sélectionnée, faute d’un champ natif de création."
            )

            if new_agents.empty:
                st.info("Aucun nouvel agent ou agent récemment édité sur la période.")
            else:
                new_agents = new_agents.sort_values(
                    ["lastEdit_dt", "messages_periode"], ascending=[False, False]
                )

                n1, n2 = st.columns([1.1, 1])

                with n1:
                    cols = [
                        c
                        for c in [
                            "name",
                            "statut_agent",
                            "lastEdit_dt",
                            "messages_periode",
                            "utilisateurs_touches",
                            "conversations",
                            "adoption_pct",
                            "insight",
                        ]
                        if c in new_agents.columns
                    ]
                    st.dataframe(
                        new_agents[cols].rename(
                            columns={
                                "name": "agent",
                                "statut_agent": "statut",
                                "lastEdit_dt": "dernière_édition",
                                "messages_periode": "messages",
                                "utilisateurs_touches": "utilisateurs",
                                "conversations": "conversations",
                                "adoption_pct": "adoption_%",
                                "insight": "insight",
                            }
                        ),
                        use_container_width=True,
                        height=420,
                    )

                with n2:
                    fig_new_agents = px.bar(
                        new_agents.head(20),
                        x="name",
                        y="messages_periode",
                        color="statut_agent",
                        hover_data=["utilisateurs_touches", "lastEdit_dt", "insight"],
                        title="Usage des nouveaux / récemment édités",
                    )
                    st.plotly_chart(fig_new_agents, use_container_width=True)

            st.divider()

            st.subheader("Table portefeuille agents")
            portfolio_display_cols = [
                c
                for c in [
                    "name",
                    "statut_agent",
                    "messages_periode",
                    "utilisateurs_touches",
                    "conversations",
                    "jours_actifs",
                    "adoption_pct",
                    "messages_par_utilisateur",
                    "lastEdit_dt",
                    "premier_usage",
                    "dernier_usage",
                    "insight",
                ]
                if c in agents_portfolio.columns
            ]

            st.dataframe(
                agents_portfolio[portfolio_display_cols]
                .sort_values(["messages_periode", "utilisateurs_touches"], ascending=False)
                .rename(
                    columns={
                        "name": "agent",
                        "statut_agent": "statut",
                        "messages_periode": "messages",
                        "utilisateurs_touches": "utilisateurs",
                        "conversations": "conversations",
                        "jours_actifs": "jours_actifs",
                        "adoption_pct": "adoption_%",
                        "messages_par_utilisateur": "messages_par_utilisateur",
                        "lastEdit_dt": "dernière_édition",
                        "premier_usage": "premier_usage",
                        "dernier_usage": "dernier_usage",
                        "insight": "insight",
                    }
                ),
                use_container_width=True,
                height=520,
            )

            st.download_button(
                "Télécharger le portefeuille agents (CSV)",
                data=agents_portfolio[portfolio_display_cols].to_csv(index=False).encode("utf-8"),
                file_name="agents_insights_portefeuille.csv",
                mime="text/csv",
            )

    # ----------------------------
    # Modèles
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
                fig = px.pie(
                    fam,
                    names="famille",
                    values="messages",
                    hole=0.45,
                    title="Répartition familles — LLM de base",
                )
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
                fig3 = px.pie(
                    fam2,
                    names="famille",
                    values="messages",
                    hole=0.45,
                    title="Répartition familles — via agents",
                )
                st.plotly_chart(fig3, use_container_width=True)

                top2 = agents_msgs["llm_modele"].value_counts().head(20).reset_index()
                top2.columns = ["modèle", "messages"]
                fig4 = px.bar(top2, x="modèle", y="messages", title="Top 20 modèles via agents (messages)")
                st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.subheader("Tendance familles (par jour)")

        if not msgs_enriched.empty and {"jour", "llm_famille"}.issubset(msgs_enriched.columns):
            daily_fam = (
                msgs_enriched.groupby(["jour", "llm_famille"])
                .size()
                .reset_index(name="messages")
                .sort_values("jour")
            )
            fig5 = px.area(
                daily_fam,
                x="jour",
                y="messages",
                color="llm_famille",
                title="Messages/jour — par famille LLM",
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Aucune tendance journalière disponible pour les familles LLM.")

    # ----------------------------
    # Utilisateurs & segmentation
    # ----------------------------
    with t_users:
        st.subheader("Utilisateurs & segmentation")
        st.caption(
            "Vue simple pour comprendre l’adoption : volume d’usage et régularité sur la période."
        )

        nb_inactifs = int((users_seg["segment_usage"] == "Inactifs (0)").sum())
        nb_faible = int((users_seg["segment_usage"] == "Faible (1–3)").sum())
        nb_moyen = int((users_seg["segment_usage"] == "Moyen (4–20)").sum())
        nb_fort = int((users_seg["segment_usage"] == "Fort (>20)").sum())
        nb_reguliers_max = int((users_seg["regularite_periode"] == "100 % des semaines").sum()) if "regularite_periode" in users_seg.columns else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Inactifs", f"{nb_inactifs:,}")
        k2.metric("Faible (1–3)", f"{nb_faible:,}")
        k3.metric("Moyen (4–20)", f"{nb_moyen:,}")
        k4.metric("Fort (>20)", f"{nb_fort:,}")
        k5.metric("Réguliers 100%", f"{nb_reguliers_max:,}")

        st.divider()

        segment_order = ["Inactifs (0)", "Faible (1–3)", "Moyen (4–20)", "Fort (>20)"]
        segment_counts = (
            users_seg["segment_usage"]
            .value_counts()
            .reindex(segment_order, fill_value=0)
            .reset_index()
        )
        segment_counts.columns = ["segment", "utilisateurs"]

        viz1, viz2 = st.columns(2)

        with viz1:
            fig_seg = px.bar(
                segment_counts,
                x="segment",
                y="utilisateurs",
                title="Répartition des utilisateurs par segment d’usage",
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        with viz2:
            fig_donut = px.pie(
                segment_counts,
                names="segment",
                values="utilisateurs",
                hole=0.5,
                title="Part des segments d’usage",
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        viz3, viz4 = st.columns(2)

        with viz3:
            fig_hist = px.histogram(
                users_seg,
                x="messageCount",
                nbins=30,
                title="Distribution du nombre de messages par utilisateur",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with viz4:
            regularity_order = [
                "0 % des semaines",
                "1 à 25 %",
                "26 à 50 %",
                "51 à 75 %",
                "76 à 99 %",
                "100 % des semaines",
            ]
            regularity_counts = (
                users_seg["regularite_periode"]
                .value_counts()
                .reindex(regularity_order, fill_value=0)
                .reset_index()
            )
            regularity_counts.columns = ["regularite", "utilisateurs"]
            regularity_counts["regularite"] = pd.Categorical(
                regularity_counts["regularite"],
                categories=regularity_order,
                ordered=True,
            )

            fig_regularity = px.bar(
                regularity_counts,
                x="regularite",
                y="utilisateurs",
                category_orders={"regularite": regularity_order},
                title="Régularité sur la période (% de semaines avec au moins 1 usage)",
            )
            st.plotly_chart(fig_regularity, use_container_width=True)

        st.info(
            "Lecture du graphique de régularité : "
            "**100 % des semaines** = l’utilisateur a utilisé Dust au moins une fois toutes les semaines de la période ; "
            "**51 à 75 %** = il a été actif sur environ la moitié à trois quarts des semaines ; "
            "**0 % des semaines** = aucun usage sur la période."
        )

        st.divider()
        st.subheader("Liste globale des utilisateurs")

        table_cols = [
            c
            for c in [
                "user_label",
                "segment_usage",
                "messageCount",
                "activeDaysCount",
                "conversations",
                "semaines_actives",
                "semaines_periode",
                "taux_couverture_semaines_pct",
                "regularite_periode",
                "dernier_message",
                "groups",
            ]
            if c in users_seg.columns
        ]

        users_seg_display = users_seg[table_cols].copy().rename(
            columns={
                "user_label": "utilisateur",
                "segment_usage": "segment",
                "messageCount": "messages",
                "activeDaysCount": "jours_actifs",
                "conversations": "conversations",
                "semaines_actives": "semaines_actives",
                "semaines_periode": "semaines_période",
                "taux_couverture_semaines_pct": "% couverture semaines",
                "regularite_periode": "régularité période",
                "dernier_message": "dernier_message",
                "groups": "groupe",
            }
        )

        sort_col = "messages" if "messages" in users_seg_display.columns else users_seg_display.columns[0]
        st.dataframe(
            users_seg_display.sort_values(sort_col, ascending=False),
            use_container_width=True,
            height=520,
        )

        st.download_button(
            "Télécharger la segmentation utilisateurs (CSV)",
            data=users_seg_display.to_csv(index=False).encode("utf-8"),
            file_name="users_segmentation.csv",
            mime="text/csv",
        )

        if "segment" in users_seg_display.columns:
            zeros = users_seg_display[users_seg_display["segment"] == "Inactifs (0)"]
            st.download_button(
                "Télécharger les inactifs (CSV)",
                data=zeros.to_csv(index=False).encode("utf-8"),
                file_name="utilisateurs_inactifs.csv",
                mime="text/csv",
            )

        if "régularité période" in users_seg_display.columns:
            weekly_regular = users_seg_display[users_seg_display["régularité période"] == "100 % des semaines"]
            st.download_button(
                "Télécharger les utilisateurs réguliers 100% (CSV)",
                data=weekly_regular.to_csv(index=False).encode("utf-8"),
                file_name="utilisateurs_reguliers_100pct.csv",
                mime="text/csv",
            )

    # ----------------------------
    # Données & exports
    # ----------------------------
    with t_data:
        st.subheader("Exports (sans emails)")

        st.markdown("### assistant_messages (nettoyé)")
        keep = [
            c
            for c in [
                "created_at",
                "jour",
                "conversation_id",
                "message_id",
                "user_id",
                "user_label",
                "assistant_name",
                "assistant_id",
                "assistant_settings",
                "type_usage",
                "statut_publication",
                "llm_famille",
                "llm_modele",
                "source",
            ]
            if c in msgs_enriched.columns
        ]
        st.dataframe(msgs_enriched[keep].head(300), use_container_width=True, height=420)

        st.download_button(
            "Télécharger assistant_messages_clean.csv",
            data=msgs_enriched[keep].to_csv(index=False).encode("utf-8"),
            file_name="assistant_messages_clean.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("### users (sans affichage email)")
        keep_u = [
            c for c in ["user_id", "user_name", "messageCount", "activeDaysCount", "lastMessageSent", "groups"]
            if c in users_df.columns
        ]
        st.dataframe(
            users_df[keep_u].sort_values("messageCount", ascending=False) if "messageCount" in users_df.columns else users_df[keep_u],
            use_container_width=True,
            height=420,
        )

        st.download_button(
            "Télécharger users_clean.csv",
            data=users_df[keep_u].to_csv(index=False).encode("utf-8"),
            file_name="users_clean.csv",
            mime="text/csv",
        )

        st.divider()
        st.markdown("### assistants (catalogue)")
        keep_a = [
            c
            for c in [
                "name",
                "description",
                "settings",
                "provider",
                "modelId",
                "messages",
                "distinctUsersReached",
                "distinctConversations",
                "lastEdit",
            ]
            if c in as_df.columns
        ]
        st.dataframe(
            as_df[keep_a].sort_values("messages", ascending=False) if "messages" in as_df.columns else as_df[keep_a],
            use_container_width=True,
            height=420,
        )

        st.download_button(
            "Télécharger assistants.csv",
            data=as_df[keep_a].to_csv(index=False).encode("utf-8"),
            file_name="assistants.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
