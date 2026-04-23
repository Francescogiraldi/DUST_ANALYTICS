import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


# =========================================================
# Configuration
# =========================================================
DUST_API_KEY = os.getenv("DUST_API_KEY", "").strip()
DUST_WORKSPACE_ID = os.getenv("DUST_WORKSPACE_ID", "").strip()
DUST_BASE_URL = os.getenv("DUST_BASE_URL", "https://dust.tt").strip().rstrip("/")

# Optionnel
OUTPUT_CSV = os.getenv("DUST_AGENTS_EXPORT_CSV", "")
REQUEST_TIMEOUT = int(os.getenv("DUST_TIMEOUT", "90"))

if not DUST_API_KEY or not DUST_WORKSPACE_ID:
    raise SystemExit(
        "Variables d'environnement manquantes : "
        "DUST_API_KEY et DUST_WORKSPACE_ID sont obligatoires."
    )

SESSION = requests.Session()
SESSION.headers.update(
    {
        "Authorization": f"Bearer {DUST_API_KEY}",
        "User-Agent": "dust-agents-export/1.0",
    }
)


# =========================================================
# Helpers HTTP / parsing
# =========================================================
def parse_any_response(resp: requests.Response) -> Any:
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if "application/json" in ctype:
        return resp.json()

    if "text/csv" in ctype or "application/csv" in ctype:
        return pd.read_csv(pd.io.common.StringIO(resp.text)).to_dict(orient="records")

    # fallback souple
    text = resp.text.strip()
    if not text:
        return None

    try:
        return resp.json()
    except Exception:
        try:
            return pd.read_csv(pd.io.common.StringIO(text)).to_dict(orient="records")
        except Exception:
            return text


def get_json_or_records(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    resp = SESSION.get(
        url,
        params=params or {},
        timeout=REQUEST_TIMEOUT,
        headers={"Accept": "application/json"},
    )
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"{resp.status_code} on GET {url} with params={params}: {(resp.text or '')[:500]}",
            response=resp,
        )
    return parse_any_response(resp)


def find_first_list_of_dicts(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        preferred_keys = [
            "agentConfigurations",
            "agents",
            "items",
            "results",
            "data",
            "rows",
        ]
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value

        for value in payload.values():
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value

    return []


def ensure_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def normalize_scalar(x: Any) -> Any:
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return x


# =========================================================
# Generic JSON helpers
# =========================================================
def first_non_empty(*values: Any) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def deep_get(obj: Any, path: List[str]) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def flatten_json(
    obj: Any,
    prefix: str = "",
    sep: str = "__",
    max_list_items: int = 100,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            out.update(flatten_json(v, key, sep=sep, max_list_items=max_list_items))
        return out

    if isinstance(obj, list):
        if not obj:
            out[prefix] = ""
            return out

        # liste simple
        if all(not isinstance(v, (dict, list)) for v in obj):
            out[prefix] = " | ".join("" if v is None else str(v) for v in obj[:max_list_items])
            return out

        # liste complexe -> JSON brut
        out[prefix] = json.dumps(obj[:max_list_items], ensure_ascii=False)
        return out

    out[prefix] = obj
    return out


def walk_json(obj: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            next_path = f"{path}.{k}" if path else str(k)
            yield from walk_json(v, next_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            next_path = f"{path}[{i}]"
            yield from walk_json(v, next_path)


# =========================================================
# Extraction sémantique agent
# =========================================================
ID_KEYS = ["sId", "id", "assistantId", "assistantConfigurationId", "agentId", "sid"]
NAME_KEYS = ["name", "assistant_name", "displayName"]
DESC_KEYS = ["description"]
MODEL_KEYS = ["modelId", "model", "llmModel"]
PROVIDER_KEYS = ["providerId", "provider"]
UPDATED_KEYS = ["lastEdit", "updatedAt", "updated_at", "editedAt"]
SETTINGS_KEYS = ["settings", "scope", "visibility", "status"]


def extract_agent_id(agent: Dict[str, Any]) -> Optional[str]:
    for k in ID_KEYS:
        v = agent.get(k)
        if v:
            return str(v)
    return None


def extract_agent_name(agent: Dict[str, Any]) -> Optional[str]:
    for k in NAME_KEYS:
        v = agent.get(k)
        if v:
            return str(v)
    return None


def extract_model(agent: Dict[str, Any]) -> Optional[str]:
    for k in MODEL_KEYS:
        v = agent.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            return first_non_empty(v.get("id"), v.get("name"), json.dumps(v, ensure_ascii=False))
    return None


def extract_provider(agent: Dict[str, Any]) -> Optional[str]:
    for k in PROVIDER_KEYS:
        v = agent.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            return first_non_empty(v.get("id"), v.get("name"), json.dumps(v, ensure_ascii=False))
    return None


def extract_status(agent: Dict[str, Any]) -> Optional[str]:
    for k in SETTINGS_KEYS:
        v = agent.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            return json.dumps(v, ensure_ascii=False)
    return None


def find_prompt_fields(agent: Dict[str, Any]) -> Dict[str, str]:
    candidates: Dict[str, str] = {}
    patterns = [
        "systemprompt",
        "system_prompt",
        "instructions",
        "instruction",
        "prompt",
        "system",
    ]

    for path, value in walk_json(agent):
        if not isinstance(value, str):
            continue
        p = path.lower().replace(".", "").replace("_", "")
        if any(token in p for token in patterns):
            text = value.strip()
            if text:
                candidates[path] = text

    # choisir le plus probable / plus riche
    ordered = sorted(candidates.items(), key=lambda kv: len(kv[1]), reverse=True)
    best = ordered[0][1] if ordered else ""
    all_paths = " | ".join(path for path, _ in ordered[:20])

    return {
        "system_prompt": best,
        "system_prompt_paths": all_paths,
    }


def extract_author_info(agent: Dict[str, Any]) -> Dict[str, str]:
    names: List[str] = []
    emails: List[str] = []

    creator_name = None
    creator_email = None

    # cas directs probables
    direct_creator = first_non_empty(
        agent.get("createdBy"),
        agent.get("creator"),
        agent.get("owner"),
        agent.get("author"),
        agent.get("user"),
    )

    if isinstance(direct_creator, dict):
        creator_name = first_non_empty(
            direct_creator.get("name"),
            direct_creator.get("fullName"),
            direct_creator.get("displayName"),
        )
        creator_email = first_non_empty(
            direct_creator.get("email"),
            direct_creator.get("primaryEmail"),
        )

    # parcours générique
    for path, value in walk_json(agent):
        p = path.lower()

        if "author" not in p and "creator" not in p and "owner" not in p and "createdby" not in p:
            continue

        if isinstance(value, dict):
            name = first_non_empty(value.get("name"), value.get("displayName"), value.get("fullName"))
            email = first_non_empty(value.get("email"), value.get("primaryEmail"))
            if name:
                names.append(str(name))
            if email:
                emails.append(str(email))

            if creator_name is None and name:
                creator_name = str(name)
            if creator_email is None and email:
                creator_email = str(email)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    name = first_non_empty(item.get("name"), item.get("displayName"), item.get("fullName"))
                    email = first_non_empty(item.get("email"), item.get("primaryEmail"))
                    if name:
                        names.append(str(name))
                    if email:
                        emails.append(str(email))

                    if creator_name is None and name:
                        creator_name = str(name)
                    if creator_email is None and email:
                        creator_email = str(email)
                elif isinstance(item, str):
                    if "@" in item:
                        emails.append(item)

        elif isinstance(value, str):
            if "@" in value:
                emails.append(value)

    names = list(dict.fromkeys([n for n in names if n]))
    emails = list(dict.fromkeys([e for e in emails if e]))

    return {
        "creator_name_guess": creator_name or "",
        "creator_email_guess": creator_email or "",
        "authors_names": " | ".join(names),
        "authors_emails": " | ".join(emails),
    }


def extract_tools_info(agent: Dict[str, Any]) -> Dict[str, Any]:
    tool_names: List[str] = []
    tool_types: List[str] = []
    tool_paths: List[str] = []
    tool_payloads: List[Dict[str, Any]] = []

    relevant_path_tokens = ["tool", "action", "mcp", "server", "app", "integration"]

    for path, value in walk_json(agent):
        if not isinstance(value, dict):
            continue

        p = path.lower()
        if not any(tok in p for tok in relevant_path_tokens):
            continue

        name = first_non_empty(
            value.get("name"),
            value.get("toolName"),
            value.get("internalName"),
            value.get("displayName"),
            value.get("appName"),
            value.get("serverName"),
        )
        typ = first_non_empty(
            value.get("type"),
            value.get("kind"),
            value.get("actionType"),
            value.get("toolType"),
        )

        if name:
            tool_names.append(str(name))
        if typ:
            tool_types.append(str(typ))

        if name or typ:
            tool_paths.append(path)
            tool_payloads.append(value)

    tool_names = list(dict.fromkeys(tool_names))
    tool_types = list(dict.fromkeys(tool_types))
    tool_paths = list(dict.fromkeys(tool_paths))

    return {
        "connected_tools_count": len(tool_names) if tool_names else len(tool_payloads),
        "connected_tools_names": " | ".join(tool_names),
        "connected_tools_types": " | ".join(tool_types),
        "connected_tools_paths": " | ".join(tool_paths[:50]),
        "connected_tools_json": json.dumps(tool_payloads[:200], ensure_ascii=False),
    }


# =========================================================
# API Dust
# =========================================================
def list_all_agents() -> List[Dict[str, Any]]:
    url = f"{DUST_BASE_URL}/api/v1/w/{DUST_WORKSPACE_ID}/assistant/agent_configurations"
    views = ["workspace", "all", "list", "published", "global"]
    merged: Dict[str, Dict[str, Any]] = {}

    for view in views:
        try:
            payload = get_json_or_records(url, params={"view": view, "withAuthors": "true"})
            items = find_first_list_of_dicts(payload)
        except Exception:
            continue

        for item in items:
            item = ensure_dict(item)
            agent_id = extract_agent_id(item)
            agent_name = extract_agent_name(item)

            dedupe_key = agent_id or f"name::{agent_name}" or f"json::{hash(json.dumps(item, sort_keys=True, default=str))}"
            previous = merged.get(dedupe_key, {})
            merged[dedupe_key] = {**previous, **item}

    return list(merged.values())


def get_full_agent_config(agent_id: str) -> Dict[str, Any]:
    url = f"{DUST_BASE_URL}/api/v1/w/{DUST_WORKSPACE_ID}/assistant/agent_configurations/{agent_id}"
    payload = get_json_or_records(url, params={"variant": "full"})
    if isinstance(payload, dict):
        return payload
    return {}


def get_last_2m_usage_from_analytics_export() -> List[Dict[str, Any]]:
    today = pd.Timestamp.now(tz=timezone.utc).normalize()
    start = (today - pd.DateOffset(months=2)).date().isoformat()
    end = today.date().isoformat()

    url = f"{DUST_BASE_URL}/api/v1/w/{DUST_WORKSPACE_ID}/analytics/export"
    payload = get_json_or_records(
        url,
        params={
            "table": "agents",
            "startDate": start,
            "endDate": end,
        },
    )

    rows = find_first_list_of_dicts(payload)
    if rows:
        return rows

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    return []


def get_last_2m_usage_from_workspace_usage() -> List[Dict[str, Any]]:
    today = pd.Timestamp.now(tz=timezone.utc).normalize()
    start = (today - pd.DateOffset(months=2)).date().isoformat()
    end = today.date().isoformat()

    url = f"{DUST_BASE_URL}/api/v1/w/{DUST_WORKSPACE_ID}/workspace-usage"
    payload = get_json_or_records(
        url,
        params={
            "mode": "range",
            "start": start,
            "end": end,
            "table": "assistants",
            "includeInactive": "true",
            "format": "json",
        },
    )

    rows = find_first_list_of_dicts(payload)
    if rows:
        return rows

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    return []


def get_last_2m_usage() -> List[Dict[str, Any]]:
    # endpoint moderne d’abord, legacy en fallback
    try:
        rows = get_last_2m_usage_from_analytics_export()
        if rows:
            return rows
    except Exception:
        pass

    try:
        rows = get_last_2m_usage_from_workspace_usage()
        if rows:
            return rows
    except Exception:
        pass

    return []


# =========================================================
# Mapping usage
# =========================================================
def extract_usage_name(row: Dict[str, Any]) -> Optional[str]:
    candidates = [
        row.get("name"),
        row.get("agent"),
        row.get("agentName"),
        row.get("assistant_name"),
        row.get("assistantName"),
    ]
    value = first_non_empty(*candidates)
    return str(value) if value else None


def extract_usage_id(row: Dict[str, Any]) -> Optional[str]:
    candidates = [
        row.get("id"),
        row.get("agentId"),
        row.get("assistantId"),
        row.get("sId"),
    ]
    value = first_non_empty(*candidates)
    return str(value) if value else None


def normalize_usage_row(row: Dict[str, Any]) -> Dict[str, Any]:
    def pick(*keys: str) -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
        return None

    return {
        "usage_agent_id": extract_usage_id(row) or "",
        "usage_agent_name": extract_usage_name(row) or "",
        "usage_last_2m_messages": pick("messages", "messageCount", "message_count", "count", "totalMessages") or 0,
        "usage_last_2m_users": pick(
            "distinctUsersReached",
            "distinctUsers",
            "uniqueUsers",
            "users",
            "userCount",
        ) or 0,
        "usage_last_2m_conversations": pick(
            "distinctConversations",
            "conversations",
            "conversationCount",
            "uniqueConversations",
        ) or 0,
        "usage_last_2m_raw_json": json.dumps(row, ensure_ascii=False),
    }


def build_usage_maps(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        norm = normalize_usage_row(row)
        if norm["usage_agent_id"]:
            by_id[norm["usage_agent_id"]] = norm
        if norm["usage_agent_name"]:
            by_name[norm["usage_agent_name"]] = norm

    return by_id, by_name


# =========================================================
# Main export
# =========================================================
def build_agent_export_rows() -> List[Dict[str, Any]]:
    agents = list_all_agents()
    usage_rows = get_last_2m_usage()
    usage_by_id, usage_by_name = build_usage_maps(usage_rows)

    export_rows: List[Dict[str, Any]] = []

    for i, base_agent in enumerate(agents, start=1):
        base_agent = ensure_dict(base_agent)
        agent_id = extract_agent_id(base_agent)
        agent_name = extract_agent_name(base_agent)

        print(f"[{i}/{len(agents)}] Agent: {agent_name or agent_id or 'inconnu'}")

        full_cfg: Dict[str, Any] = {}
        if agent_id:
            try:
                full_cfg = get_full_agent_config(agent_id)
            except Exception as e:
                full_cfg = {"_full_config_error": str(e)}

        merged = {**base_agent}
        if isinstance(full_cfg, dict):
            # on garde aussi la config complète si elle expose plus de champs
            merged = {**merged, **full_cfg}

        prompt_info = find_prompt_fields(merged)
        author_info = extract_author_info(merged)
        tools_info = extract_tools_info(merged)

        usage = {}
        if agent_id and agent_id in usage_by_id:
            usage = usage_by_id[agent_id]
        elif agent_name and agent_name in usage_by_name:
            usage = usage_by_name[agent_name]

        flat = flatten_json(merged)

        curated = {
            "agent_id": agent_id or "",
            "agent_name": agent_name or "",
            "description": first_non_empty(merged.get("description"), merged.get("summary")) or "",
            "status_or_settings": extract_status(merged) or "",
            "model": extract_model(merged) or "",
            "provider": extract_provider(merged) or "",
            "last_updated": first_non_empty(*[merged.get(k) for k in UPDATED_KEYS]) or "",
            "workspace_id": DUST_WORKSPACE_ID,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            **author_info,
            **prompt_info,
            **tools_info,
            **usage,
            "raw_json": json.dumps(merged, ensure_ascii=False),
        }

        row = {**curated}
        for k, v in flat.items():
            safe_key = f"flat__{k}"
            row[safe_key] = normalize_scalar(v)

        export_rows.append(row)

    return export_rows


def main() -> None:
    rows = build_agent_export_rows()
    df = pd.DataFrame(rows)

    # nettoyage léger des colonnes
    preferred_cols = [
        "agent_id",
        "agent_name",
        "description",
        "creator_name_guess",
        "creator_email_guess",
        "authors_names",
        "authors_emails",
        "status_or_settings",
        "model",
        "provider",
        "last_updated",
        "usage_last_2m_messages",
        "usage_last_2m_users",
        "usage_last_2m_conversations",
        "connected_tools_count",
        "connected_tools_names",
        "connected_tools_types",
        "system_prompt",
        "system_prompt_paths",
        "workspace_id",
        "exported_at_utc",
        "raw_json",
    ]

    existing_preferred = [c for c in preferred_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_preferred]
    df = df[existing_preferred + other_cols]

    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_CSV or f"dust_agents_full_export_{now_str}.csv"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV généré : {output_path}")
    print(f"Lignes exportées : {len(df)}")


if __name__ == "__main__":
    main()
