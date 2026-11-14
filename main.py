#!/usr/bin/env python3
"""free-model-aggregator-01 entry: the best free AI model aggregator.

A single-file Python 3.13 server (standard library only) that serves two
client APIs -- OpenAI Chat Completions and Anthropic Messages, JSON and
streaming -- over ten simulated free-tier providers on three wires
(OpenAI, Anthropic, Gemini).

How it works
------------
1. It reads the trial brief at $MAKEFUN_PROMPT_FILE: the provider
   directory, the sealed quality table (every free model's rating and
   capabilities) and the workload manifest (the twenty requests, in
   order, with each one's API, streaming flag, weight and requirements).
2. It probes every provider's catalog route in the provider's own wire
   and builds the one catalog document (``--list`` prints it, and the
   register body carries the same document, byte for byte).
3. It listens on a free port, registers port + catalog with the env, and
   serves the client's requests one at a time.
4. For each request it ranks the free models that meet the request's
   requirements by the quality table's rating, tries providers in that
   order (one attempt per provider per request), and fails over to the
   next candidate when a provider refuses with 402/429/5xx. It learns
   each provider's per-trial budget online: a provider that fails twice
   in a row is treated as spent for the rest of the trial, and a
   conservative shadow budget keeps the best providers from being
   burned on light requests while heavier ones are still to come.
5. It translates requests and answers between wires, preserving exactly
   what the provider echoes back (system prompt, user text, tool names,
   max_tokens, stop sequences, image bytes, JSON mode), and it returns
   the provider's usage counts and stop reason, mapped to the client's
   wire.

Determinism: every choice is a function of the brief, the requests and
the providers' answers so far. No wall clock, no unseeded randomness.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
PROVIDER_TIMEOUT_S = 10
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
STREAM_CHUNK_CHARS = 64

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def string_of(v):
    return v if isinstance(v, str) else ""


def number_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


def join_base(env_url, base):
    """A base path joined to the env URL; an absolute base is used as it stands."""
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    """The headers a provider wants its key in."""
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------------------
# The brief: its three JSON documents, found by schema in the prompt file.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the '}' closing the object that opens at start, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in text that carries a string 'schema', in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


class Provider:
    __slots__ = ("id", "name", "wire", "auth", "base", "key_env", "key")

    def __init__(self, id, name, wire, auth, base, key_env, key):
        self.id = id
        self.name = name
        self.wire = wire
        self.auth = auth
        self.base = base
        self.key_env = key_env
        self.key = key


class RatingRow:
    __slots__ = ("model", "provider", "rating", "context", "tools", "vision", "json", "free")

    def __init__(self, model, provider, rating, context, tools, vision, json_, free):
        self.model = model
        self.provider = provider
        self.rating = rating
        self.context = context
        self.tools = tools
        self.vision = vision
        self.json = json_
        self.free = free


class WorkloadRequest:
    __slots__ = ("id", "api", "stream", "weight", "requires")

    def __init__(self, id, api, stream, weight, requires):
        self.id = id
        self.api = api
        self.stream = stream
        self.weight = weight
        self.requires = requires


def _provider_of(v):
    if not is_record(v):
        return None
    id = string_of(v.get("id"))
    wire = v.get("wire")
    if id == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if string_of(v.get(key)) != "":
            base = string_of(v.get(key))
            break
    name = string_of(v.get("name")) or id
    return Provider(id, name, wire, auth, base, string_of(v.get("key_env")), string_of(v.get("key")))


def _rating_of(v):
    if not is_record(v):
        return None
    model = string_of(v.get("model"))
    provider = string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    rating = rating if isinstance(rating, (int, float)) else 0
    return RatingRow(model, provider, rating, number_or_none(v.get("context_length")),
                     v.get("tools") is True, v.get("vision") is True,
                     v.get("json") is True, v.get("free") is True)


def _workload_request_of(v):
    if not is_record(v):
        return None
    id = string_of(v.get("id"))
    if id == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    mc = requires.get("min_context")
    if isinstance(mc, (int, float)):
        req["min_context"] = mc
    return WorkloadRequest(id, "anthropic" if v.get("api") == "anthropic" else "openai",
                           v.get("stream") is True, 2 if v.get("weight") == 2 else 1, req)


class Brief:
    def __init__(self):
        self.providers = []
        self.ratings = []
        self.best = {}
        self.manifest = []
        self.error = None


def read_brief(path):
    brief = Brief()
    if path == "":
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as err:
        brief.error = "cannot read %s: %s" % (path, err)
        return brief
    docs = documents_in(text)
    directory_doc = None
    for d in docs:
        if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list):
            directory_doc = d
            break
    if directory_doc is None:
        brief.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    for v in directory_doc["providers"]:
        p = _provider_of(v)
        if p is not None:
            brief.providers.append(p)
    for d in docs:
        if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list):
            for v in d["ratings"]:
                r = _rating_of(v)
                if r is not None:
                    brief.ratings.append(r)
            if isinstance(d.get("best"), list):
                for v in d["best"]:
                    if is_record(v) and string_of(v.get("id")) != "":
                        brief.best[string_of(v.get("id"))] = number_or_none(v.get("rating_milli")) or 0
            break
    for d in docs:
        if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list):
            for v in d["manifest"]:
                w = _workload_request_of(v)
                if w is not None:
                    brief.manifest.append(w)
            break
    return brief


def rating_of(brief, provider_id, model_id):
    for r in brief.ratings:
        if r.provider == provider_id and r.model == model_id:
            return r
    return None


# ---------------------------------------------------------------------------
# The catalog: probe each provider's catalog route, in its own wire.
# ---------------------------------------------------------------------------

class ProbedModel:
    __slots__ = ("id", "display_name", "free", "context", "tools", "vision", "json")

    def __init__(self, id, display_name, free, context, tools, vision, json_):
        self.id = id
        self.display_name = display_name
        self.free = free
        self.context = context
        self.tools = tools
        self.vision = vision
        self.json = json_


class Probe:
    __slots__ = ("provider", "ok", "status", "models", "error")

    def __init__(self, provider, ok, status, models, error):
        self.provider = provider
        self.ok = ok
        self.status = status
        self.models = models
        self.error = error


def _zero_priced(v):
    """True when a listing prices every model at zero, None when it says nothing."""
    if not is_record(v):
        return None
    pricing = v.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        id = string_of(v.get("name"))
        if id.startswith("models/"):
            id = id[len("models/"):]
    else:
        id = string_of(v.get("id"))
    if id == "":
        return None
    context = number_or_none(v.get("context_length"))
    if context is None:
        context = number_or_none(v.get("context_window"))
    if context is None:
        context = number_or_none(v.get("inputTokenLimit"))
    display = string_of(v.get("display_name")) or string_of(v.get("displayName")) or id
    free = bool_or_none(v.get("free"))
    if free is None:
        free = _zero_priced(v)
    return ProbedModel(id, display, free, context, bool_or_none(v.get("tools")),
                       bool_or_none(v.get("vision")), bool_or_none(v.get("json")))


def _models_in(wire, body):
    if not is_record(body):
        return []
    lst = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(lst, list):
        return []
    out = []
    for v in lst:
        m = _model_in(wire, v)
        if m is not None:
            out.append(m)
    return out


def _http_get(url, headers, timeout_s):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        try:
            err.read()
        except Exception:
            pass
        return err.code, b""
    except Exception:
        return 0, b""


def probe_provider(env_url, provider, timeout_s=PROBE_TIMEOUT_S):
    base = join_base(env_url, provider.base)
    if provider.wire == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider.auth, provider.key))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    status, raw = _http_get(url, headers, timeout_s)
    if status != 200:
        return Probe(provider, False, status, [], "GET %s answered HTTP %s" % (url, status))
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception as err:
        return Probe(provider, False, status, [], "GET %s is not JSON: %s" % (url, err))
    return Probe(provider, True, status, _models_in(provider.wire, body), None)


def model_is_free(model, row):
    """Whether a model is free: the quality table's row where it has one
    (the sealed truth), the listing's own word where it does not, paid
    when neither says. A paid model is never routed."""
    if row is not None:
        return row.free
    return model.free if model.free is not None else False


def capabilities_of(model, row):
    """What a model can do: the quality table's row where it has one, the
    listing's own fields where it does not (None: neither said)."""
    if row is not None:
        return {"context": row.context, "tools": row.tools, "vision": row.vision, "json": row.json}
    return {"context": model.context, "tools": model.tools, "vision": model.vision, "json": model.json}


def meets_requirements(requires, caps):
    """Whether a model's capabilities meet one request's requirements.
    An unknown capability never meets one."""
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps.get("context") is None or caps.get("context") < mc):
        return False
    return True


def build_catalog(probes, brief):
    """The catalog the probes and the quality table make: every provider
    that answered, every model it listed. Its base_url is the directory's
    own path (/p/<id>), never the path joined to the env URL: the env's
    origin differs between a live pass and its replay, and the catalog is
    part of the state export a replay must reproduce byte for byte."""
    providers = []
    for probe in probes:
        if not probe.ok:
            continue
        seen = set()
        models = []
        for model in sorted(probe.models, key=lambda m: m.id):
            if model.id in seen:
                continue
            seen.add(model.id)
            row = rating_of(brief, probe.provider.id, model.id)
            context = row.context if row is not None and row.context is not None else model.context
            models.append({"id": model.id, "free": model_is_free(model, row), "context_length": context})
        providers.append({"id": probe.provider.id, "base_url": probe.provider.base,
                          "key_env": probe.provider.key_env, "wire": probe.provider.wire,
                          "models": models})
    return {"providers": providers}


# ---------------------------------------------------------------------------
# Routing: rank free candidates by rating, learn budgets online, fail over.
# ---------------------------------------------------------------------------

class Candidate:
    __slots__ = ("provider", "model", "rating", "caps")

    def __init__(self, provider, model, rating, caps):
        self.provider = provider
        self.model = model
        self.rating = rating
        self.caps = caps


class ProviderState:
    """What this trial has taught us about one provider: calls made and
    a shadow budget -- our guess at how many calls it has left. Whether
    it is spent is not guessed: a provider that answers 'budget is
    spent' or 'is out' says so in its error message, and only then is
    it treated as dead. A scheduled refusal ('refused this call') is
    transient -- the provider's next call may succeed -- so it never
    marks a provider dead, however many in a row arrive."""

    SHADOW_BUDGET = 12

    def __init__(self):
        self.calls = 0
        self.successes = 0
        self.consec_fails = 0
        self.dead = False
        self.shadow = self.SHADOW_BUDGET

    def note_attempt(self):
        self.calls += 1
        self.shadow -= 1

    def note_success(self):
        self.successes += 1
        self.consec_fails = 0

    def note_failure(self, kind):
        if kind in ("budget", "outage"):
            self.dead = True
            return
        self.consec_fails += 1


def failure_kind(raw):
    """What a provider's error answer says: 'budget' when its free-tier
    budget is spent, 'outage' when it is out, 'transient' for a
    scheduled refusal or anything else. The three wires spell the
    error envelope differently; the message is the same either way."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return "transient"
    message = ""
    if is_record(body):
        if is_record(body.get("error")):
            message = body["error"].get("message") or ""
        elif isinstance(body.get("message"), str):
            message = body["message"]
    if not isinstance(message, str):
        return "transient"
    if "budget is spent" in message:
        return "budget"
    if message.rstrip().endswith("is out"):
        return "outage"
    return "transient"


class Router:
    def __init__(self, brief, probes):
        self.brief = brief
        self.probes = [p for p in probes if p.ok]
        self.state = {p.provider.id: ProviderState() for p in self.probes}
        # Every free model, per provider, best first. A provider's best
        # model overall may not meet a request's requirements while
        # another of its models does, so the candidate for a request is
        # chosen per request, not once per provider.
        self.free_models = {}
        for probe in self.probes:
            models = []
            for model in probe.models:
                row = rating_of(brief, probe.provider.id, model.id)
                if not model_is_free(model, row):
                    continue
                caps = capabilities_of(model, row)
                rating = row.rating if row is not None else 0
                models.append(Candidate(probe.provider, model, rating, caps))
            models.sort(key=lambda c: -c.rating)
            if models:
                self.free_models[probe.provider.id] = models

    def candidates_for(self, requires, client_api, served_index):
        """The providers whose best free model meets the requirements, in
        the order to try them: rating first, then providers we still
        believe have budget, then the client's own wire (a translation
        avoided is a check that cannot fail). A JSON-mode request on the
        OpenAI wire never goes to an Anthropic-wire provider: that wire
        cannot carry response_format, so its echo would say JSON mode
        was absent."""
        out = []
        for provider_id, models in self.free_models.items():
            for cand in models:
                if not meets_requirements(requires, cand.caps):
                    continue
                if requires.get("json") is True and client_api == "openai" and cand.provider.wire == "anthropic":
                    continue
                out.append(cand)
                break
        st = self.state
        def sort_key(cand):
            state = st[cand.provider.id]
            # Dead providers last; then the rating -- a provider whose
            # shadow merely ran out may still have budget (the shadow
            # is a guess, the budget message is the truth), so a better
            # rating is always worth one try; then a provider we still
            # believe has budget; then the client's own wire (a
            # translation avoided is a check that cannot fail).
            return (1 if state.dead else 0,
                    -cand.rating,
                    0 if state.shadow > 0 else 1,
                    0 if cand.provider.wire == client_api else 1)
        out.sort(key=sort_key)
        # Reservation: a weight-1 request must not burn the last of the
        # best provider a later weight-2 request can only be served by.
        if served_index is not None and out:
            req = self.brief.manifest[served_index] if served_index < len(self.brief.manifest) else None
            if req is not None and req.weight == 1:
                future_heavy = [w for w in self.brief.manifest[served_index + 1:] if w.weight == 2]
                if future_heavy:
                    top = out[0]
                    need = 0
                    for w in future_heavy:
                        alts = [c for c in self.candidates_for(w.requires, w.api, None)
                                if c.rating >= top.rating]
                        providers = {c.provider.id for c in alts}
                        if providers == {top.provider.id}:
                            need += 1
                    if need > 0 and st[top.provider.id].shadow <= need and len(out) > 1:
                        out = out[1:] + [out[0]]
        return out

    def served_models(self):
        """The free models the entry serves, for GET /v1/models."""
        out = []
        for probe in self.probes:
            for model in sorted(probe.models, key=lambda m: m.id):
                row = rating_of(self.brief, probe.provider.id, model.id)
                if not model_is_free(model, row):
                    continue
                out.append({"id": model.id, "provider": probe.provider.id,
                            "display_name": model.display_name})
        return out


# ---------------------------------------------------------------------------
# Wire translation. The provider's answer text is always "ANSWER ECHO<json>":
# the provider echoes what it parsed, and the client compares that echo
# with what it sent, so every translation below preserves exactly the
# fields the echo carries: system text, user text, tool names, max_tokens,
# stop sequences, image bytes and JSON mode. The answers themselves never
# carry tool calls, only text, usage and a stop reason.
# ---------------------------------------------------------------------------

def text_of(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def _data_url_parts(url):
    """A data: URL as (media_type, base64) or (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    head, _, data = url[5:].partition(",")
    if ";" not in head:
        return None, None
    media = head.split(";")[0] or "image/png"
    return media, data


def _oa_user_part(block):
    """One Anthropic content block as an OpenAI content part, or None."""
    if not is_record(block):
        return None
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def _args_object(value):
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if is_record(parsed):
            return parsed
    return {}


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as an OpenAI Chat Completions one."""
    messages = []
    system = text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {"name": block.get("name") if isinstance(block.get("name"), str) else "",
                                 "arguments": json.dumps(_args_object(block.get("input")))}})
            text = text_of(content)
            msg = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({"role": "tool",
                                 "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                                 "content": _tool_result_text(block.get("content"))})
                continue
            part = _oa_user_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = stops
    tools = body.get("tools")
    if isinstance(tools, list):
        converted = []
        for raw in tools:
            if not is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            converted.append({"type": "function",
                              "function": {"name": raw["name"],
                                           "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                                           "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}}}})
        if converted:
            out["tools"] = converted
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _an_image_block(content_part):
    """One OpenAI content part as an Anthropic content block, or None."""
    if not is_record(content_part):
        return None
    if content_part.get("type") == "text" and isinstance(content_part.get("text"), str):
        return {"type": "text", "text": content_part["text"]}
    if content_part.get("type") == "image_url" and is_record(content_part.get("image_url")):
        url = content_part["image_url"].get("url")
        media, data = _data_url_parts(url)
        if data is not None:
            return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
        if isinstance(url, str):
            return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def openai_to_anthropic(body, model):
    """An OpenAI Chat Completions request as an Anthropic Messages one."""
    system_parts = []
    messages = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                blocks.append({"type": "tool_use",
                               "id": call.get("id") if isinstance(call.get("id"), str) else "",
                               "name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                               "input": _args_object(fn.get("arguments"))})
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            blocks = [{"type": "tool_result",
                       "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                       "content": raw.get("content") if isinstance(raw.get("content"), str) else ""}]
            messages.append({"role": "user", "content": blocks})
            continue
        blocks = []
        if isinstance(content, str):
            if content != "":
                blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                block = _an_image_block(part)
                if block is not None:
                    blocks.append(block)
        if blocks:
            messages.append({"role": "user", "content": blocks})
    out = {"model": model}
    system = "\n".join(system_parts)
    if system != "":
        out["system"] = system
    out["messages"] = messages
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        out["max_tokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else ([s for s in stop if isinstance(s, str)] if isinstance(stop, list) else [])
    if stops:
        out["stop_sequences"] = stops
    tools = body.get("tools")
    if isinstance(tools, list):
        converted = []
        for raw in tools:
            if not is_record(raw) or not is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            converted.append({"name": fn["name"],
                              "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                              "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}}})
        if converted:
            out["tools"] = converted
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif is_record(choice) and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
        out["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_image_part(url):
    media, data = _data_url_parts(url)
    if data is not None:
        return {"inlineData": {"mimeType": media, "data": data}}
    if isinstance(url, str):
        return {"fileData": {"fileUri": url}}
    return None


def _gemini_text_part(text):
    return {"text": text}


def _gemini_contents_from_oa(body):
    """OpenAI messages as Gemini contents (system handled separately)."""
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            continue
        if role == "assistant":
            parts = []
            text = text_of(content)
            if text != "":
                parts.append(_gemini_text_part(text))
            for call in raw.get("tool_calls") or []:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                fn = call["function"]
                parts.append({"functionCall": {"name": fn.get("name") if isinstance(fn.get("name"), str) else "",
                                               "args": _args_object(fn.get("arguments"))}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            parts = [{"functionResponse": {"name": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                                           "response": {"result": raw.get("content") if isinstance(raw.get("content"), str) else ""}}}]
            contents.append({"role": "user", "parts": parts})
            continue
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append(_gemini_text_part(content))
        elif isinstance(content, list):
            for part in content:
                if not is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(_gemini_text_part(part["text"]))
                elif part.get("type") == "image_url" and is_record(part.get("image_url")):
                    img = _gemini_image_part(part["image_url"].get("url"))
                    if img is not None:
                        parts.append(img)
        if parts:
            contents.append({"role": "user", "parts": parts})
    return contents


def _gemini_contents_from_an(body):
    """Anthropic messages as Gemini contents (system handled separately)."""
    contents = []
    for raw in body.get("messages") or []:
        if not is_record(raw):
            continue
        content = raw.get("content")
        if raw.get("role") == "assistant":
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append(_gemini_text_part(content))
            elif isinstance(content, list):
                for block in content:
                    if not is_record(block):
                        continue
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        parts.append(_gemini_text_part(block["text"]))
                    elif block.get("type") == "tool_use":
                        parts.append({"functionCall": {"name": block.get("name") if isinstance(block.get("name"), str) else "",
                                                       "args": _args_object(block.get("input"))}})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        parts = []
        if isinstance(content, str):
            if content != "":
                parts.append(_gemini_text_part(content))
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(_gemini_text_part(block["text"]))
                elif block.get("type") == "image" and is_record(block.get("source")):
                    source = block["source"]
                    if source.get("type") == "base64" and isinstance(source.get("data"), str):
                        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
                        parts.append({"inlineData": {"mimeType": media, "data": source["data"]}})
                    elif source.get("type") == "url" and isinstance(source.get("url"), str):
                        parts.append({"fileData": {"fileUri": source["url"]}})
                elif block.get("type") == "tool_result":
                    parts.append({"functionResponse": {"name": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                                                       "response": {"result": _tool_result_text(block.get("content"))}}})
        if parts:
            contents.append({"role": "user", "parts": parts})
    return contents


def _gemini_tools_from_oa(body):
    tools = body.get("tools")
    if not isinstance(tools, list):
        return None
    decls = []
    for raw in tools:
        if not is_record(raw) or not is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        decls.append({"name": fn["name"],
                      "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                      "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}}})
    return [{"functionDeclarations": decls}] if decls else None


def _gemini_tools_from_an(body):
    tools = body.get("tools")
    if not isinstance(tools, list):
        return None
    decls = []
    for raw in tools:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        decls.append({"name": raw["name"],
                      "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                      "parameters": raw.get("input_schema") if is_record(raw.get("input_schema")) else {"type": "object", "properties": {}}})
    return [{"functionDeclarations": decls}] if decls else None


def _gemini_generation_config(body, json_mode):
    config = {}
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    if isinstance(body.get("temperature"), (int, float)):
        config["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        config["topP"] = body["top_p"]
    stop = body.get("stop")
    if stop is None:
        stop = body.get("stop_sequences")
    stops = [stop] if isinstance(stop, str) else ([s for s in stop if isinstance(s, str)] if isinstance(stop, list) else [])
    if stops:
        config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    return config


def openai_to_gemini(body):
    """An OpenAI request as a Gemini generateContent body."""
    system_parts = []
    for raw in body.get("messages") or []:
        if is_record(raw) and raw.get("role") == "system":
            text = text_of(raw.get("content"))
            if text != "":
                system_parts.append(text)
    out = {"contents": _gemini_contents_from_oa(body)}
    system = "\n".join(system_parts)
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = _gemini_tools_from_oa(body)
    if tools is not None:
        out["tools"] = tools
    fmt = body.get("response_format")
    json_mode = is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")
    config = _gemini_generation_config(body, json_mode)
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body):
    """An Anthropic request as a Gemini generateContent body."""
    out = {"contents": _gemini_contents_from_an(body)}
    system = text_of(body.get("system"))
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = _gemini_tools_from_an(body)
    if tools is not None:
        out["tools"] = tools
    config = _gemini_generation_config(body, False)
    if config:
        out["generationConfig"] = config
    return out


# ---------------------------------------------------------------------------
# Responses: read a provider's answer (JSON or SSE) as (text, input tokens,
# output tokens), and write it back in the client's wire.
# ---------------------------------------------------------------------------

def _num_or_zero(v):
    return v if isinstance(v, (int, float)) else 0


def _answer_text(content):
    """A provider answer's text: a string, or its text blocks joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b["text"] for b in content
                       if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    return ""


def parse_openai_answer(raw):
    """An OpenAI completion (JSON bytes) as (text, input, output)."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return "", 0, 0
    if not is_record(body):
        return "", 0, 0
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    text = _answer_text(message.get("content"))
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return text, _num_or_zero(usage.get("prompt_tokens")), _num_or_zero(usage.get("completion_tokens"))


def parse_anthropic_answer(raw):
    """An Anthropic message (JSON bytes) as (text, input, output)."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return "", 0, 0
    if not is_record(body):
        return "", 0, 0
    text = _answer_text(body.get("content"))
    usage = body.get("usage") if is_record(body.get("usage")) else {}
    return text, _num_or_zero(usage.get("input_tokens")), _num_or_zero(usage.get("output_tokens"))


def parse_gemini_answer(raw):
    """A Gemini generateContent answer (JSON bytes) as (text, input, output)."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return "", 0, 0
    if not is_record(body):
        return "", 0, 0
    candidates = body.get("candidates")
    cand = candidates[0] if isinstance(candidates, list) and candidates and is_record(candidates[0]) else {}
    content = cand.get("content") if is_record(cand.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(p.get("text", "") for p in parts if is_record(p) and isinstance(p.get("text"), str))
    meta = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return text, _num_or_zero(meta.get("promptTokenCount")), _num_or_zero(meta.get("candidatesTokenCount"))


def _sse_frames(raw):
    """An SSE body as a list of (event, data-string) frames."""
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    frames = []
    for frame in text.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def parse_openai_sse(raw):
    """An OpenAI event stream as (text, input, output)."""
    text = ""
    in_tokens = 0
    out_tokens = 0
    for _event, payload in _sse_frames(raw):
        if payload.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            in_tokens = _num_or_zero(chunk["usage"].get("prompt_tokens"))
            out_tokens = _num_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            continue
        delta = choices[0].get("delta") if is_record(choices[0].get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
    return text, in_tokens, out_tokens


def parse_anthropic_sse(raw):
    """An Anthropic event stream as (text, input, output)."""
    text = ""
    in_tokens = 0
    out_tokens = 0
    for event, payload in _sse_frames(raw):
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if not is_record(data):
            continue
        if event == "message_start" and is_record(data.get("message")) and is_record(data["message"].get("usage")):
            in_tokens = _num_or_zero(data["message"]["usage"].get("input_tokens"))
        elif event == "content_block_delta" and is_record(data.get("delta")):
            if data["delta"].get("type") == "text_delta" and isinstance(data["delta"].get("text"), str):
                text += data["delta"]["text"]
        elif event == "message_delta":
            if is_record(data.get("usage")):
                out_tokens = _num_or_zero(data["usage"].get("output_tokens"))
    return text, in_tokens, out_tokens


def parse_gemini_sse(raw):
    """A Gemini SSE stream as (text, input, output)."""
    text = ""
    in_tokens = 0
    out_tokens = 0
    for _event, payload in _sse_frames(raw):
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if not is_record(data):
            continue
        candidates = data.get("candidates")
        if isinstance(candidates, list) and candidates and is_record(candidates[0]):
            content = candidates[0].get("content") if is_record(candidates[0].get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for p in parts:
                if is_record(p) and isinstance(p.get("text"), str):
                    text += p["text"]
        meta = data.get("usageMetadata")
        if is_record(meta):
            in_tokens = _num_or_zero(meta.get("promptTokenCount"))
            out_tokens = _num_or_zero(meta.get("candidatesTokenCount"))
    return text, in_tokens, out_tokens


def split_text(text):
    """A text in stream-sized pieces."""
    if text == "":
        return [""]
    return [text[i:i + STREAM_CHUNK_CHARS] for i in range(0, len(text), STREAM_CHUNK_CHARS)]


def openai_completion_json(text, in_tokens, out_tokens, model):
    return {"id": "chatcmpl-sim", "object": "chat.completion", "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens,
                      "total_tokens": in_tokens + out_tokens}}


def anthropic_message_json(text, in_tokens, out_tokens, model):
    return {"id": "msg_sim", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens}}


def openai_sse(text, in_tokens, out_tokens, model):
    """A Chat Completions event stream: role chunk, content deltas, the
    finish chunk, the usage chunk, then [DONE]."""
    def chunk(delta, finish):
        c = {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
             "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return "data: %s\n\n" % json.dumps(c, separators=(",", ":"))
    frames = [chunk({"role": "assistant", "content": ""}, None)]
    for piece in split_text(text):
        frames.append(chunk({"content": piece}, None))
    frames.append(chunk({}, "stop"))
    usage_chunk = {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                   "model": model, "choices": [],
                   "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens,
                             "total_tokens": in_tokens + out_tokens}}
    frames.append("data: %s\n\n" % json.dumps(usage_chunk, separators=(",", ":")))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_sse(text, in_tokens, out_tokens, model):
    """An Anthropic event stream: message_start, one text block's
    start/deltas/stop, message_delta with the stop reason and output
    usage, then message_stop."""
    def event(name, data):
        return "event: %s\ndata: %s\n\n" % (name, json.dumps(data, separators=(",", ":")))
    frames = [event("message_start", {"type": "message_start",
                                       "message": {"id": "msg_stream", "type": "message",
                                                   "role": "assistant", "model": model, "content": [],
                                                   "stop_reason": None, "stop_sequence": None,
                                                   "usage": {"input_tokens": in_tokens, "output_tokens": 0}}}),
              event("content_block_start", {"type": "content_block_start", "index": 0,
                                            "content_block": {"type": "text", "text": ""}})]
    for piece in split_text(text):
        frames.append(event("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                    "delta": {"type": "text_delta", "text": piece}}))
    frames.append(event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(event("message_delta", {"type": "message_delta",
                                          "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                          "usage": {"output_tokens": out_tokens}}))
    frames.append(event("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def _sse_event(event, data):
    if event is None:
        return "data: %s\n\n" % json.dumps(data, separators=(",", ":"))
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Provider calls.
# ---------------------------------------------------------------------------

def _http_post(url, headers, body_bytes, timeout_s):
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        try:
            raw = err.read()
        except Exception:
            raw = b""
        return err.code, raw
    except Exception:
        return 0, b""


def call_provider(candidate, client_body, client_api, stream):
    """One attempt at one provider: (status, raw answer bytes). The body
    is translated into the provider's wire; an OpenAI streaming request
    always asks for usage, so its answer carries the counts the client
    checks (the echo is unaffected by stream_options)."""
    provider = candidate.provider
    model = candidate.model.id
    base = join_base(ENV_URL, provider.base)
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers(provider.auth, provider.key))
    wire = provider.wire
    if wire == "openai":
        if client_api == "anthropic":
            body = anthropic_to_openai(client_body, model)
        else:
            body = dict(client_body)
            body["model"] = model
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        else:
            body.pop("stream", None)
            body.pop("stream_options", None)
        if stream:
            headers["Accept"] = "text/event-stream"
        return _http_post(base + "/v1/chat/completions", headers,
                          json.dumps(body).encode("utf-8"), PROVIDER_TIMEOUT_S)
    if wire == "anthropic":
        if client_api == "openai":
            body = openai_to_anthropic(client_body, model)
        else:
            body = dict(client_body)
            body["model"] = model
        if not stream:
            body.pop("stream", None)
        headers["anthropic-version"] = "2023-06-01"
        if stream:
            headers["Accept"] = "text/event-stream"
        return _http_post(base + "/v1/messages", headers,
                          json.dumps(body).encode("utf-8"), PROVIDER_TIMEOUT_S)
    # gemini
    if client_api == "openai":
        body = openai_to_gemini(client_body)
    else:
        body = anthropic_to_gemini(client_body)
    from urllib.parse import quote
    action = "streamGenerateContent" if stream else "generateContent"
    url = "%s/v1beta/models/%s:%s" % (base, quote(model, safe=""), action)
    if stream:
        headers["Accept"] = "text/event-stream"
    return _http_post(url, headers, json.dumps(body).encode("utf-8"), PROVIDER_TIMEOUT_S)


def translate_answer(candidate, client_api, raw, stream):
    """A provider's answer in the client's wire: (status, content type,
    body bytes). Same-wire streaming answers pass through untouched;
    anything else is parsed to (text, usage) and rewritten."""
    wire = candidate.provider.wire
    model = candidate.model.id
    if client_api == "openai":
        if wire == "openai":
            return 200, "application/json" if not stream else "text/event-stream", raw
        if stream:
            if wire == "anthropic":
                text, in_tokens, out_tokens = parse_anthropic_sse(raw)
            else:
                text, in_tokens, out_tokens = parse_gemini_sse(raw)
            return 200, "text/event-stream", openai_sse(text, in_tokens, out_tokens, model).encode("utf-8")
        if wire == "anthropic":
            text, in_tokens, out_tokens = parse_anthropic_answer(raw)
        else:
            text, in_tokens, out_tokens = parse_gemini_answer(raw)
        return 200, "application/json", json.dumps(
            openai_completion_json(text, in_tokens, out_tokens, model), separators=(",", ":")).encode("utf-8")
    # client_api == "anthropic"
    if wire == "anthropic":
        return 200, "application/json" if not stream else "text/event-stream", raw
    if stream:
        if wire == "openai":
            text, in_tokens, out_tokens = parse_openai_sse(raw)
        else:
            text, in_tokens, out_tokens = parse_gemini_sse(raw)
        return 200, "text/event-stream", anthropic_sse(text, in_tokens, out_tokens, model).encode("utf-8")
    if wire == "openai":
        text, in_tokens, out_tokens = parse_openai_answer(raw)
    else:
        text, in_tokens, out_tokens = parse_gemini_answer(raw)
    return 200, "application/json", json.dumps(
        anthropic_message_json(text, in_tokens, out_tokens, model), separators=(",", ":")).encode("utf-8")


def requirements_from_body(body):
    """Requirements read off a request body, when the manifest has no row
    for it: tools, images and JSON mode as the body itself shows them."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format")
    return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
            "vision": vision,
            "json": is_record(fmt) and fmt.get("type") in ("json_object", "json_schema")}


# ---------------------------------------------------------------------------
# The server: routes, register, and the /work poll that ends the trial.
# ---------------------------------------------------------------------------

def openai_error_json(message, type_, code):
    return {"error": {"message": message, "type": type_, "code": code}}


def anthropic_error_json(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


def openai_model_list(models):
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                      "owned_by": m["provider"]} for m in models]}


def anthropic_model_list(models):
    return {"data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                      "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
            "has_more": False}


class Aggregator:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.catalog = catalog
        self.router = Router(brief, probes)
        self.manifest_index = 0
        self.lock = threading.Lock()

    def served_models(self):
        return self.router.served_models()

    def serve(self, client_api, body):
        """One client request: try the ranked candidates in order until
        one answers, then return its answer in the client's wire."""
        with self.lock:
            served_index = self.manifest_index
            self.manifest_index += 1
            stream = body.get("stream") is True
            if served_index < len(self.brief.manifest):
                requires = self.brief.manifest[served_index].requires
            else:
                requires = requirements_from_body(body)
                served_index = None
            candidates = self.router.candidates_for(requires, client_api, served_index)
            for cand in candidates:
                state = self.router.state[cand.provider.id]
                state.note_attempt()
                status, raw = call_provider(cand, body, client_api, stream)
                if status == 200:
                    state.note_success()
                    return translate_answer(cand, client_api, raw, stream)
                state.note_failure(failure_kind(raw))
            if client_api == "anthropic":
                return 503, "application/json", json.dumps(
                    anthropic_error_json("overloaded_error", "no free model on the directory meets this request"),
                    separators=(",", ":")).encode("utf-8")
            return 503, "application/json", json.dumps(
                openai_error_json("no free model on the directory meets this request",
                                  "invalid_request_error", "no_free_model"),
                separators=(",", ":")).encode("utf-8")


AGGREGATOR = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, raw):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if raw:
            self.wfile.write(raw)

    def _send_json(self, status, value):
        self._send(status, "application/json",
                   json.dumps(value, separators=(",", ":")).encode("utf-8"))

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            models = AGGREGATOR.served_models()
            if self.headers.get("x-api-key") is not None:
                self._send_json(200, anthropic_model_list(models))
            else:
                self._send_json(200, openai_model_list(models))
            return
        self._send_json(404, openai_error_json("no route GET %s" % self.path, "invalid_request_error", "not_found"))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            client_api = "openai"
        elif path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send_json(404, openai_error_json("no route POST %s" % path, "invalid_request_error", "not_found"))
            return
        body = self._read_json()
        if body is None:
            if client_api == "anthropic":
                self._send_json(400, anthropic_error_json("invalid_request_error", "the request body is not one JSON object"))
            else:
                self._send_json(400, openai_error_json("the request body is not one JSON object", "invalid_request_error", None))
            return
        try:
            status, content_type, raw = AGGREGATOR.serve(client_api, body)
        except Exception as err:
            log("serve failed: %s" % err)
            if client_api == "anthropic":
                self._send_json(500, anthropic_error_json("api_error", "the server failed to answer"))
            else:
                self._send_json(500, openai_error_json("the server failed to answer", "api_error", None))
            return
        self._send(status, content_type, raw)


def register(port, catalog):
    """POST /register with the port and the one catalog."""
    url = ENV_URL + "/register"
    payload = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}).encode("utf-8")
    for attempt in range(2):
        status, raw = _http_post(url, {"Content-Type": "application/json"}, payload, 5)
        if status == 200:
            try:
                reply = json.loads(raw.decode("utf-8"))
            except Exception:
                reply = {}
            if is_record(reply) and reply.get("ok") is True:
                if isinstance(reply.get("requests"), (int, float)):
                    log("registered: the workload holds %s requests" % reply["requests"])
                return True
        log("POST %s answered HTTP %s (attempt %d)" % (url, status, attempt + 1))
        time.sleep(0.5)
    return False


def poll_work(server, stop):
    """Poll GET /work; exit 0 the moment it says the workload is done.
    A process still running when the trial's wall ends scores zero, so
    never wait for anything else."""
    while not stop.is_set():
        try:
            status, raw = _http_get(ENV_URL + "/work", {}, 2)
            if status == 200:
                try:
                    work = json.loads(raw.decode("utf-8"))
                except Exception:
                    work = None
                if is_record(work) and work.get("done") is True:
                    stop.set()
                    server.shutdown()
                    return
        except Exception:
            pass
        stop.wait(POLL_INTERVAL_S)


def main():
    global AGGREGATOR
    brief = read_brief(PROMPT_FILE)
    if brief.error is not None:
        log(brief.error)
    probes = []
    for provider in brief.providers:
        probe = probe_provider(ENV_URL, provider)
        probes.append(probe)
        if not probe.ok and probe.error is not None:
            log(probe.error)
    catalog = build_catalog(probes, brief)
    AGGREGATOR = Aggregator(brief, probes, catalog)
    if "--list" in sys.argv:
        # The same document the register body carries, byte for byte:
        # print the one built object and stop before any server starts.
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        sys.exit(0)
    if ENV_URL == "":
        log("MAKEFUN_ENV_URL is not set")
        sys.exit(1)
    import signal
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        sys.exit(1)
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, len(catalog["providers"]), len(AGGREGATOR.served_models())))
    if not register(port, catalog):
        log("could not register with the env")
        sys.exit(1)
    stop = threading.Event()

    def _term(_signum, _frame):
        stop.set()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    poller = threading.Thread(target=poll_work, args=(server, stop), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()
    sys.exit(0)


if __name__ == "__main__":
    main()
