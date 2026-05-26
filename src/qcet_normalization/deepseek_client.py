"""Thin OpenAI-compatible client for DeepSeek-R1.

Supports two backends out of the box (both are OpenAI-compatible):

  - "openrouter" (default for legacy reasons): https://openrouter.ai/api/v1
      Provider routing pinned to AtlasCloud (`only: ["atlas-cloud"]`) at fp8.
      Auth: OPENROUTER_API_KEY.
  - "novita" (direct provider, cheaper): https://api.novita.ai/openai
      No routing layer; talks straight to Novita's DeepSeek-R1-0528 endpoint.
      Auth: NOVITA_API_KEY.

Switch backends by exporting LLM_PROVIDER=novita (or =openrouter), or by
passing `provider="novita"` to DeepSeekClient(). Cache keys include the
extra_body, so switching backends naturally invalidates only the differing
entries instead of mixing two providers' answers under one key.

Design goals:
- Zero side effects on import; credentials read only when a call is made.
- Default temperature=0.3 (DeepSeek-R1 is a reasoning model; the DeepSeek
  family expects non-zero temperatures, so we keep it > 0 but pick a low
  value to reduce run-to-run label drift). With temperature > 0 the model
  is still non-deterministic: the JSONL cache is the reproducibility anchor
  — once a (model, system, user, temperature, extra_body) tuple has been
  called, the cached answer is replayed bitwise on re-runs. To re-classify
  from scratch, delete cache/llm_cache.jsonl.
- OpenRouter calibration history (kept as documentation rationale for the
  AtlasCloud pin): Stage-0 calibration on a 30-pair set exposed a large
  fidelity gap between fp8 DeepSeek-R1 endpoints on OpenRouter:
    * AtlasCloud: 83.3% EXACT, avg 1000 completion tokens/call.
      Full reasoning chain → R1 behaves as R1.
    * SiliconFlow: 66.7% EXACT, avg 67 completion tokens/call.
      No reasoning chain → effectively a different model.
  Novita on OpenRouter was skipped at the time, but Novita's *direct* API
  serves DeepSeek-R1-0528 with the full reasoning chain (verified before
  this code path was added).
- JSONL cache keyed by (model, system, user, temperature, extra_body). Cache
  hits short-circuit before any network call. Cache stores the full request
  + response so the replication package is complete.
- Exponential backoff retry on transient errors.
- When json_mode=True and parsing fails, retries up to `parse_retries`
  times before raising; bad responses are NEVER written to cache.

Install once:
    pip install openai

Set before running anything that calls .chat():
    export OPENROUTER_API_KEY=sk-or-v1-...    # OpenRouter mode
    export NOVITA_API_KEY=novita-xxxxx        # Novita mode
    export LLM_PROVIDER=novita                # selects backend (default: openrouter)

The client has ONE public method: `chat(system, user, ...)`. Everything else
is internal.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

def _lazy_openai():
    """Import openai lazily so importing this module doesn't require the pkg."""
    try:
        from openai import OpenAI
        from openai import APIError, APIConnectionError, RateLimitError
    except ImportError as exc:
        raise SystemExit(
            "The 'openai' package is required. Install with: pip install openai"
        ) from exc
    return OpenAI, (APIError, APIConnectionError, RateLimitError)


# Backend registry. Adding a new OpenAI-compatible provider only requires a
# new entry here plus exporting LLM_PROVIDER=<name>.
PROVIDER_CONFIGS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "deepseek/deepseek-r1-0528",
        "env_var": "OPENROUTER_API_KEY",
        # OpenRouter-specific routing: pin AtlasCloud + fp8 for reproducibility
        # (see calibration story in module docstring).
        "extra_body": {
            "provider": {
                "only": ["atlas-cloud"],
                "quantizations": ["fp8"],
            },
        },
    },
    "novita": {
        "base_url": "https://api.novita.ai/openai",
        "model": "deepseek/deepseek-r1-0528",
        "env_var": "NOVITA_API_KEY",
        # Direct provider; no provider-routing layer. Empty extra_body keeps
        # the cache key distinct from the OpenRouter+AtlasCloud entries, so a
        # mid-pipeline backend switch will re-query rather than replay an
        # answer from a different runtime.
        "extra_body": {},
    },
}

DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "novita").lower()
if DEFAULT_PROVIDER not in PROVIDER_CONFIGS:
    raise SystemExit(
        f"LLM_PROVIDER={DEFAULT_PROVIDER!r} is not registered. "
        f"Known: {sorted(PROVIDER_CONFIGS)}"
    )

# Backwards-compatible module-level constants (some scripts import them).
DEFAULT_BASE_URL: str = PROVIDER_CONFIGS[DEFAULT_PROVIDER]["base_url"]
DEFAULT_MODEL: str = PROVIDER_CONFIGS[DEFAULT_PROVIDER]["model"]
DEFAULT_ENV_VAR: str = PROVIDER_CONFIGS[DEFAULT_PROVIDER]["env_var"]
DEFAULT_EXTRA_BODY: dict[str, Any] = PROVIDER_CONFIGS[DEFAULT_PROVIDER]["extra_body"]
DEFAULT_PARSE_RETRIES = 3

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = HERE / "cache" / "llm_cache.jsonl"


_LEADING_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_+-]*)\s*\n?")
_TRAILING_FENCE_RE = re.compile(r"\n?\s*```\s*$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_OBJECT_SPAN_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_json(raw: str) -> tuple[dict[str, Any] | None, str]:
    """Best-effort JSON extraction from a model response.

    Progressive strategy (each step runs only if the previous failed):
      1. Parse as-is.
      2. Strip <think>...</think> reasoning blocks.
      3. Strip leading/trailing Markdown code fences — handles both matched
         (```json ... ```) and unbalanced (leading-only or trailing-only).
      4. Extract the widest {...} span and try parsing that.

    Returns (parsed_dict_or_None, cleaned_text). `parsed` is None if every
    strategy failed.
    """
    candidates: list[str] = []

    s0 = raw.strip()
    candidates.append(s0)

    s1 = _THINK_BLOCK_RE.sub("", s0).strip()
    if s1 != s0:
        candidates.append(s1)

    s2 = _LEADING_FENCE_RE.sub("", s1)
    s2 = _TRAILING_FENCE_RE.sub("", s2).strip()
    if s2 != s1:
        candidates.append(s2)

    m = _OBJECT_SPAN_RE.search(s2)
    if m:
        s3 = m.group(0).strip()
        if s3 != s2:
            candidates.append(s3)

    last_text = candidates[-1]
    for text in candidates:
        try:
            return json.loads(text), text
        except json.JSONDecodeError:
            continue
    return None, last_text


def _hash_key(
    model: str,
    system: str,
    user: str,
    temperature: float,
    extra_body: dict[str, Any] | None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "user": user,
            "t": temperature,
            "extra_body": extra_body or {},
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheEntry:
    key: str
    model: str
    system: str
    user: str
    temperature: float
    extra_body: dict[str, Any]
    response_content: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_prompt_tokens: int | None
    provider: str | None
    created_unix: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CacheEntry":
        return cls(
            key=d["key"],
            model=d["model"],
            system=d["system"],
            user=d["user"],
            temperature=float(d["temperature"]),
            extra_body=d.get("extra_body") or {},
            response_content=d["response_content"],
            finish_reason=d.get("finish_reason"),
            prompt_tokens=d.get("prompt_tokens"),
            completion_tokens=d.get("completion_tokens"),
            cached_prompt_tokens=d.get("cached_prompt_tokens"),
            provider=d.get("provider"),
            created_unix=float(d.get("created_unix", 0.0)),
        )


def _extract_cached_tokens(usage: Any) -> int | None:
    """Read the number of cached-prompt tokens from a usage object.

    Different OpenRouter backends surface this in different fields:
      - OpenAI-style:    usage.prompt_tokens_details.cached_tokens
      - DeepSeek-style:  usage.prompt_cache_hit_tokens
      - None at all when the provider didn't serve from cache.
    """
    if usage is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        ct = getattr(details, "cached_tokens", None)
        if ct is None and isinstance(details, dict):
            ct = details.get("cached_tokens")
        if ct is not None:
            return int(ct)
    ct = getattr(usage, "prompt_cache_hit_tokens", None)
    if ct is not None:
        return int(ct)
    return None


def _extract_provider(completion: Any) -> str | None:
    """OpenRouter sets completion.provider on the top level; optional."""
    p = getattr(completion, "provider", None)
    if isinstance(p, str):
        return p
    return None


@dataclass
class JsonlCache:
    path: Path
    _index: dict[str, CacheEntry] = field(default_factory=dict)
    _loaded: bool = False
    # Thread-safety: all get/put/ensure_loaded go through this single lock.
    # Append-writes to the JSONL file can be larger than PIPE_BUF (a single
    # cached entry with system prompt is ~20KB), so we must not rely on POSIX
    # atomicity — the lock serializes file writes as well as index mutations.
    # RLock is used because `get`/`put` call `ensure_loaded`, which also
    # acquires the lock (would deadlock with a plain Lock).
    _lock: threading.RLock = field(
        default_factory=threading.RLock, compare=False, repr=False
    )

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with open(self.path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = CacheEntry.from_dict(json.loads(line))
                        self._index[entry.key] = entry
            self._loaded = True

    def get(self, key: str) -> CacheEntry | None:
        self.ensure_loaded()
        with self._lock:
            return self._index.get(key)

    def put(self, entry: CacheEntry) -> None:
        self.ensure_loaded()
        with self._lock:
            self._index[entry.key] = entry
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")


@dataclass
class DeepSeekClient:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    cache_path: Path = DEFAULT_CACHE_PATH
    max_retries: int = 5
    parse_retries: int = DEFAULT_PARSE_RETRIES
    default_extra_body: dict[str, Any] = field(
        default_factory=lambda: json.loads(json.dumps(DEFAULT_EXTRA_BODY))
    )
    env_var: str = DEFAULT_ENV_VAR
    # Free-form label used in error messages and logs. Defaults to whatever
    # LLM_PROVIDER resolves to at import time.
    provider_name: str = DEFAULT_PROVIDER
    _client: Any | None = None
    _cache: JsonlCache | None = None
    _transient_errors: tuple[type, ...] = field(default_factory=tuple)
    # Guards lazy initialization of _client and _cache so concurrent callers
    # don't each construct their own OpenAI client / JsonlCache. The OpenAI
    # SDK itself is thread-safe for concurrent `chat.completions.create`
    # calls on a single client instance.
    _init_lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )

    @classmethod
    def for_provider(cls, name: str, **overrides: Any) -> "DeepSeekClient":
        """Build a client preset for a registered backend.

        Example: DeepSeekClient.for_provider("novita") -> Novita-direct.
        Any kw `overrides` (e.g. cache_path) win over the registry defaults.
        """
        name = name.lower()
        if name not in PROVIDER_CONFIGS:
            raise ValueError(
                f"Unknown provider {name!r}. Known: {sorted(PROVIDER_CONFIGS)}"
            )
        cfg = PROVIDER_CONFIGS[name]
        kwargs = dict(
            model=cfg["model"],
            base_url=cfg["base_url"],
            env_var=cfg["env_var"],
            default_extra_body=json.loads(json.dumps(cfg["extra_body"])),
            provider_name=name,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._init_lock:
            if self._client is not None:  # re-check after acquiring
                return self._client
            key = os.environ.get(self.env_var)
            if not key:
                raise RuntimeError(
                    f"{self.env_var} env var not set (provider={self.provider_name}). "
                    f"Export it before running."
                )
            OpenAI, transient = _lazy_openai()
            self._client = OpenAI(api_key=key, base_url=self.base_url)
            self._transient_errors = transient
        return self._client

    def _ensure_cache(self) -> JsonlCache:
        if self._cache is not None:
            return self._cache
        with self._init_lock:
            if self._cache is not None:
                return self._cache
            self._cache = JsonlCache(path=self.cache_path)
        return self._cache

    def _call_once(self, create_kwargs: dict[str, Any]) -> Any:
        """Single API call, with transport-level retry only. Raises on final failure."""
        client = self._ensure_client()
        transient_errors = self._transient_errors or tuple()
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return client.chat.completions.create(**create_kwargs)
            except transient_errors as e:
                last_err = e
                sleep_s = min(2 ** attempt, 60) + random.uniform(0, 1)
                time.sleep(sleep_s)
        raise RuntimeError(
            f"{self.provider_name} API failed after {self.max_retries} retries: {last_err}"
        )

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        json_mode: bool = True,
        use_cache: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a single chat completion.

        Retry policy (distinct from transport-level retries):
          When json_mode=True and the model returns something we can't parse as
          JSON (even after stripping Markdown code fences and <think> blocks),
          we retry the API call up to `parse_retries` times (default 3). Bad
          responses are never written to cache; only a successful parse (or
          json_mode=False) gets persisted. This means re-running the pipeline
          against a cache that has some bad entries will naturally re-query
          only those entries.

        Parameters:
          extra_body: optional override for the provider-level extras. When None,
            the client's default_extra_body is used (fp8 quantization pinned).
            Pass an explicit {} to opt out of any extra_body.

        Returns a dict:
            {
              "content":            str (cleaned model output, JSON-parseable if json_mode=True),
              "parsed":             dict | None (parsed JSON if json_mode=True and parse succeeds),
              "cache_hit":          bool,
              "prompt_tokens":      int | None,
              "completion_tokens":  int | None,
            }
        """
        cache = self._ensure_cache()
        eb = self.default_extra_body if extra_body is None else extra_body
        key = _hash_key(self.model, system, user, temperature, eb)

        if use_cache:
            hit = cache.get(key)
            if hit is not None:
                return self._wrap(hit, cache_hit=True, json_mode=json_mode)

        response_format = {"type": "json_object"} if json_mode else None
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if response_format:
            create_kwargs["response_format"] = response_format
        if eb:
            create_kwargs["extra_body"] = eb

        max_parse_attempts = self.parse_retries if json_mode else 1
        last_bad_content: str = ""
        for parse_attempt in range(1, max_parse_attempts + 1):
            completion = self._call_once(create_kwargs)
            choice = completion.choices[0]
            usage = completion.usage
            raw_content = choice.message.content or ""

            if json_mode:
                parsed, cleaned = _coerce_json(raw_content)
                if parsed is None:
                    last_bad_content = raw_content
                    if parse_attempt < max_parse_attempts:
                        # Short backoff; non-determinism (temp>0) gives a fresh
                        # sample on the next try.
                        time.sleep(0.5 + random.uniform(0, 0.5))
                        continue
                    raise ValueError(
                        f"Non-JSON response after {max_parse_attempts} attempts. "
                        f"Last content (first 200 chars): {last_bad_content[:200]!r}"
                    )
                content_to_store = cleaned
            else:
                parsed = None
                content_to_store = raw_content

            entry = CacheEntry(
                key=key,
                model=self.model,
                system=system,
                user=user,
                temperature=temperature,
                extra_body=eb or {},
                response_content=content_to_store,
                finish_reason=choice.finish_reason,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                cached_prompt_tokens=_extract_cached_tokens(usage),
                provider=_extract_provider(completion),
                created_unix=time.time(),
            )
            cache.put(entry)
            return self._wrap(entry, cache_hit=False, json_mode=json_mode)

        raise RuntimeError("unreachable")  # defensive

    @staticmethod
    def _wrap(entry: CacheEntry, *, cache_hit: bool, json_mode: bool) -> dict[str, Any]:
        content = entry.response_content
        parsed: dict[str, Any] | None = None
        if json_mode:
            parsed, cleaned = _coerce_json(content)
            content = cleaned if parsed is not None else content
        return {
            "content": content,
            "parsed": parsed,
            "cache_hit": cache_hit,
            "prompt_tokens": entry.prompt_tokens,
            "completion_tokens": entry.completion_tokens,
            "cached_prompt_tokens": entry.cached_prompt_tokens,
            "provider": entry.provider,
        }


def _smoke_test() -> None:
    """Minimal connectivity test; only runs if user has a key."""
    if not os.environ.get(DEFAULT_ENV_VAR):
        print(f"{DEFAULT_ENV_VAR} not set; skipping smoke test.")
        return
    client = DeepSeekClient()
    r = client.chat(
        system="You answer only with strict JSON.",
        user='Reply with {"ok": true, "word": "hello"}. Nothing else.',
    )
    print(f"cache_hit={r['cache_hit']} tokens={r['prompt_tokens']}/{r['completion_tokens']}")
    print("parsed:", r["parsed"])
    print("content:", r["content"][:120])


if __name__ == "__main__":
    _smoke_test()
