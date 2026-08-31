"""
dashboard_server.py — Hardened Multi-Threaded REST API & Web Dashboard Server

Zero-external-dependency HTTP server that hosts the Single Page Application
and provides security-hardened REST API endpoints for RAG queries, live defense
telemetry, controlled poisoning simulation, and local LLM integration.

Security Hardening Implemented:
- Dynamic CORS allowlisting with explicit origin validation (no wildcard '*')
- Full suite of HTTP security headers (CSP, X-Content-Type-Options, X-Frame-Options, etc.)
- In-memory thread-safe sliding-window IP rate limiter with HTTP 429 response
- Request payload size enforcement (MAX_CONTENT_LENGTH) with HTTP 413 response
- Server-side JSON schema validation and sanitization on all endpoints
- SSRF URL validation for local LLM backend endpoints
- Safe generic error handling without internal trace or path exposure
"""
import os
import json
import time
import urllib.parse
from dataclasses import asdict
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple, Set

from dashboard.llm_client import GLOBAL_LLM_CLIENT, validate_endpoint_url
from dashboard.rag_defense_engine import GLOBAL_RAG_ENGINE
from unified_rag_defense.topics_data import TOPICS

TOPIC_MAP = {i: t for i, t in enumerate(TOPICS)}
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Environment & Security Configuration
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 1048576))  # 1MB default
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", 180))
ALLOWED_ORIGINS_ENV = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8899,http://localhost:8899"
)
ALLOWED_ORIGINS_SET: Set[str] = {o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()}

VALID_ATTACK_TYPES = {
    "clean", "standard", "pidp", "collusion", "collusion_minor",
    "collusion_major", "collusion_stealth", "silent"
}

VALID_SYSTEMS = {
    "omniguard", "vanilla_rag", "drs_only", "shieldrag", "raguard", "trishield"
}

VALID_PROVIDERS = {"builtin", "ollama", "openai_compat"}


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window IP rate limiter."""
    def __init__(self, max_requests: int = 180, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        import threading
        self._lock = threading.Lock()
        self._clients: Dict[str, List[float]] = {}

    def is_allowed(self, client_ip: str) -> Tuple[bool, int]:
        now = time.time()
        with self._lock:
            # Periodic cleanup if map grows large
            if len(self._clients) > 5000:
                cutoff = now - self.window_seconds
                self._clients = {
                    ip: [t for t in ts if t > cutoff]
                    for ip, ts in self._clients.items()
                    if any(t > cutoff for t in ts)
                }

            timestamps = self._clients.get(client_ip, [])
            valid_ts = [t for t in timestamps if t > now - self.window_seconds]

            if len(valid_ts) >= self.max_requests:
                oldest = valid_ts[0]
                retry_after = max(1, int(self.window_seconds - (now - oldest)))
                self._clients[client_ip] = valid_ts
                return False, retry_after

            valid_ts.append(now)
            self._clients[client_ip] = valid_ts
            return True, 0


GLOBAL_RATE_LIMITER = SlidingWindowRateLimiter(max_requests=RATE_LIMIT_PER_MINUTE, window_seconds=60.0)


class DashboardAPIHandler(SimpleHTTPRequestHandler):
    """Hardened HTTP request handler with REST API routing, CORS, and security controls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def _get_client_ip(self) -> str:
        """Extract client IP address safely."""
        return self.client_address[0] if self.client_address else "127.0.0.1"

    def _is_origin_allowed(self, origin: str) -> bool:
        """Validate whether the incoming Origin header is permitted."""
        if not origin:
            return False
        if origin in ALLOWED_ORIGINS_SET:
            return True
        try:
            parsed = urlparse(origin)
            if parsed.scheme in ("http", "https") and parsed.hostname in ("127.0.0.1", "localhost", "::1"):
                return True
        except Exception:
            return False
        return False

    def end_headers(self):
        """Append hardened HTTP security and dynamic CORS headers."""
        # 1. Content Security Policy & Anti-Sniffing / Anti-Clickjacking
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Permissions-Policy", "geolocation=(), camera=(), microphone=(), payment=()")
        self.send_header("X-XSS-Protection", "1; mode=block")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self' http://localhost:* http://127.0.0.1:*; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';"
        )

        # 2. Dynamic CORS Origin Allowlist Check
        origin = self.headers.get("Origin")
        if origin and self._is_origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Vary", "Origin")

        # 3. Cache Control
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self.end_headers()

    def _send_json(self, data: Any, status: int = 200, extra_headers: Optional[Dict[str, str]] = None):
        """Helper to serialize and transmit JSON payload."""
        try:
            body = json.dumps(data, indent=2).encode("utf-8")
        except Exception:
            body = b'{"error": "Internal JSON serialization error", "status": "error"}'
            status = 500

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, message: str, status: int = 400, extra_headers: Optional[Dict[str, str]] = None):
        """Helper to return standardized structured error messages."""
        self._send_json({"error": message, "status": "error"}, status=status, extra_headers=extra_headers)

    def _parse_json_body(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parses JSON request body with strict size enforcement and format validation.
        Returns: (parsed_dict, error_message)
        """
        content_length_header = self.headers.get("Content-Length")
        if not content_length_header:
            return {}, None

        try:
            content_length = int(content_length_header)
        except ValueError:
            return None, "Invalid Content-Length header."

        if content_length < 0:
            return None, "Invalid Content-Length header."

        if content_length > MAX_CONTENT_LENGTH:
            return None, f"Payload Too Large. Maximum allowed size is {MAX_CONTENT_LENGTH} bytes."

        if content_length == 0:
            return {}, None

        try:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(raw_body), None
        except UnicodeDecodeError:
            return None, "Invalid UTF-8 encoding in request body."
        except json.JSONDecodeError:
            return None, "Malformed JSON syntax in request body."
        except Exception:
            return None, "Failed to read request body."

    def _check_rate_limit(self) -> bool:
        """Enforces rate limiting. Returns True if allowed, False if blocked."""
        client_ip = self._get_client_ip()
        allowed, retry_after = GLOBAL_RATE_LIMITER.is_allowed(client_ip)
        if not allowed:
            self._send_error(
                "Too Many Requests. Rate limit exceeded.",
                status=429,
                extra_headers={"Retry-After": str(retry_after)}
            )
            return False
        return True

    def do_GET(self):
        """Route GET requests for static files and REST endpoints."""
        if not self._check_rate_limit():
            return

        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._handle_status()
        elif path == "/api/topics":
            self._handle_topics()
        elif path == "/api/llm/status":
            self._handle_llm_status()
        else:
            # Serve static files (index.html, styles.css, app.js)
            if path in ("", "/"):
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        """Route POST requests with rate limiting and payload validation."""
        if not self._check_rate_limit():
            return

        parsed = urlparse(self.path)
        path = parsed.path

        body, err = self._parse_json_body()
        if err:
            status_code = 413 if "Payload Too Large" in err else 400
            self._send_error(err, status=status_code)
            return

        if body is None:
            body = {}

        try:
            if path == "/api/chat":
                self._handle_chat(body)
            elif path == "/api/compare":
                self._handle_compare(body)
            elif path == "/api/inject":
                self._handle_inject(body)
            elif path == "/api/corpus/reset":
                self._handle_corpus_reset(body)
            elif path == "/api/llm/test":
                self._handle_llm_test(body)
            elif path == "/api/llm/config":
                self._handle_llm_config(body)
            else:
                self._send_error(f"Unknown endpoint: {path}", status=404)
        except Exception as e:
            # Safe generic error logging without leaking internals to client
            self._send_error("An unexpected error occurred while processing the request.", status=500)

    # --- Endpoint Handlers with Strict Validation ---

    def _handle_status(self):
        llm_status = GLOBAL_LLM_CLIENT.probe_all()
        engine = GLOBAL_RAG_ENGINE
        total_clean = len(engine.world.clean_docs)
        custom_poisons = len(engine.custom_poison_docs)
        trust_entries = len(engine.trust_store.scores)

        res = {
            "status": "healthy",
            "uptime_ms": round((time.time() - getattr(self.server, "start_time", time.time())) * 1000),
            "corpus": {
                "clean_docs_count": total_clean,
                "topics_count": len(TOPICS),
                "custom_poison_count": custom_poisons,
                "trust_store_entries": trust_entries
            },
            "defense_thresholds": {
                "ring0_repetition_threshold": 0.50,
                "ring1_variance_fraction": 0.40,
                "ring1_drs_threshold": round(float(engine.drs.threshold), 3),
                "ring2_risk_cohesion_threshold": 0.55,
                "ring2_contention_threshold": 0.15
            },
            "llm": llm_status
        }
        self._send_json(res)

    def _handle_topics(self):
        topics_list = []
        for i, t in enumerate(TOPICS):
            sample_queries = [
                f"{t['name'].replace('_', ' ').title()}: What is the key result regarding {t['keywords'][0]}?",
                f"How does {t['keywords'][1]} function in {t['name'].replace('_', ' ')}?",
                f"Explain the mechanism of {t['keywords'][2]}."
            ]
            topics_list.append({
                "topic_id": i,
                "name": t["name"],
                "display_name": t["name"].replace("_", " ").title(),
                "answer": t["answer"],
                "wrong_answer": t["wrong_answer"],
                "keywords": t["keywords"],
                "sample_queries": sample_queries
            })
        self._send_json({"topics": topics_list})

    def _handle_llm_status(self):
        status = GLOBAL_LLM_CLIENT.probe_all()
        self._send_json(status)

    def _handle_llm_test(self, body: Dict[str, Any]):
        provider = str(body.get("provider", "ollama")).strip().lower()
        url = body.get("url")
        api_key = str(body.get("api_key", "")).strip()

        if provider not in VALID_PROVIDERS:
            self._send_error(f"Invalid provider '{provider}'. Must be one of: {list(VALID_PROVIDERS)}")
            return

        if url:
            valid, url_err = validate_endpoint_url(str(url))
            if not valid:
                self._send_error(f"Invalid endpoint URL: {url_err}")
                return

        if provider == "ollama":
            from dashboard.llm_client import OllamaClient
            client = OllamaClient(base_url=url or "http://localhost:11434")
            available, msg = client.is_available()
            models = client.list_models() if available else []
            self._send_json({"available": available, "message": msg, "models": models})
        elif provider == "openai_compat":
            from dashboard.llm_client import OpenAICompatibleClient
            client = OpenAICompatibleClient(base_url=url or "http://localhost:1234/v1", api_key=api_key)
            available, msg = client.is_available()
            models = client.list_models() if available else []
            self._send_json({"available": available, "message": msg, "models": models})
        else:
            self._send_json({"available": True, "message": "Built-in Synthesizer Ready", "models": ["builtin-omniguard-v1"]})

    def _handle_llm_config(self, body: Dict[str, Any]):
        provider = str(body.get("provider", "builtin")).strip().lower()
        url = body.get("url")
        model = body.get("model")
        api_key = body.get("api_key")

        if provider not in VALID_PROVIDERS:
            self._send_error(f"Invalid provider '{provider}'. Must be one of: {list(VALID_PROVIDERS)}")
            return

        if url:
            valid, url_err = validate_endpoint_url(str(url))
            if not valid:
                self._send_error(f"Invalid endpoint URL: {url_err}")
                return

        if model and len(str(model)) > 128:
            self._send_error("Model name exceeds maximum length of 128 characters.")
            return

        GLOBAL_LLM_CLIENT.set_config(provider=provider, url=url, model=model, api_key=api_key)
        self._send_json({"status": "updated", "config": GLOBAL_LLM_CLIENT.probe_all()})

    def _handle_chat(self, body: Dict[str, Any]):
        query_text = str(body.get("query", "")).strip()
        if not query_text:
            self._send_error("Query text cannot be empty.")
            return

        if len(query_text) > 2000:
            self._send_error("Query text exceeds maximum length of 2000 characters.")
            return

        topic_id = body.get("topic_id")
        if topic_id is not None:
            try:
                topic_id = int(topic_id)
                if topic_id < 0 or topic_id >= len(TOPICS):
                    self._send_error(f"Invalid topic_id '{topic_id}'. Must be between 0 and {len(TOPICS)-1}.")
                    return
            except (ValueError, TypeError):
                self._send_error("topic_id must be a valid integer.")
                return

        attack_type = str(body.get("attack_type", "clean")).strip().lower()
        if attack_type not in VALID_ATTACK_TYPES:
            self._send_error(f"Invalid attack_type '{attack_type}'. Must be one of: {list(VALID_ATTACK_TYPES)}")
            return

        try:
            k_poison = int(body.get("k_poison", 3))
            if k_poison < 1 or k_poison > 20:
                self._send_error("k_poison must be between 1 and 20.")
                return
        except (ValueError, TypeError):
            self._send_error("k_poison must be an integer.")
            return

        adversarial_suffix = body.get("adversarial_suffix")
        if adversarial_suffix is not None:
            adversarial_suffix = str(adversarial_suffix).strip()
            if len(adversarial_suffix) > 1000:
                self._send_error("adversarial_suffix exceeds maximum length of 1000 characters.")
                return

        persist_trust = bool(body.get("persist_trust", True))
        system_choice = str(body.get("system", "omniguard")).strip().lower()
        if system_choice not in VALID_SYSTEMS:
            self._send_error(f"Invalid system '{system_choice}'. Must be one of: {list(VALID_SYSTEMS)}")
            return

        try:
            temperature = float(body.get("temperature", 0.2))
            if temperature < 0.0 or temperature > 2.0:
                self._send_error("temperature must be between 0.0 and 2.0.")
                return
        except (ValueError, TypeError):
            self._send_error("temperature must be a valid floating-point number.")
            return

        engine = GLOBAL_RAG_ENGINE
        # 1. Create Query object
        query_obj = engine.create_query_object(
            query_text=query_text,
            topic_id=topic_id,
            adversarial_suffix=adversarial_suffix
        )
        matched_topic_data = TOPIC_MAP.get(query_obj.topic_id, TOPICS[0])

        # 2. Prepare candidate document pool with selected attack
        active_query, pool = engine.prepare_attack_pool(
            query=query_obj,
            attack_type=attack_type,
            k_poison=k_poison
        )

        # 3. Execute defense system
        if system_choice == "omniguard":
            exec_result = engine.run_omniguard_with_telemetry(active_query, pool, persist_trust=persist_trust)
            retrieved_docs_for_llm = exec_result.retrieved_docs
            telemetry_data = asdict(exec_result.telemetry) if exec_result.telemetry else None
        else:
            # Baseline selection
            from unified_rag_defense.baselines import (
                vanilla_rag, drs_only, shieldrag_only, raguard_zkip, trishield
            )
            t0 = time.time()
            if system_choice == "vanilla_rag":
                res = vanilla_rag(active_query, pool, engine.world)
                system_name = "Vanilla RAG (No Defense)"
            elif system_choice == "drs_only":
                res = drs_only(active_query, pool, engine.drs, engine.world)
                system_name = "DRS Filter Only"
            elif system_choice == "shieldrag":
                res = shieldrag_only(active_query, pool, engine.world)
                system_name = "ShieldRAG Only"
            elif system_choice == "raguard":
                res = raguard_zkip(active_query, pool, engine.world)
                system_name = "RAGuard / ZKIP"
            elif system_choice == "trishield":
                res = trishield(active_query, pool, engine.world, engine.centroid)
                system_name = "TriShieldRAG"
            else:
                res = vanilla_rag(active_query, pool, engine.world)
                system_name = "Vanilla RAG"

            elapsed_ms = (time.time() - t0) * 1000.0
            # Retrieve top-k for LLM context
            from unified_rag_defense.retrieval import effective_embedding, top_k
            q_emb = effective_embedding(active_query, engine.world)
            entries = top_k(q_emb, pool, k=5)
            retrieved_docs_for_llm = [
                {
                    "rank": r + 1,
                    "doc_id": d.doc_id,
                    "cosine_similarity": round(float(s), 3),
                    "trust_score": round(float(d.trust_score), 3),
                    "claim_answer": d.answer,
                    "is_poison": d.is_poison,
                    "text_snippet": d.text[:140] + "...",
                    "text": d.text,
                    "label": d.label
                }
                for r, (d, s) in enumerate(entries)
            ]
            telemetry_data = None
            exec_result = type("ExecResult", (), {
                "system_name": system_name,
                "answer": res.answer,
                "calls": res.calls,
                "route": "fixed",
                "is_correct": res.answer == query_obj.correct_answer,
                "is_attack_success": res.answer == "ATTACKER_TARGET",
                "latency_ms": round(elapsed_ms, 2),
                "retrieved_docs": retrieved_docs_for_llm
            })()

        # 4. Local LLM Grounded Generation
        llm_gen = GLOBAL_LLM_CLIENT.generate(
            query_text=active_query.text,
            context_docs=retrieved_docs_for_llm,
            defense_name=exec_result.system_name,
            determined_answer=exec_result.answer,
            topic_name=matched_topic_data["name"],
            temperature=temperature
        )

        response_payload = {
            "query": {
                "raw_text": query_text,
                "sanitized_text": active_query.text,
                "topic_id": query_obj.topic_id,
                "topic_name": matched_topic_data["name"],
                "topic_display_name": matched_topic_data["name"].replace("_", " ").title(),
                "ground_truth_answer": query_obj.correct_answer,
                "has_adversarial_suffix": bool(query_obj.suffix_text)
            },
            "attack": {
                "attack_type": attack_type,
                "k_poison": k_poison,
                "suffix_applied": bool(active_query.suffix_text)
            },
            "defense": {
                "system_name": exec_result.system_name,
                "determined_answer": exec_result.answer,
                "calls": exec_result.calls,
                "route": exec_result.route,
                "is_correct": exec_result.is_correct,
                "is_attack_success": exec_result.is_attack_success,
                "pipeline_latency_ms": exec_result.latency_ms
            },
            "telemetry": telemetry_data,
            "retrieved_documents": retrieved_docs_for_llm,
            "llm_generation": {
                "text": llm_gen.text,
                "provider": llm_gen.provider,
                "model": llm_gen.model,
                "latency_ms": round(llm_gen.latency_ms, 2),
                "prompt_tokens": llm_gen.prompt_tokens,
                "completion_tokens": llm_gen.completion_tokens,
                "is_fallback": llm_gen.is_fallback,
                "error": llm_gen.error
            }
        }
        self._send_json(response_payload)

    def _handle_compare(self, body: Dict[str, Any]):
        query_text = str(body.get("query", "")).strip()
        if not query_text:
            self._send_error("Query text cannot be empty.")
            return

        if len(query_text) > 2000:
            self._send_error("Query text exceeds maximum length of 2000 characters.")
            return

        topic_id = body.get("topic_id")
        if topic_id is not None:
            try:
                topic_id = int(topic_id)
                if topic_id < 0 or topic_id >= len(TOPICS):
                    self._send_error(f"Invalid topic_id '{topic_id}'.")
                    return
            except (ValueError, TypeError):
                self._send_error("topic_id must be an integer.")
                return

        attack_type = str(body.get("attack_type", "clean")).strip().lower()
        if attack_type not in VALID_ATTACK_TYPES:
            self._send_error(f"Invalid attack_type '{attack_type}'.")
            return

        try:
            k_poison = int(body.get("k_poison", 3))
            if k_poison < 1 or k_poison > 20:
                self._send_error("k_poison must be between 1 and 20.")
                return
        except (ValueError, TypeError):
            self._send_error("k_poison must be an integer.")
            return

        adversarial_suffix = body.get("adversarial_suffix")
        if adversarial_suffix is not None:
            adversarial_suffix = str(adversarial_suffix).strip()
            if len(adversarial_suffix) > 1000:
                self._send_error("adversarial_suffix exceeds maximum length of 1000 characters.")
                return

        engine = GLOBAL_RAG_ENGINE
        query_obj = engine.create_query_object(
            query_text=query_text,
            topic_id=topic_id,
            adversarial_suffix=adversarial_suffix
        )
        matched_topic_data = TOPIC_MAP.get(query_obj.topic_id, TOPICS[0])

        active_query, pool = engine.prepare_attack_pool(
            query=query_obj,
            attack_type=attack_type,
            k_poison=k_poison
        )

        comparison_matrix = engine.run_side_by_side_comparison(active_query, pool)

        self._send_json({
            "query": {
                "text": active_query.text,
                "topic_name": matched_topic_data["name"].replace("_", " ").title(),
                "ground_truth_answer": query_obj.correct_answer,
                "attack_type": attack_type,
                "k_poison": k_poison
            },
            "systems_comparison": comparison_matrix
        })

    def _handle_inject(self, body: Dict[str, Any]):
        try:
            topic_id = int(body.get("topic_id", 0))
            if topic_id < 0 or topic_id >= len(TOPICS):
                self._send_error(f"Invalid topic_id '{topic_id}'. Must be between 0 and {len(TOPICS)-1}.")
                return
        except (ValueError, TypeError):
            self._send_error("topic_id must be a valid integer.")
            return

        text = str(body.get("text", "")).strip()
        target_answer = str(body.get("target_answer", "ATTACKER_TARGET")).strip()

        if not text:
            self._send_error("Poison document text cannot be empty.")
            return

        if len(text) > 5000:
            self._send_error("Poison document text exceeds maximum length of 5000 characters.")
            return

        if len(target_answer) > 128:
            self._send_error("Target answer exceeds maximum length of 128 characters.")
            return

        engine = GLOBAL_RAG_ENGINE
        doc = engine.inject_custom_poison(topic_id=topic_id, text=text, target_answer=target_answer)
        self._send_json({
            "status": "injected",
            "doc_id": doc.doc_id,
            "topic_id": doc.topic_id,
            "target_answer": doc.answer,
            "total_custom_poisons": len(engine.custom_poison_docs)
        })

    def _handle_corpus_reset(self, body: Dict[str, Any]):
        GLOBAL_RAG_ENGINE.reset_trust_store()
        self._send_json({"status": "reset_successful", "message": "Trust store and custom poisons cleared."})


def run_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Start the multi-threaded HTTP server."""
    os.makedirs(STATIC_DIR, exist_ok=True)
    server = ThreadingHTTPServer((host, port), DashboardAPIHandler)
    server.start_time = time.time()
    return server


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = run_server("127.0.0.1", port)
    print(f"Hardened Server listening on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
