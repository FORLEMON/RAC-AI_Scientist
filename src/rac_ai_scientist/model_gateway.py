"""Per-episode Chat Completions gateway with independent lifecycle budgets.

The evaluated host receives an ephemeral gateway credential, never the Azure
resource key. Accounting uses conservative Azure retail rates, ignoring cache
discounts. Streaming consumers get equivalent SSE from a complete response so
provider usage is known before releasing it. Every condition uses this gateway.
"""
from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import threading
import time
import urllib.error
import urllib.request


class Redactor:
    def __init__(self, values=()):
        self.values = {str(x) for x in values if x}

    def __call__(self, text):
        for value in sorted(tuple(self.values), key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
        text = re.sub(r"\b(?:rit_|clp_|rmt_|rmg_|sni_)[A-Za-z0-9_-]+", "[REDACTED]", text)
        return text


class BudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    def __init__(self, message, status, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class Account:
    def __init__(self, budget, pricing):
        self.budget = budget
        self.pricing = pricing
        self.calls = self.input_tokens = self.output_tokens = 0
        self.cost = 0.0
        self.deadline = time.monotonic() + budget['max_wall_seconds']

    def cost_of(self, inp, out):
        return (inp * self.pricing['input_usd_per_million'] + out * self.pricing['output_usd_per_million']) / 1_000_000

    def reserve(self, payload):
        if time.monotonic() >= self.deadline or self.calls >= self.budget['max_agent_calls']:
            raise BudgetExceeded('wall-time or model-call limit reached')
        if payload.get('n', 1) != 1:
            raise ValueError('only one completion per request is permitted')
        # UTF-8 bytes plus chat/tool framing is a conservative text-token bound.
        estimate = len(json.dumps({k: payload[k] for k in ('messages', 'tools', 'tool_choice') if k in payload}, ensure_ascii=False).encode()) + 1024
        estimate += 64 * len(payload.get('messages', []))
        requested = int(payload.get('max_completion_tokens') or payload.get('max_tokens') or 8192)
        output = min(requested, self.budget['max_output_tokens'] - self.output_tokens)
        remaining = self.budget['max_provider_cost_usd'] - self.cost - self.cost_of(estimate, 0)
        output = min(output, int(max(0, remaining) * 1_000_000 / self.pricing['output_usd_per_million']))
        if estimate > self.budget['max_input_tokens'] - self.input_tokens or output < 1:
            raise BudgetExceeded('token or USD limit reached')
        self.calls += 1
        return estimate, output

    def finish(self, inp, out):
        self.input_tokens += inp
        self.output_tokens += out
        cost = self.cost_of(inp, out)
        self.cost += cost
        return cost

    def snapshot(self):
        return {'calls': self.calls, 'input_tokens': self.input_tokens, 'output_tokens': self.output_tokens,
                'budget_cost_usd': self.cost, 'cost_source': 'conservative_azure_retail_estimate', 'pricing': self.pricing}


def sse_response(body):
    base = {k: body[k] for k in ('id', 'created', 'model', 'system_fingerprint') if k in body}
    base['object'] = 'chat.completion.chunk'
    for choice in body.get('choices', []):
        delta = dict(choice['message'])
        if delta.get('tool_calls'):
            delta['tool_calls'] = [dict(item, index=index) for index, item in enumerate(delta['tool_calls'])]
        yield {**base, 'choices': [{'index': choice.get('index', 0), 'delta': delta, 'finish_reason': None}]}
        yield {**base, 'choices': [{'index': choice.get('index', 0), 'delta': {}, 'finish_reason': choice['finish_reason']}]}
    yield {**base, 'choices': [], 'usage': body.get('usage', {})}


class ModelGateway:
    def __init__(self, *, base_url, api_key, model, budget, pricing, log_dir: Path, bind, redactor):
        self.base_url, self.api_key, self.model = base_url.rstrip('/'), api_key, model
        self.account = Account(budget, pricing)
        self.log_dir = log_dir
        self.bind = bind
        self.token = 'rmg_' + secrets.token_urlsafe(32)
        self.redact = redactor
        self.redact.values.update((self.token, api_key))
        self.lock = threading.Lock()
        self.exhausted = threading.Event()
        self.server = None

    def save(self, name, data):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / name).write_text(self.redact(json.dumps(data, ensure_ascii=False, indent=2)) + '\n', encoding='utf-8')

    def complete(self, payload):
        with self.lock:
            if payload.get('model', self.model).removeprefix('openai/') != self.model:
                raise ValueError('episode model cannot change')
            payload = dict(payload, model=self.model, stream=False)
            payload.pop('stream_options', None)
            # OpenHands sends an OpenAI cache hint that Azure DeepSeek rejects.
            # Removing the hint does not change messages, tools or generation.
            payload.pop('prompt_cache_key', None)
            inp, out = self.account.reserve(payload)
            payload.pop('max_completion_tokens', None)
            payload['max_tokens'] = out
            index = self.account.calls
            self.save(f'{index:06d}.request.json', payload)
            started = time.monotonic()
            try:
                request = urllib.request.Request(self.base_url + '/chat/completions', data=json.dumps(payload).encode(),
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key})
                with urllib.request.urlopen(request, timeout=max(1, min(300, self.account.deadline - time.monotonic()))) as response:
                    body = json.load(response)
                usage = body.get('usage') or {}
                if not all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
                    raise ValueError('provider omitted usage; reserving full request budget')
            except Exception as exc:
                # On ambiguous failures, charge the entire reservation rather
                # than assuming an upstream request was free.
                rate_limited = isinstance(exc, urllib.error.HTTPError) and exc.code == 429
                # An explicit 429 was rejected before generation. Preserve its
                # retry guidance without inventing usage for this rejection.
                if not rate_limited:
                    self.account.finish(inp, out)
                detail = exc.read().decode('utf-8', 'replace') if isinstance(exc, urllib.error.HTTPError) else str(exc)
                response_headers = {k:v for k,v in exc.headers.items() if 'ratelimit' in k.lower() or 'retry' in k.lower()} if isinstance(exc, urllib.error.HTTPError) else {}
                self.save(f'{index:06d}.error.json', {'type': type(exc).__name__, 'error': detail, 'http_status': getattr(exc, 'code', None),
                    'rate_headers': response_headers, 'reserved_input': inp, 'reserved_output': out, 'reservation_charged': not rate_limited})
                self.save('usage.json', self.account.snapshot())
                if rate_limited:
                    raise ProviderError(self.redact(detail), 429, exc.headers.get('Retry-After', '60')) from None
                raise RuntimeError(self.redact(detail)) from None
            cost = self.account.finish(usage['prompt_tokens'], usage['completion_tokens'])
            body['response_cost'] = cost
            self.save(f'{index:06d}.response.json', body)
            self.save('usage.json', {**self.account.snapshot(), 'last_request_seconds': time.monotonic() - started})
            return body

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + owner.token):
                    self.send_error(403)
                    return
                if self.path.rstrip('/') != '/v1/chat/completions':
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 32 * 1024 * 1024:
                        raise ValueError('invalid body size')
                    payload = json.loads(self.rfile.read(length))
                    streaming = bool(payload.get('stream'))
                    body = owner.complete(payload)
                    self.send_response(200)
                    if streaming:
                        data = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in sse_response(body)) + 'data: [DONE]\n\n'
                        self.send_header('Content-Type', 'text/event-stream')
                    else:
                        data = json.dumps(body)
                        self.send_header('Content-Type', 'application/json')
                except Exception as exc:
                    owner.save(f'gateway-error-{time.time_ns()}.json', {'type': type(exc).__name__, 'error': str(exc)})
                    if isinstance(exc, BudgetExceeded):
                        owner.exhausted.set()
                    self.send_response(exc.status if isinstance(exc, ProviderError) else 429 if isinstance(exc, BudgetExceeded) else 502)
                    if isinstance(exc, ProviderError) and exc.retry_after:
                        self.send_header('Retry-After', exc.retry_after)
                    self.send_header('Content-Type', 'application/json')
                    data = json.dumps({'error': {'message': owner.redact(str(exc)), 'type': type(exc).__name__}})
                try:
                    encoded = data.encode()
                    self.send_header('Content-Length', str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer((self.bind, 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.save('usage.json', self.account.snapshot())
        return f'http://{self.bind}:{self.server.server_port}/v1'

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
