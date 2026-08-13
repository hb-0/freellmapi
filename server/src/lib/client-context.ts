import { AsyncLocalStorage } from 'async_hooks';
import type { NextFunction, Request, Response } from 'express';
import { classifyClientAgent, type ClientAgent } from './client-classifier.js';

export interface ClientContext {
  ip: string | null;
  userAgent: string | null;
  agent: ClientAgent | null;
  // Structural shape of the inbound messages (count, role sequence,
  // tool_calls / thinking presence) — never their content. Set by the chat
  // entry points after request parsing; read by logRequest() so the analytics
  // drill-down can answer "was this multi-turn / did it carry tool_calls?"
  // (#750) without storing dialog text.
  shape: RequestShape | null;
}

// The minimal per-request structure needed to reproduce a provider 4xx like
// "reasoning_content must be passed back" outside the #255 replay path.
export interface RequestShape {
  messageCount: number;
  roleSequence: string;
  hasToolCalls: boolean;
  hasReasoning: boolean;
}

// Request-scoped caller identity, readable from anywhere below the middleware
// without threading parameters through every logRequest() call site (the chat
// proxy, responses, anthropic, fusion, embeddings and media paths all log).
const storage = new AsyncLocalStorage<ClientContext>();

// Resolve the client IP from the socket peer address. The X-Forwarded-For
// header is only trusted when Express's "trust proxy" setting is enabled
// (opt-in via app.set('trust proxy', ...) or the TRUST_PROXY env var in
// run.ts). Without that, a spoofed header from a LAN client is ignored.
function resolveClientIp(req: Request): string | null {
  const trustProxy = req.app?.get('trust proxy') ?? false;
  let raw: string | null;
  if (trustProxy) {
    const xff = req.headers['x-forwarded-for'];
    raw = (Array.isArray(xff) ? xff[0] : xff)?.split(',')[0]?.trim() || req.socket.remoteAddress || null;
  } else {
    raw = req.socket.remoteAddress || null;
  }
  // Normalize IPv4-mapped IPv6 ("::ffff:192.168.0.5" -> "192.168.0.5").
  return raw?.replace(/^::ffff:/i, '') ?? null;
}

// Privacy opt-out: REQUEST_ANALYTICS_LOG_CLIENT=false stores nulls instead of
// the caller's IP/UA. Read per request (not at module load) so tests and
// embedders can toggle it without re-importing.
function clientLoggingEnabled(): boolean {
  return process.env.REQUEST_ANALYTICS_LOG_CLIENT !== 'false';
}

export function clientContextMiddleware(req: Request, _res: Response, next: NextFunction): void {
  if (!clientLoggingEnabled()) {
    storage.run({ ip: null, userAgent: null, agent: null, shape: null }, next);
    return;
  }
  const ua = req.headers['user-agent'];
  storage.run({
    ip: resolveClientIp(req),
    userAgent: typeof ua === 'string' ? ua.slice(0, 256) : null,
    agent: classifyClientAgent(req),
    shape: null,
  }, next);
}

export function getClientContext(): ClientContext {
  return storage.getStore() ?? { ip: null, userAgent: null, agent: null, shape: null };
}

// Record the inbound messages' shape for the current request. Called by each
// chat entry point right after request parsing; logRequest() picks it up at
// insert time. No-op outside a request context.
export function setRequestShape(shape: RequestShape | null): void {
  const ctx = storage.getStore();
  if (ctx) ctx.shape = shape;
}

export function getRequestShape(): RequestShape | null {
  return storage.getStore()?.shape ?? null;
}

// Cap for the compressed role sequence stored per row: enough segments to
// tell multi-turn from single-turn without growing unboundedly (20 segments
// × ~10 chars ≈ 200 chars max).
const MAX_SEQUENCE_LENGTH = 20;

// Loose structural probe over any inbound protocol's messages array (OpenAI
// chat.completions, Responses `input` items, Anthropic messages, Ollama) —
// it only ever reads `role` / `type` / `tool_calls` / `reasoning_content`
// and content-block type tags, never message text.
export function summarizeRequestMessages(messages: unknown): RequestShape | null {
  if (!Array.isArray(messages) || messages.length === 0) return null;

  let hasToolCalls = false;
  let hasReasoning = false;
  const roles: string[] = [];

  for (const raw of messages) {
    if (!raw || typeof raw !== 'object') continue;
    const m = raw as Record<string, unknown>;

    const role = typeof m.role === 'string'
      ? m.role
      : (typeof m.type === 'string' ? m.type : 'unknown');
    roles.push(role);

    if (!hasToolCalls) {
      hasToolCalls =
        (Array.isArray(m.tool_calls) && m.tool_calls.length > 0)
        || m.type === 'function_call'
        || (Array.isArray(m.content) && m.content.some(
          b => !!b && typeof b === 'object' && (b as { type?: string }).type === 'tool_use'));
    }
    if (!hasReasoning) {
      hasReasoning =
        (typeof m.reasoning_content === 'string' && m.reasoning_content.length > 0)
        || (Array.isArray(m.content) && m.content.some(
          b => !!b && typeof b === 'object' && ((b as { type?: string }).type === 'thinking' || (b as { type?: string }).type === 'reasoning')));
    }
  }

  // Compress consecutive repeats ("user,user,assistant" → "user×2,assistant")
  // so long single-role prefixes (system prompts) stay one segment.
  const segments: Array<{ role: string; count: number }> = [];
  for (const role of roles) {
    const last = segments[segments.length - 1];
    if (last && last.role === role) last.count += 1;
    else segments.push({ role, count: 1 });
  }
  const roleSequence = segments
    .slice(0, MAX_SEQUENCE_LENGTH)
    .map(s => s.count > 1 ? `${s.role}×${s.count}` : s.role)
    .join(',')
    + (segments.length > MAX_SEQUENCE_LENGTH ? ',…' : '');

  return {
    messageCount: messages.length,
    roleSequence,
    hasToolCalls,
    hasReasoning,
  };
}
