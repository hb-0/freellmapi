import { beforeAll, beforeEach, describe, expect, it } from 'vitest';
import type { Express } from 'express';
import { createApp } from '../../app.js';
import { getDb, initDb } from '../../db/index.js';
import { mintDashboardToken } from '../helpers/auth.js';
import { clientContextMiddleware } from '../../lib/client-context.js';
import { setRequestShape, summarizeRequestMessages, setRequestBody } from '../../lib/client-context.js';
import { logRequest } from '../../lib/request-log.js';

// Read surface for per-attempt routing traces: the requests list carries a
// cheap attemptCount per row, and GET /api/analytics/requests/:id returns the
// row plus its ordinal-ordered attempts ladder (machine-readable).

let dashToken = '';

async function request(app: Express, path: string) {
  const server = app.listen(0);
  const addr = server.address() as any;
  const url = `http://127.0.0.1:${addr.port}${path}`;

  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${dashToken}` },
  });
  const data = await res.json().catch(() => null);
  server.close();

  return { status: res.status, body: data };
}

function insertRequest(status: string, createdAt: string): number {
  const insert = getDb().prepare(`
    INSERT INTO requests (platform, model_id, status, input_tokens, output_tokens, latency_ms, error, created_at)
    VALUES ('cerebras', 'qwen-z', ?, 10, 5, 1234, NULL, ?)
  `).run(status, createdAt);
  return Number(insert.lastInsertRowid);
}

function insertAttempt(requestId: number, ordinal: number, platform: string, modelId: string, keyOrdinal: number, outcome: string, startOffsetMs: number, durationMs: number, errorSummary: string | null = null) {
  getDb().prepare(`
    INSERT INTO request_attempts (request_id, ordinal, platform, model_id, key_ordinal, outcome, start_offset_ms, duration_ms, error_summary)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `).run(requestId, ordinal, platform, modelId, keyOrdinal, outcome, startOffsetMs, durationMs, errorSummary);
}

function recentSql(hour: number): string {
  const day = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  return `${day} ${String(hour).padStart(2, '0')}:00:00`;
}

describe('per-attempt traces in the analytics requests API', () => {
  let app: Express;

  beforeAll(() => {
    process.env.ENCRYPTION_KEY = '0'.repeat(64);
    initDb(':memory:');
    app = createApp();
    dashToken = mintDashboardToken();
  });

  beforeEach(() => {
    getDb().prepare('DELETE FROM request_attempts').run();
    getDb().prepare('DELETE FROM requests').run();
  });

  it('includes attemptCount per row in the list payload', async () => {
    const laddered = insertRequest('success', recentSql(10));
    insertAttempt(laddered, 0, 'groq', 'llama-x', 1, 'rate_limited', 0, 210);
    insertAttempt(laddered, 1, 'google', 'gemini-y', 2, 'timeout', 215, 10000);
    insertAttempt(laddered, 2, 'cerebras', 'qwen-z', 3, 'ok', 10220, 900);
    insertRequest('error', recentSql(11)); // mid-ladder row: no children

    const { status, body } = await request(app, '/api/analytics/requests?range=7d');
    expect(status).toBe(200);
    // Newest first: the childless error row, then the laddered success row.
    expect(body.rows.map((r: any) => r.attemptCount)).toEqual([0, 3]);
  });

  it('returns the ordered attempts ladder — with per-hop error summaries — in the per-request detail payload', async () => {
    const id = insertRequest('success', recentSql(12));
    insertAttempt(id, 0, 'groq', 'llama-x', 1, 'rate_limited', 0, 210, '429 rate limit reached, retry in 7m12s');
    insertAttempt(id, 1, 'google', 'gemini-y', 2, 'timeout', 215, 10000, 'upstream timeout after 10000ms');
    insertAttempt(id, 2, 'cerebras', 'qwen-z', 3, 'ok', 10220, 900);

    const { status, body } = await request(app, `/api/analytics/requests/${id}`);
    expect(status).toBe(200);
    expect(body).toMatchObject({
      id,
      platform: 'cerebras',
      modelId: 'qwen-z',
      status: 'success',
      latencyMs: 1234,
    });
    expect(body.attempts).toEqual([
      { ordinal: 0, platform: 'groq', modelId: 'llama-x', keyOrdinal: 1, outcome: 'rate_limited', startOffsetMs: 0, durationMs: 210, errorSummary: '429 rate limit reached, retry in 7m12s' },
      { ordinal: 1, platform: 'google', modelId: 'gemini-y', keyOrdinal: 2, outcome: 'timeout', startOffsetMs: 215, durationMs: 10000, errorSummary: 'upstream timeout after 10000ms' },
      { ordinal: 2, platform: 'cerebras', modelId: 'qwen-z', keyOrdinal: 3, outcome: 'ok', startOffsetMs: 10220, durationMs: 900, errorSummary: null },
    ]);
  });

  it('returns an empty attempts array for a row without children, and 404/400 for missing/invalid ids', async () => {
    const id = insertRequest('error', recentSql(13));

    const detail = await request(app, `/api/analytics/requests/${id}`);
    expect(detail.status).toBe(200);
    expect(detail.body.attempts).toEqual([]);

    expect((await request(app, '/api/analytics/requests/999999')).status).toBe(404);
    expect((await request(app, '/api/analytics/requests/not-a-number')).status).toBe(400);
  });

  it('returns the inbound message shape and nulls for pre-migration rows', async () => {
    const shaped = getDb().prepare(`
      INSERT INTO requests (platform, model_id, status, input_tokens, output_tokens, latency_ms, error, message_count, role_sequence, has_tool_calls, has_reasoning, request_body, created_at)
      VALUES ('cerebras', 'qwen-z', 'error', 10, 0, 50, 'boom', 5, 'user,assistant,tool,user', 1, 1, '{"messages":[{"role":"user","content":"hi"}]}', ?)
    `).run(recentSql(14));
    const legacy = insertRequest('error', recentSql(15));

    const shapedDetail = await request(app, `/api/analytics/requests/${shaped.lastInsertRowid}`);
    expect(shapedDetail.status).toBe(200);
    expect(shapedDetail.body).toMatchObject({
      messageCount: 5,
      roleSequence: 'user,assistant,tool,user',
      hasToolCalls: true,
      hasReasoning: true,
      requestBody: '{"messages":[{"role":"user","content":"hi"}]}',
    });

    const legacyDetail = await request(app, `/api/analytics/requests/${legacy}`);
    expect(legacyDetail.status).toBe(200);
    expect(legacyDetail.body).toMatchObject({
      messageCount: null,
      roleSequence: null,
      hasToolCalls: null,
      hasReasoning: null,
      requestBody: null,
    });
  });

  it('threads the inbound shape from the chat entry point into the logged row', async () => {
    // Simulate what proxy/responses/anthropic do after request parsing: set
    // the shape inside the middleware's request scope, then log.
    const messages = [
      { role: 'user', content: 'use the tool' },
      { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'ls', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: 'c1', content: 'a.txt' },
      { role: 'user', content: 'again' },
    ];
    const fakeReq = { headers: {}, socket: { remoteAddress: '127.0.0.1' }, app: { get: () => false } } as any;
    let loggedId = 0;
    clientContextMiddleware(fakeReq, {} as any, () => {
      setRequestShape(summarizeRequestMessages(messages));
      setRequestBody({ model: 'test-model', messages });
      logRequest('test', 'test-model', 1, 'success', 1, 2, 3, null);
      loggedId = (getDb().prepare('SELECT id FROM requests ORDER BY id DESC LIMIT 1').get() as { id: number }).id;
    });
    expect(loggedId).toBeGreaterThan(0);

    const { status, body } = await request(app, `/api/analytics/requests/${loggedId}`);
    expect(status).toBe(200);
    expect(body).toMatchObject({
      messageCount: 4,
      roleSequence: 'user,assistant,tool,user',
      hasToolCalls: true,
      hasReasoning: false,
    });
    const parsedBody = JSON.parse(body.requestBody);
    expect(parsedBody.model).toBe('test-model');
    expect(parsedBody.messages).toHaveLength(4);
    expect(parsedBody.messages[0]).toEqual({ role: 'user', content: 'use the tool' });
  });
});
