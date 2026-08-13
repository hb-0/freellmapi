// Migration: request body shape (requests.message_count / role_sequence /
// has_tool_calls / has_reasoning)
// Created: 2026-08-13
//
// DOWN: reversible
//
// When a request fails with a provider 4xx ("reasoning_content must be passed
// back", #750), the error text alone cannot tell whether the missed scenario
// is multi-turn or tool_calls-heavy. These columns record the SHAPE of the
// inbound messages (count, compressed role sequence, tool_calls / thinking
// presence) — never their content — so the analytics drill-down can answer
// "was this multi-turn / did it carry tool_calls?" without storing dialog
// text.
//
// Guarded like the baseline's column adds: catalog-sync re-runs migrations
// over a live schema, so ALTERs must be idempotent.

import type { Db } from '../types.js';

function hasColumn(db: Db, table: string, column: string): boolean {
  const columns = db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[];
  return columns.some(col => col.name === column);
}

export function up(db: Db): void {
  if (!hasColumn(db, 'requests', 'message_count')) {
    db.prepare('ALTER TABLE requests ADD COLUMN message_count INTEGER').run();
  }
  if (!hasColumn(db, 'requests', 'role_sequence')) {
    db.prepare('ALTER TABLE requests ADD COLUMN role_sequence TEXT').run();
  }
  if (!hasColumn(db, 'requests', 'has_tool_calls')) {
    db.prepare('ALTER TABLE requests ADD COLUMN has_tool_calls INTEGER').run();
  }
  if (!hasColumn(db, 'requests', 'has_reasoning')) {
    db.prepare('ALTER TABLE requests ADD COLUMN has_reasoning INTEGER').run();
  }
}

export function down(db: Db): void {
  if (hasColumn(db, 'requests', 'has_reasoning')) {
    db.prepare('ALTER TABLE requests DROP COLUMN has_reasoning').run();
  }
  if (hasColumn(db, 'requests', 'has_tool_calls')) {
    db.prepare('ALTER TABLE requests DROP COLUMN has_tool_calls').run();
  }
  if (hasColumn(db, 'requests', 'role_sequence')) {
    db.prepare('ALTER TABLE requests DROP COLUMN role_sequence').run();
  }
  if (hasColumn(db, 'requests', 'message_count')) {
    db.prepare('ALTER TABLE requests DROP COLUMN message_count').run();
  }
}
