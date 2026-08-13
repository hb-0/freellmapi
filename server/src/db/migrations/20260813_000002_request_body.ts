// Migration: full request body capture (requests.request_body)
// Created: 2026-08-13
//
// DOWN: reversible
//
// Debug-only companion to the request-shape migration (#750): stores the FULL
// inbound request body (messages, tools, reasoning_content — everything the
// shape columns summarize) as JSON text, so a provider 4xx can be reproduced
// from the analytics drill-down. Intended for single-user self-hosting where
// there is no privacy concern; content redaction is limited to inline image
// data-URIs and an overall size cap applied at write time (client-context).
//
// Guarded like the baseline's column adds: catalog-sync re-runs migrations
// over a live schema, so ALTERs must be idempotent.

import type { Db } from '../types.js';

function hasColumn(db: Db, table: string, column: string): boolean {
  const columns = db.prepare(`PRAGMA table_info(${table})`).all() as { name: string }[];
  return columns.some(col => col.name === column);
}

export function up(db: Db): void {
  if (!hasColumn(db, 'requests', 'request_body')) {
    db.prepare('ALTER TABLE requests ADD COLUMN request_body TEXT').run();
  }
}

export function down(db: Db): void {
  if (hasColumn(db, 'requests', 'request_body')) {
    db.prepare('ALTER TABLE requests DROP COLUMN request_body').run();
  }
}
