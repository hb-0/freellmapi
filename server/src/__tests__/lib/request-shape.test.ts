import { describe, expect, it } from 'vitest';
import { summarizeRequestMessages } from '../../lib/client-context.js';

// summarizeRequestMessages is a loose structural probe over any inbound
// protocol's messages array: it must survive OpenAI chat.completions, the
// Responses `input` items, Anthropic messages, and Ollama shapes, and it must
// NEVER retain message content (only counts, roles, and flags).

describe('summarizeRequestMessages', () => {
  it('returns null for empty or non-array input', () => {
    expect(summarizeRequestMessages([])).toBeNull();
    expect(summarizeRequestMessages(undefined)).toBeNull();
    expect(summarizeRequestMessages('a string')).toBeNull();
    expect(summarizeRequestMessages({})).toBeNull();
  });

  it('summarizes a plain OpenAI-style single turn', () => {
    const shape = summarizeRequestMessages([
      { role: 'system', content: 'you are a helper' },
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'hi' },
    ]);
    expect(shape).toEqual({
      messageCount: 3,
      roleSequence: 'system,user,assistant',
      hasToolCalls: false,
      hasReasoning: false,
    });
  });

  it('flags tool_calls on OpenAI assistant messages', () => {
    const shape = summarizeRequestMessages([
      { role: 'user', content: 'list files' },
      { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'ls', arguments: '{}' } }] },
      { role: 'tool', tool_call_id: 'c1', content: 'a.txt' },
      { role: 'user', content: 'again' },
    ]);
    expect(shape).toMatchObject({
      messageCount: 4,
      hasToolCalls: true,
      hasReasoning: false,
    });
    expect(shape!.roleSequence).toBe('user,assistant,tool,user');
  });

  it('flags reasoning_content (thinking mode) and compresses consecutive roles', () => {
    const shape = summarizeRequestMessages([
      { role: 'system', content: 'a' },
      { role: 'system', content: 'b' },
      { role: 'user', content: 'q' },
      { role: 'assistant', content: 'a', reasoning_content: 'thinking hard' },
    ]);
    expect(shape).toMatchObject({
      messageCount: 4,
      hasReasoning: true,
      hasToolCalls: false,
    });
    // Consecutive system roles collapse into one segment.
    expect(shape!.roleSequence).toBe('system×2,user,assistant');
  });

  it('handles Responses API input items (message / function_call / output / reasoning)', () => {
    const shape = summarizeRequestMessages([
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'hi' }] },
      { type: 'function_call', call_id: 'c1', name: 'search', arguments: '{}' },
      { type: 'function_call_output', call_id: 'c1', output: 'results' },
      { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'next' }] },
    ]);
    expect(shape).toMatchObject({
      messageCount: 4,
      hasToolCalls: true,
      hasReasoning: false,
    });
    expect(shape!.roleSequence).toBe('user,function_call,function_call_output,user');
  });

  it('flags Anthropic tool_use and thinking blocks inside content arrays', () => {
    const shape = summarizeRequestMessages([
      { role: 'user', content: 'plan' },
      { role: 'assistant', content: [{ type: 'thinking', thinking: '…' }, { type: 'text', text: 'here' }] },
      { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'plan', input: {} }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'done' }] },
    ]);
    expect(shape).toMatchObject({
      messageCount: 4,
      hasToolCalls: true,
      hasReasoning: true,
    });
    // Two consecutive assistant turns (text then tool_use) collapse.
    expect(shape!.roleSequence).toBe('user,assistant×2,user');
  });

  it('caps the role sequence and never stores message content', () => {
    const messages = Array.from({ length: 40 }, (_, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `secret dialogue ${i}`,
    }));
    const shape = summarizeRequestMessages(messages);
    expect(shape!.messageCount).toBe(40);
    expect(shape!.roleSequence.endsWith(',…')).toBe(true);
    expect(shape!.roleSequence.length).toBeLessThan(200);
    expect(JSON.stringify(shape)).not.toContain('secret dialogue');
  });
});
