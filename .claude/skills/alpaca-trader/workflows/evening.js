export const meta = {
  name: 'alpaca-evening',
  description: 'Turn scored theses into distilled memory-update suggestions (no file writes).',
  phases: [{ title: 'Reflect', detail: 'one reflection per scored thesis' }],
}

const a = args || {}
const theses = a.openTheses || []

const UPDATE_SCHEMA = {
  type: 'object',
  required: ['updates'],
  properties: {
    updates: {
      type: 'array',
      items: {
        type: 'object',
        required: ['scope', 'name', 'heading', 'body'],
        properties: {
          scope: { type: 'string', enum: ['ticker', 'sector', 'pattern', 'lesson'] },
          name: { type: 'string' },
          heading: { type: 'string' },
          body: { type: 'string' },
        },
      },
    },
  },
}

log(`Reflecting on ${theses.length} scored theses`)

const perThesis = await parallel(
  theses.map((t) => () =>
    agent(
      `Reflect on this closed/marked trade and produce concise memory updates.
Trade: ${JSON.stringify(t)}.
Was the thesis right? Did ophir's prediction calibrate to the realized return?
Return 1-3 updates: a 'ticker' note for ${t.symbol}, optionally a 'sector' note,
and (only if a generalizable rule emerged) a 'pattern' or 'lesson' note. Keep
each body to a few sentences; write durable knowledge, not a play-by-play.`,
      { label: `reflect:${t.symbol}`, phase: 'Reflect', schema: UPDATE_SCHEMA },
    ),
  ),
)

const updates = perThesis
  .filter(Boolean)
  .flatMap((r) => (r.updates || []))
log(`Produced ${updates.length} memory updates`)
return { updates }
