export const meta = {
  name: 'alpaca-morning',
  description: 'Screen candidates, analyze each, adversarially verify, return trade proposals (no placement).',
  phases: [
    { title: 'Analyze', detail: 'one analyst per shortlisted candidate' },
    { title: 'Verify', detail: 'devil\'s-advocate vote per proposed trade' },
  ],
}

const a = args || {}
const shortlistSize = a.shortlistSize || 15
const verifyVotes = a.verifyVotes || 1
const seeds = a.seedCandidates || { core: [], tactical: [] }
const forecasts = a.ophirForecasts || {}

const PROPOSAL_SCHEMA = {
  type: 'object',
  required: ['recommend', 'symbol', 'sleeve'],
  properties: {
    recommend: { type: 'boolean' },
    symbol: { type: 'string' },
    sleeve: { type: 'string', enum: ['core', 'tactical'] },
    side: { type: 'string', enum: ['buy', 'sell'] },
    asset_class: { type: 'string', enum: ['equity', 'option'] },
    notional: { type: 'number' },
    sector: { type: ['string', 'null'] },
    is_defined_risk: { type: 'boolean' },
    is_short_option: { type: 'boolean' },
    thesis: { type: 'string' },
    signals: { type: 'object' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['survives'],
  properties: { survives: { type: 'boolean' }, reason: { type: 'string' } },
}

// Build the shortlist from the seed candidates the main agent gathered.
const candidates = [
  ...seeds.core.map((c) => ({ ...c, sleeve: 'core' })),
  ...seeds.tactical.map((c) => ({ ...c, sleeve: 'tactical' })),
].slice(0, shortlistSize)

log(`Analyzing ${candidates.length} candidates (verifyVotes=${verifyVotes})`)

const analyzed = await pipeline(
  candidates,
  (c) =>
    agent(
      `You are a trading analyst. Symbol ${c.symbol} (sleeve: ${c.sleeve}).
Use Alpaca MCP read tools (get_stock_snapshot, get_stock_bars, get_news, and for
options get_option_chain/get_option_snapshot) to assess a ${c.sleeve} trade.
ophir forecast for this symbol (may be absent): ${JSON.stringify(forecasts[c.symbol] || null)}.
Blend ophir + momentum + sentiment per the sleeve weighting (core is ophir-led,
tactical is technicals-led). Propose at most ONE order. Set recommend=false if no
edge. notional is the dollar size you want (premium-at-risk for options). Do NOT
place any order. Return the proposal object.`,
      { label: `analyze:${c.symbol}`, phase: 'Analyze', schema: PROPOSAL_SCHEMA },
    ),
  (proposal, _c, _i) => {
    if (!proposal || !proposal.recommend) return null
    return parallel(
      Array.from({ length: verifyVotes }, (_v) => () =>
        agent(
          `Adversarially review this proposed ${proposal.sleeve} trade and try to
REFUTE it. Default survives=false if the thesis is weak, the signal is thin, or
risk is unclear. Proposal: ${JSON.stringify(proposal)}.`,
          { label: `verify:${proposal.symbol}`, phase: 'Verify', schema: VERDICT_SCHEMA },
        ),
      ),
    ).then((votes) => {
      const ok = votes.filter(Boolean).filter((v) => v.survives).length
      const need = Math.ceil(verifyVotes / 2)
      return ok >= need ? proposal : null
    })
  },
)

const proposals = analyzed.flat ? analyzed.flat().filter(Boolean) : analyzed.filter(Boolean)
log(`Surviving proposals: ${proposals.length}`)
return { proposals }
