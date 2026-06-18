"""Clean-room trading-agent layers built on top of the ophir model.

Subpackages here reuse ophir's model and feature contract (e.g.
:mod:`ophir.ticker`) but keep trading/agent logic separate from the model
code. The first layer is market-data ingestion (:mod:`ophir.agent.ingest`
and :mod:`ophir.agent.feed`).
"""
