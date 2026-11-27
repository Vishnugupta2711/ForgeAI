# Synthesis Prompt v2 — Ship-Spares Procurement Demo Corpus (`demos/ship_spares`)

Generate an expanded ship-spares procurement corpus (`spares_catalogue.csv` with 160 rows, plus `supplier_terms_and_returns.pdf` and `logistics_and_lead_times.pdf` at roughly 5,000–7,000 characters each across 8 detailed sections).

Requirements:
1. Keep `spares_catalogue.csv` (160 rows) exact: columns `part_no`, `description`, `unit_price_usd`, `min_order_qty`, `lead_time_days`, `supplier`, `stock_location`, `hazmat_flag` across 4 suppliers (`NordicMarine Supply`, `BalticValve GmbH`, `PacificPump Co`, `AegisElectro Ltd`) and 4 ports (`Rotterdam-WH1`, `Singapore-WH2`, `Hamburg-WH3`, `Busan-WH4`).
2. `supplier_terms_and_returns.pdf` (`DOC SUP-01`, 8 sections): approved supplier authority, MOQ and multi-line PO consolidation rules, volume discounts (5% >= 10 units, 10% >= 25 units on single SKU, non-stacking), USD 125 small-order handling fee (< USD 500 discounted subtotal), USD 85 hazmat documentation fee, Net 30 / 2% early-payment discount (>= USD 5,000 within 10 days), cancellation fees, 30-day/60-day return restocking fees (15% / 25% non-hazmat only) vs strict non-returnable rule for `hazmat_flag=true`, and 14-day zero-fee defective replacement warranty.
3. `logistics_and_lead_times.pdf` (`DOC SUP-02`, 8 sections): port hub mapping, `effective_lead_days = lead_time_days + (4 if hazmat) + (3 if transfer)`, Thursday 16:00 UTC consolidation cutoff + EU Rotterdam/Hamburg 1-day shuttle exception + peak season buffers, expedited air freight (`ceil(lead_time_days / 2)` for non-hazmat <150 kg at USD 450 base + USD 6.50/kg >30 kg), IMDG hazmat air prohibition, split shipments, DAP/barge delivery rules, and cold-chain/receiving inspection.
4. No external weather forecasts or unrelated API specs — expected tools remain `csv_lookup` and `calculator` only.
5. Keep PDF generation 100% reproducible from `build_assets.py` with automatic page wrapping.
