# Synthesis Prompt — Ship-Spares Procurement Demo Corpus (`demos/ship_spares`)

Generate a realistic ship-spares procurement corpus (`spares_catalogue.csv` with 160 rows, `supplier_terms_and_returns.pdf`, `logistics_and_lead_times.pdf`).

Requirements:
1. `spares_catalogue.csv` (160 rows): columns `part_no`, `description`, `unit_price_usd`, `min_order_qty`, `lead_time_days`, `supplier`, `stock_location`, `hazmat_flag` across 4 suppliers (`NordicMarine Supply`, `BalticValve GmbH`, `PacificPump Co`, `AegisElectro Ltd`) and 4 ports (`Rotterdam-WH1`, `Singapore-WH2`, `Hamburg-WH3`, `Busan-WH4`).
2. `supplier_terms_and_returns.pdf`: payment terms (Net 30, 2% early pay discount >= $5,000 within 10 days), line volume discounts (5% qty >= 10, 10% qty >= 25), $125 small-order handling fee under $500, 15% restocking fee within 30 days, and strict rule that hazmat items are non-returnable.
3. `logistics_and_lead_times.pdf`: effective lead-time formula = catalogue `lead_time_days` + 4 days customs/hazmat clearance if `hazmat_flag == true` + 3 days inter-port transfer if stock location differs from destination port; air-freight expedite reduces base transit by 50% (rounded up) for $450 flat surcharge on non-hazmat orders only.
4. No weather or live external forecast requests — expected tools: `csv_lookup` and `calculator` only.
