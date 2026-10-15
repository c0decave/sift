# Project Phoenix — Kickoff Notes

## Goals

Rebuild the reporting pipeline to cut nightly batch time from 6h to under 1h.

## Scope

- Replace the legacy ETL with an incremental loader
- Migrate report storage to columnar format
- Keep the existing dashboard API contract unchanged

## Risks

1. Historical data quality is uneven before 2022.
2. The nightly window overlaps with the EU backup job.

## Next steps

- [ ] Benchmark the columnar writer
- [ ] Draft the incremental loader design doc
- [ ] Schedule a data-quality audit

See the [design doc](https://example.com/phoenix) for details.
