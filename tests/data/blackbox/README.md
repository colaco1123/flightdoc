# Real blackbox logs

These logs come from the test data of [fc-blackbox](https://github.com/ilya-epifanov/fc-blackbox)
by Ilya Epifanov (commit `de08350`), licensed MIT OR Apache-2.0. The MIT license text is in
`LICENSE-fc-blackbox`. They are gzip-compressed here; `inav-LOG00004-first3MB.TXT.gz` is the
first 3 MiB of the original 16 MiB flash dump.

| File | Firmware | Notes |
|---|---|---|
| `btfl_001.bbl` | Betaflight 4.2.8 | two sessions |
| `LOG00037.BFL` | Betaflight 4.2.0 | short flight with GPS |
| `crashing-LOG00002.BFL` | EmuFlight 3.7.0 | craft that would not take off level |
| `inav-LOG00004-first3MB.TXT` | INAV 3.0.1 | fixed wing |

`expected.json` holds row counts and column sums for each session, recorded after checking the
decoder against `blackbox_decode` value for value.
