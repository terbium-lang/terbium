# Terbium

Rewrite branch of Terbium. See legacy Terbium [here](https://github.com/TerbiumLang/Terbium/tree/legacy)

See the new specification: https://jay3332.gitbook.io/terbium/spec

## Known issues

- Parsing
  - cannot parse `f().attr`
    - potential fix: https://discord.com/channels/273534239310479360/1087679298309214261/1103015503284412456
  - implicit returns with braced-expressions only register as normal expressions
