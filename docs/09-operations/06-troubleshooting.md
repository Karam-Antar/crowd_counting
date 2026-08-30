# Troubleshooting

This project often fails because of hidden dataset assumptions rather than a straightforward code defect.

## Typical investigation path

- verify dataset pairing,
- check label scaling and preprocessing,
- inspect the active training script and config,
- compare with the most relevant successful experiment,
- confirm whether a folder or split assumption changed.

Many project issues are easier to diagnose by checking the data contract than by debugging the model itself.
