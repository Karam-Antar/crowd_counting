# Split Strategy

The project is designed to support deterministic train / validation splitting when a valid folder is not present. The use of a fixed seed and explicit split logic is part of the project’s reproducibility conventions.

When a valid folder exists, it is preferred. This is important because manually curated validation data often represents a more reliable benchmark than a random split.
