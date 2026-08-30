# Constraints and Tradeoffs

This project has several practical constraints:

- dataset layout assumptions are strict,
- label scaling and preprocessing are part of the model contract,
- older experiment runs may use different logic and are not always directly comparable,
- the codebase has been shaped by empirical experiments rather than a single permanent design.

These tradeoffs are important to document, because long-lived ML projects often accumulate experimental drift that is difficult to interpret without written context.
