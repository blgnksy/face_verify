## v0.3.0 (2026-06-24)

### Feat

- add bump workflow and update CI to ignore specific paths
- clamp combined score in verify function for improved stability
- automize release
- mvp face verification for linux authentication

### Fix

- update checkout ref to use 'main' branch in bump workflow
- update bump command to skip changelog generation and ensure it runs without errors
- disable changelog update on version bump and adjust bump command
- update git configuration in bump workflow for improved safety and clarity
- remove commitizen installation step from bump workflow
- add GitHub Actions workflow for automated version bump and release
