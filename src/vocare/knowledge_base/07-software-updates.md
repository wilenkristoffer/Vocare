# Software Updates

## How updates arrive

The unit checks for updates automatically once a day when connected to the pharmacy network,
and shows a prompt on the touchscreen when one's available. Updates never install
automatically mid-shift - they wait for an explicit confirmation and, if orders are queued,
wait until the queue is empty.

## Applying an update

1. Confirm no orders are queued or in progress.
2. Tap "Install now" on the update prompt (or Settings > Software > Check for updates).
3. The unit restarts, applies the update (typically 5-10 minutes), and runs a self-check.
4. If the self-check passes, the unit returns to normal operation automatically.

## If an update fails (E07)

- The unit automatically rolls back to the previous working version - it will not get stuck on
  a broken update.
- Retry once from Settings > Software > Check for updates.
- If it fails a second time, stop retrying and escalate - repeated update failures on the same
  build usually mean something environment-specific (network instability, storage) rather than
  a one-off glitch.

## Version support

Each major version is supported for security updates for 3 years after release. The unit will
warn on-screen 90 days before its current version reaches end of support.
