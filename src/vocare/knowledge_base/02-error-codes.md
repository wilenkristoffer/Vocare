# AutoDose Error Codes

## E01 - Canister not detected

The unit can't read the RFID tag on a loaded canister. Usually a seating issue, occasionally a
damaged tag.

- Remove the canister and re-seat it fully until it clicks.
- If the error persists on the same slot with a different canister, the slot's RFID reader may
  be faulty - this needs a technician visit, do not attempt to bypass the reader.

## E02 - Dispense count mismatch

The weight sensor's count after dispensing doesn't match the expected count for the order.

- The unit automatically halts and holds the pouch for manual review - this is a safety
  behavior, not a fault to "fix" by retrying blindly.
- A pharmacist must visually recount and either confirm or reject the pouch before the unit
  will resume.

## E03 - Label printer out of stock / jammed

- Check the label roll and printer door are seated correctly.
- Out-of-stock and jam both surface as E03; the on-screen detail line distinguishes them.

## E04 - Network connection lost

The unit lost contact with the pharmacy order system. Queued orders are held locally and will
resume automatically once connectivity returns. See the network connectivity document for
troubleshooting steps.

## E05 - Bulk hopper low (AutoDose 500 XL only)

Informational, not a fault - the hopper for one of the 20 bulk medications is below the refill
threshold. Refill within the next business day to avoid a dispense delay.

## E06 - Calibration drift detected

The unit's self-check found the dispensing head is outside tolerance. See the calibration
document. The unit will not dispense until recalibrated or a technician confirms it's safe to
continue.

## E07 - Software update failed to apply

See the software updates document. The unit keeps running its previous version until the
update succeeds.

## E08 - Unrecognized medication code

The order references a medication code not in the unit's loaded canister map. This usually means
a new medication needs a canister assigned in the pharmacy system's mapping table - it is not
something the AutoDose unit itself can resolve, escalate to whoever manages canister mappings.
