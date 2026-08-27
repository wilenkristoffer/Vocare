# Network Connectivity

## E04 - connection lost

The unit needs a connection to the pharmacy order system to receive new orders, but it keeps
working offline for orders already queued.

## First checks

1. Check the unit's network status icon (top-right of the touchscreen) - it shows whether the
   issue is Wi-Fi/Ethernet link, or reaching the pharmacy server specifically.
2. If the link itself is down: check the Ethernet cable / Wi-Fi status like any other network
   device on site.
3. If the link is up but the server is unreachable: check whether other workstations in the
   pharmacy can reach the pharmacy system - if none can, it's a site-wide issue, not the
   AutoDose unit.

## While disconnected

- Orders already queued before the disconnect will continue to dispense normally.
- New orders won't arrive until connectivity is restored; the pharmacy system will typically
  queue them on its side and deliver them once the unit reconnects.
- The unit does not need to be restarted when connectivity returns - it reconnects
  automatically.

## Recurring disconnects

If E04 happens repeatedly (more than once a week) rather than as an isolated event, that's
worth escalating to IT/networking rather than treating each occurrence as independent - it
usually points to a Wi-Fi coverage or network configuration issue near the unit's location.
