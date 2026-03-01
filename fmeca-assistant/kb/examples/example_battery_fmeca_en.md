## Example: Backup Battery System for Telecom Base Station

Function:
Provide DC power to the telecom equipment during mains failure until the generator starts and stabilizes.

Failure Modes:
- Battery string open circuit (no continuity).
- Loss of effective capacity below the design autonomy.
- DC bus overvoltage due to charger malfunction.
- Unbalanced strings causing premature ageing of specific batteries.
- Thermal runaway in one or several cells.

Causes:
- Loose connections or broken inter-cell links.
- Ageing beyond design life, high ambient temperature, frequent deep discharges.
- Wrong charger voltage settings, failed voltage regulation, faulty sensors.
- Mixing old and new batteries or different chemistries in one string.
- Insufficient ventilation, blocked air paths, missing temperature monitoring.

Local Effects:
- Battery room alarms for high temperature or abnormal voltage.
- Visible swelling, leakage, or discoloration on some units.
- Frequent operation of DC protection devices, repeated alarms for low autonomy.

System Effects:
- Shortened backup time during mains loss.
- Inability to ride through generator start and synchronization delays.
- Repeated deep discharges further accelerating degradation.
- Increased maintenance trips and unplanned replacements.

End Effects:
- Loss of service for the base station during power events.
- Possible loss of coverage for emergency communications.
- Higher operational costs and reduced asset life.

Current Controls:
- Periodic capacity testing and impedance measurements.
- Remote monitoring of voltage, current, and temperature.
- Visual inspections during scheduled maintenance visits.

Assessment (illustrative):
- Severity S = 8 (service disruption is serious but not immediately life-threatening).
- Occurrence O = 5 (if batteries are not proactively replaced).
- Detection D = 6 (some issues only visible during discharge tests).
- RPN = 8 × 5 × 6 = 240.

Recommended Actions:
- Implement automated trend analysis on capacity and internal resistance.
- Define replacement policy based on measured degradation, not only calendar age.
- Improve training of field technicians on correct torque and cabling practices.
- Install thermal sensors close to battery strings and add dedicated alarms.