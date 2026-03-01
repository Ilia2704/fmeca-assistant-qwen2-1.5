## Example: Passenger Elevator in Office Building

Function:
Transport passengers safely and reliably between floors within acceptable waiting and travel times.

Failure Modes:
- Elevator car stuck between floors.
- Door does not open or close properly.
- Overspeed or uncontrolled movement.
- Failure of cabin communication system (alarm button, intercom).
- Unreliable floor leveling (car stops above or below threshold).

Causes:
- Motor or brake failure, traction loss, overtemperature.
- Door operator malfunction, worn rollers, obstructed tracks, sensor faults.
- Defective speed governor, encoder failure, control software issues.
- Power supply problems to communication systems, handset damage.
- Incorrect adjustment of leveling sensors or wear of mechanical components.

Local Effects:
- Passengers trapped in the car until rescue.
- Doors repeatedly closing on obstacles, or stuck in open/closed position.
- Comfort issues due to jerky movement or noise.
- Confusing floor alignment, risk of trips and falls.

System Effects:
- Reduced availability of elevator service.
- Long queues at peak hours, lower satisfaction of building occupants.
- Increased frequency of emergency calls and service interventions.

End Effects:
- Potential injuries or panic in case of entrapment events.
- Violations of safety regulations and potential legal consequences.
- Increased operational expenditure and loss of reputation for the building owner.

Current Controls:
- Periodic statutory inspections and preventive maintenance.
- Safety chains and redundant braking systems.
- Door obstacle detection and re-opening logic.
- Emergency alarm system and procedure for trapped passengers.

Assessment (illustrative):
- Severity S = 10 (safety-critical, potential for serious injury).
- Occurrence O = 3 (if maintenance is done by qualified provider).
- Detection D = 4 (most dangerous faults are covered by safety circuits).
- RPN = 10 × 3 × 4 = 120.

Recommended Actions:
- Enhance logging and analysis of fault codes for trend detection.
- Improve training of security and reception staff on entrapment scenarios.
- Implement remote monitoring to detect repeated door faults or overspeed trips.
- Review and optimize maintenance intervals for door mechanisms.