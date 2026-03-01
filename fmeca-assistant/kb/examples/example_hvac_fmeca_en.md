## Example: HVAC System for Data Center

Function:
Maintain temperature and humidity within the specified limits for safe server operation.

Failure Modes:
- Loss of cooling capacity in one chiller unit.
- Complete failure of a Computer Room Air Conditioner (CRAC) unit.
- Obstructed air filters in supply ducts.
- Incorrect setpoint configuration in the building management system (BMS).
- Simultaneous failure of redundant cooling paths during maintenance.

Causes:
- Chiller: refrigerant leak, compressor seizure, power supply failure.
- CRAC: fan motor failure, bearing wear, clogged coils, controller fault.
- Filters: lack of maintenance, construction dust, poor installation.
- Controls: wrong parameters download, firmware bug, operator mistake.
- Redundancy: poor maintenance planning, overlapping downtime windows.

Local Effects:
- Reduced local airflow in affected cold aisles.
- Increased return air temperature for a subset of racks.
- Frequent compressor cycling and nuisance alarms on BMS.

System Effects:
- Global rise of inlet temperature for multiple server racks.
- Activation of high-temperature alarms at server level.
- Performance throttling, unexpected reboots, or forced shutdown of servers.

End Effects:
- Degraded data center availability and SLA violations.
- Lost transactions or outage of business-critical applications.
- Reputational damage and potential contractual penalties.

Current Controls:
- Temperature and humidity sensors per aisle or per rack group.
- BMS alarms for chiller trips and CRAC status.
- Preventive maintenance on filters, coils, and fans.
- Periodic integrated functional tests of redundant cooling paths.

Assessment (illustrative):
- Severity S = 9 (service outage is business-critical).
- Occurrence O = 4 (if maintenance is reasonably done).
- Detection D = 5 (alarms exist, but can be noisy and misconfigured).
- RPN = 9 × 4 × 5 = 180.

Recommended Actions:
- Improve zoning of temperature monitoring (per rack instead of per room).
- Add automatic workload migration or load shedding when temperature rises.
- Tighten PM intervals for filters in dusty environments.
- Introduce change management for BMS configuration with rollback.