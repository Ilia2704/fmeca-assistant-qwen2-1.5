## Example: Conveyor System in Packaging Line

Function:
Transport packaged products from the filling machine to the palletizing area at the required throughput.

Failure Modes:
- Conveyor belt breakage or severe misalignment.
- Motor or gearbox failure on a drive station.
- Blockage or jam due to fallen products or foreign objects.
- Failure of photoelectric sensors used for accumulation control.
- Incorrect speed setting or loss of synchronization with upstream equipment.

Causes:
- Ageing and wear of the belt, insufficient tensioning, damaged splices.
- Lubrication issues, overload, misalignment of shafts, bearing failure.
- Poor housekeeping, loose packaging materials, tools left on the conveyor.
- Sensor contamination, misalignment, damaged cables or PLC input cards.
- Manual overrides in control system, wrong recipe selection, software bugs.

Local Effects:
- Visible belt damage, abnormal noise or vibration.
- Accumulation of products in specific sections, starved sections downstream.
- Frequent stops and restarts of motors due to overload trips.
- Unreliable presence detection, false accumulation or gaps.

System Effects:
- Reduced or unstable throughput of the entire packaging line.
- Increased reject rate due to product collisions or falls.
- Increased operator workload to clear jams and restart sequences.

End Effects:
- Missed production targets and delivery delays.
- Higher maintenance costs and spare parts consumption.
- Potential injuries if jams are cleared without proper lockout/tagout.

Current Controls:
- Preventive maintenance on belts, rollers, and gearboxes.
- Torque-limited couplings and overload protection on motors.
- Guards and interlocks on access points.
- Periodic cleaning and inspection of sensors and reflectors.

Assessment (illustrative):
- Severity S = 7 (productivity loss, possible minor injuries).
- Occurrence O = 6 (if cleaning and PM are often deferred).
- Detection D = 4 (problems are usually visible but sometimes late).
- RPN = 7 × 6 × 4 = 168.

Recommended Actions:
- Introduce standardized cleaning routines and visual 5S checks.
- Add condition-based monitoring for critical gearboxes (vibration, temperature).
- Improve operator training on jam clearing and lockout procedures.
- Implement simple line OEE dashboard to highlight recurring stops.