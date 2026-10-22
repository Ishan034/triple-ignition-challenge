<!-- Example output of python -m contamination.demo (synthetic batch B031). Regenerate rather than edit. -->
# Batch investigation report: B031
*full record · reference c3a42a7b2a1b · knowledge base: draft - requires MaverickX scientist review*

## 1. What changed?
- **ph**: low from 26 h (first raised 26 h, persistent); peak observed 6.57 vs reference 6.75 (robust z -9.1); detectors: cusum, ewma, robust_threshold
- **biomass_od600**: high from 32 h (first raised 32.5 h, persistent); peak observed 10.9 vs reference 8.16 (robust z +9.7); detectors: cusum, ewma, robust_threshold
- **enzyme_activity**: low from 36 h (first raised 39 h, persistent); peak observed 78.3 vs reference 119 (robust z -6.5); detectors: cusum, ewma, robust_threshold
- **dissolved_oxygen**: low from 43 h (first raised 43 h, persistent); peak observed 80.4 vs reference 88 (robust z -5.7); detectors: cusum

## 2. Plausible explanations (ranked for review, not probabilities)
- **Biological contamination** (H01): leading, score +11
  - supporting: E01 Several signals deviate together, E05 Biomass / turbidity above reference, E06 Low enzyme activity for the amount of biomass, E07 Culture more acidic than expected, E08 Extra oxygen demand with high biomass, E10 Visual / odour observation recorded, E12 Confirmatory test positive
  - contradicting: none
- **Raw-material or process-condition change** (H03): not supported, score -1
  - supporting: E01 Several signals deviate together
  - contradicting: E12 Confirmatory test positive
- **Equipment or actuator fault** (H06): not supported, score -2
  - supporting: none
  - contradicting: E12 Confirmatory test positive
- **Measurement fault (sensor or probe)** (H04): not supported, score -6
  - supporting: none
  - contradicting: E01 Several signals deviate together, E10 Visual / odour observation recorded, E12 Confirmatory test positive
- **Expected process variation (delayed / slow growth)** (H02): not supported, score -8
  - supporting: none
  - contradicting: E05 Biomass / turbidity above reference, E06 Low enzyme activity for the amount of biomass, E08 Extra oxygen demand with high biomass, E10 Visual / odour observation recorded, E12 Confirmatory test positive
- **Unresolved / previously unseen cause** (H05): fallback, score +0

## 3. What information is missing?
- E17 Controller working unusually hard while holding its variable: No controller outputs are recorded for this process.
- E18 Controlled variable drifts while its controller does not respond: No controller outputs are recorded for this process.

## 4. Assessment (dimensions kept separate)
- **leading explanation**: Biological contamination
- **contamination evidence**: confirmed by independent test
- **process deviation**: substantially abnormal
- **quality impact**: deviation established (enzyme_activity below reference from 36 h)
- **affected scope**: linked cases B035, B040, B045 via medium lot M-12, inoculum lot I-07
- **confidence**: strong (supported by an independent test)

## 5. Possible sources (leads for investigation, not proof)
- **medium lot M-12** (strong lead): 4 of 4 classified batches using it are contamination cases (100%), vs 2 of 51 that did not use it (4%); one-sided Fisher p = 4.4e-05, Bonferroni-adjusted 0.00026 over 6 resources. A shared resource is a lead, not proof of cause.
  - suggested check: Test a retained sample of the medium lot (sterility / bioburden)
- **inoculum lot I-07** (possible lead): 2 of 5 classified batches using it are contamination cases (40%), vs 4 of 50 that did not use it (8%); one-sided Fisher p = 0.086, Bonferroni-adjusted 0.52 over 6 resources. A shared resource is a lead, not proof of cause.
  - suggested check: Check purity of the inoculum stock (streak plate / microscopy)

## 6. Recommended next checks
1. Review batch record (inoculum age, medium lot, setpoints, operator notes) (~0.5 h, low cost): informative about Raw-material or process-condition change (H03)
2. Check the actuators (pump and valve status, flow totaliser vs demand, cooling supply) (~0.5 h, low cost): informative about Raw-material or process-condition change (H03)
3. Plate count on selective medium (~36 h, low cost): informative about Biological contamination (H01)

## 7. Scientist decision
- Reviewed by: ____  ·  Checks performed: ____  ·  Final assessment: ____

## Evidence detail
- E01 Several signals deviate together: **present**. Persistent alerts on: biomass_od600, dissolved_oxygen, enzyme_activity, ph.
- E02 Only one signal deviates: **absent**. Variables alerted: biomass_od600, dissolved_oxygen, enzyme_activity, ph.
- E03 Isolated reading(s) that return to normal: **absent**. Threshold alerts are not limited to isolated readings.
- E04 Trajectory is delayed but otherwise normal: **absent**. Growth phase best fits the reference shifted -1 h (median |z| 0.16 vs 0.62 unshifted, 6 readings).
- E05 Biomass / turbidity above reference: **present**. Biomass above reference from 32 h.
- E06 Low enzyme activity for the amount of biomass: **present**. Activity low with normal/high biomass at 36 h, 40 h, 44 h, 48 h.
- E07 Culture more acidic than expected: **present**. pH below reference from 26 h.
- E08 Extra oxygen demand with high biomass: **present**. DO low while OD600 high at 19 reading(s).
- E09 Deviating signal is decoupled from linked signals: **absent**. Deviations are accompanied by changes in linked signals.
- E10 Visual / odour observation recorded: **present**. 31 h: "Off-colour broth and unusual odour noted (simulated)"
- E11 Data-quality problems near the alerts: **absent**. No data-quality warnings near alerts.
- E12 Confirmatory test positive: **present**. T01 positive at 32 h (simulated: foreign rod-shaped cells seen).
- E13 Confirmatory test negative: **absent**. Checked (T01); result did not indicate this.
- E14 Process condition deviation: **absent**. No persistent low alert on temperature.
- E15 Instrument check found a fault: **unknown**. Not checked yet (T02 would tell).
- E16 Instrument check found no fault: **unknown**. Not checked yet (T02 would tell).
- E17 Controller working unusually hard while holding its variable: **unknown**. No controller outputs are recorded for this process.
- E18 Controlled variable drifts while its controller does not respond: **unknown**. No controller outputs are recorded for this process.
- E19 Equipment check found a fault: **unknown**. Not checked yet (T07 would tell).
- E20 Equipment check found no fault: **unknown**. Not checked yet (T07 would tell).

> Decision support only. Rankings order explanations for scientist review. They are not probabilities, and no organism is identified from process data. The knowledge base is a draft that has not been validated by MaverickX scientists.