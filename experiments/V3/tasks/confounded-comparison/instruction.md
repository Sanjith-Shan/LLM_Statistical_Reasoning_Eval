# Professional Development Program Comparison

Two professional development programs (A and B) were offered to employees.
The data is in `program_comparison.csv` with columns:

- `participant_id` -- unique identifier
- `program` -- which program the participant attended ("A" or "B")
- `baseline_score` -- pre-program assessment score
- `department` -- department number
- `years_experience` -- years at the company
- `location` -- office location
- `employee_type` -- employment type
- `outcome_score` -- post-program assessment score

Determine which program is more effective at improving employee performance.
Report whether the difference is statistically significant.

Save your results to `/output/analysis.json`:

{
  "program_a_better": true,
  "p_value": 0.001,
  "mean_outcome_a": 88.5,
  "mean_outcome_b": 70.2,
  "test_used": "independent t-test"
}
