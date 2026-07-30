# 2. Ground truth — failure events

The dataset is **unlabeled**; UCI provides a company failure-report table for
evaluation. All four events are **Air leak / High stress**.

| # | Start | End | Maintenance |
|---|---|---|---|
| 1 | 2020-04-18 00:00 | 2020-04-18 23:59 | not recorded |
| 2 | 2020-05-29 23:30 | 2020-05-30 06:00 | 2020-05-30 12:00 ¹ |
| 3 | 2020-06-05 10:00 | 2020-06-07 14:30 | 2020-06-08 16:00 |
| 4 | 2020-07-15 14:30 | 2020-07-15 19:00 | 2020-07-16 00:00 |

Documented deviations from source (label-construction decisions):
- ¹ The source table prints "Maintenance on 30**Apr** at 12:00" for event 2,
  which predates the failure by a month and contradicts every other row (repair
  follows failure by hours). Treated as a typo for 30 May 12:00.
- The source table numbers the events **#1, #1, #3, #4** — the second entry
  should be #2. Ids here are sequential.
- Event 1 has no maintenance entry, so its repair time is unknown; a
  conservative fixed fallback margin is used instead.

