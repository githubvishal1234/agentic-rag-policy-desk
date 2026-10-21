# Eval cases for review

31 questions over the synthetic Aldermere corpus in `data/eval_corpus/` (11 docs, 36 chunks at the API's default chunking).

- **Evidence** is the passage retrieval must surface (graded: is a chunk from that file containing this text in the top-k?).
- **Answer facts** are for the later answer eval: each group must appear in the answer; `a / b` means either form counts.
- `unanswerable` cases have no evidence; they are skipped by the retrieval eval and used by the answer eval to check the model says it can't find the answer instead of inventing one.

| id | category | evidence source(s) | answer facts |
|---|---|---|---|
| lookup-01 | lookup | leave_policy.txt | 22 |
| lookup-02 | lookup | travel_policy.txt | 280 |
| lookup-03 | lookup | onboarding_guide.txt | 6 months / six months |
| lookup-04 | lookup | expense_policy.txt | 45 |
| lookup-05 | lookup | benefits_overview.txt | 1500 |
| lookup-06 | lookup | facilities_and_safety.txt | north car park |
| paraphrase-01 | paraphrase | leave_policy.txt | 5 days / five days |
| paraphrase-02 | paraphrase | it_security_policy.txt | 2 hours / two hours; service desk |
| paraphrase-03 | paraphrase | remote_work_policy.txt | 400 |
| paraphrase-04 | paraphrase | benefits_overview.txt | Employee Assistance Program / EAP |
| paraphrase-05 | paraphrase | data_retention_policy.txt | 6 months / six months |
| paraphrase-06 | paraphrase | travel_policy.txt | business class; director |
| paraphrase-07 | paraphrase | benefits_overview.txt | 90 days |
| paraphrase-08 | paraphrase | leave_policy.txt | 5 / five; 31 March / March 31 |
| exact-01 | exact_term | it_security_policy.txt | USB / encrypted drive / removable |
| exact-02 | exact_term | remote_work_policy.txt | abroad / another country |
| exact-03 | exact_term | expense_policy.txt | training |
| exact-04 | exact_term | code_of_conduct.txt | gift / hospitality |
| exact-05 | exact_term | code_of_conduct.txt | 5050 |
| exact-06 | exact_term | it_security_policy.txt | remote |
| exact-07 | exact_term | performance_review_policy.txt | 60 days / sixty days |
| multi-01 | multi_part | leave_policy.txt, remote_work_policy.txt | 22; three days / 3 days |
| multi-02 | multi_part | travel_policy.txt | 280; 95 |
| multi-03 | multi_part | expense_policy.txt | director; receipt |
| multi-04 | multi_part | onboarding_guide.txt | security awareness; code of conduct; 3 months / three months |
| multi-05 | multi_part | data_retention_policy.txt, it_security_policy.txt | 13 months / thirteen months; Vault |
| unanswerable-01 | unanswerable | — | (must not invent an answer) |
| unanswerable-02 | unanswerable | — | (must not invent an answer) |
| unanswerable-03 | unanswerable | — | (must not invent an answer) |
| unanswerable-04 | unanswerable | — | (must not invent an answer) |
| unanswerable-05 | unanswerable | — | (must not invent an answer) |

## lookup-01

````text
How many days of annual leave do full-time employees get?
````

- `leave_policy.txt`: "22 days of paid annual leave"

## lookup-02

````text
What is the hotel cap per night in Tier 1 cities?
````

- `travel_policy.txt`: "$280 in Tier 1 cities"

## lookup-03

````text
How long is the probation period for new employees?
````

- `onboarding_guide.txt`: "probation period of 6 months"

## lookup-04

````text
Within how many days must expenses be submitted?
````

- `expense_policy.txt`: "within 45 days of the date the cost was incurred"

## lookup-05

````text
How much is the learning budget per year?
````

- `benefits_overview.txt`: "learning budget of $1,500 per year"

## lookup-06

````text
Where is the fire assembly point?
````

- `facilities_and_safety.txt`: "The assembly point is the north car park"

## paraphrase-01

````text
My father died last week. How much paid time off am I allowed?
````

- `leave_policy.txt`: "up to 5 days of paid bereavement leave for the death of an immediate family member"

## paraphrase-02

````text
I left my work phone in a taxi. What should I do?
````

- `it_security_policy.txt`: "must be reported to the Service Desk within 2 hours"

## paraphrase-03

````text
Can I get money back for a new desk chair for my home?
````

- `remote_work_policy.txt`: "one-time home office stipend of $400"

## paraphrase-04

````text
Is there someone I can talk to confidentially if I'm struggling with stress?
````

- `benefits_overview.txt`: "free, confidential counselling 24 hours a day"

## paraphrase-05

````text
How long do we keep paperwork about people we decided not to hire?
````

- `data_retention_policy.txt`: "Records of unsuccessful job applicants are deleted 6 months after the hiring decision"

## paraphrase-06

````text
Can I fly in a more comfortable cabin on a 12-hour trip to Sydney?
````

- `travel_policy.txt`: "Business class is allowed only for flights over 10 hours and requires director approval"

## paraphrase-07

````text
When can I start claiming for teeth cleaning?
````

- `benefits_overview.txt`: "Dental and vision cover begin after 90 days"

## paraphrase-08

````text
What happens to vacation I don't use this year?
````

- `leave_policy.txt`: "Up to 5 unused days may be carried over into the next year"

## exact-01

````text
What is form SEC-11 used for?
````

- `it_security_policy.txt`: "An exception for an encrypted drive can be requested with form SEC-11"

## exact-02

````text
What is form HR-22?
````

- `remote_work_policy.txt`: "A Remote Work Abroad Request (form HR-22)"

## exact-03

````text
What does the expense code EXP-TRN cover?
````

- `expense_policy.txt`: "EXP-TRN for training and courses"

## exact-04

````text
What is ETH-07?
````

- `code_of_conduct.txt`: "must be declared on form ETH-07 within 7 days"

## exact-05

````text
What is the extension for the Speak Up line?
````

- `code_of_conduct.txt`: "by phone on extension 5050"

## exact-06

````text
When do I need NetGate?
````

- `it_security_policy.txt`: "Remote connections to internal systems must go through the NetGate VPN"

## exact-07

````text
How long does a PIP last?
````

- `performance_review_policy.txt`: "performance improvement plan (PIP) lasting 60 days"

## multi-01

````text
How many annual leave days do I get, and how many days a week do I need to be in the office?
````

- `leave_policy.txt`: "22 days of paid annual leave"
- `remote_work_policy.txt`: "expected in the office three days per week"

## multi-02

````text
What's the hotel limit in London and the daily meal allowance for an international trip?
````

- `travel_policy.txt`: "$280 in Tier 1 cities"
- `travel_policy.txt`: "$95 for international trips"

## multi-03

````text
Who has to approve a $4,000 expense claim, and do I need receipts?
````

- `expense_policy.txt`: "Claims above $2,500 also require approval from a director"
- `expense_policy.txt`: "Itemised receipts are required for any expense over $25"

## multi-04

````text
Which trainings must I finish in my first month, and when is my first probation check-in?
````

- `onboarding_guide.txt`: "Security awareness training and code of conduct training must be completed within your first 30 days"
- `onboarding_guide.txt`: "check-in reviews at 3 months"

## multi-05

````text
How long are security logs kept, and where must salary records be stored?
````

- `data_retention_policy.txt`: "System and security logs are retained for 13 months"
- `it_security_policy.txt`: "may only be stored in the Vault document system"

## unanswerable-01

````text
Can I bring my dog to the office?
````

## unanswerable-02

````text
How many stock options do new employees receive?
````

## unanswerable-03

````text
What is the dress code?
````

## unanswerable-04

````text
What is the maximum annual bonus as a percentage of salary?
````

## unanswerable-05

````text
Does Aldermere offer a paid sabbatical after five years of service?
````
