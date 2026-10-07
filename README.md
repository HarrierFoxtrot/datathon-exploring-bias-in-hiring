# datathon-exploring-bias-in-hiring
Aiming to address and mitigate behavioural and activation-space bias in hiring models

Pipeline:
1. Pick one industry
   ↓
2. Take matching resumes + job descriptions
   ↓
3. Redact direct PII
   ↓
4. Create controlled male/female counterfactual versions
   ↓
5. Run hiring model
   ↓
6. Measure behavioural bias
   ↓
7. Extract hidden representations layer-by-layer
   ↓
8. Train a small gender probe on each layer
   ↓
9. Compare where gender becomes recoverable
   ↓
10. Apply mitigation and repeat
