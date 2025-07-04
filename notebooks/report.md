
### 📊 Summary of EDA and Preprocessing

The CFPB complaint dataset contains nearly 9.6 million records, but only about 2.98 million of them include consumer complaint narratives. To align with the challenge’s business goals, we applied a strict filter that focuses on five key financial products: Credit card, Personal loan, BNPL, Savings account, and Money transfers. After removing records without narratives, we retained **82,164 cleaned complaints**—a focused dataset designed to maximize business relevance and retrievability.

However, the EDA also revealed that several high-volume product categories were excluded—such as Credit reporting and Debt collection—with millions of complaints containing rich narrative content. To account for broader use cases and future tuning, we processed an **expanded dataset** as well, including more product types, which resulted in **1.27 million cleaned complaint narratives**.

No word count threshold was enforced at this stage, ensuring that short but potentially useful entries remain available for downstream filtering or model evaluation. This dual-path strategy (strict and expanded filtering) offers flexibility for experimentation and supports a robust RAG development pipeline.

---

