---
name: General Feedback & Discussion
about: Share your experience, feedback, or start a discussion about your test.
title: "[FEEDBACK] "
labels: feedback, discussion
assignees: ''

---

**Thank you for taking the time to provide feedback! This is incredibly valuable for fine-tuning the software.**

**1. What were you trying to do?**
A brief summary of your goal. (e.g., "I was trying to generate playlists for my rock collection," or "I was testing the new embedding clustering feature.")

**2. What did you like?**
Please share what you enjoyed or what worked well. (e.g., "The AI naming was surprisingly accurate," or "The analysis task was faster than I expected.")

**3. What was confusing or could be improved?**
Was there anything in the UI, the process, or the results that felt off, was hard to understand, or could be better? This is the perfect place for "gut feelings"!

**4. Task Parameters You Used (Very Important!)**
If your feedback is related to an Analysis or Clustering task, please share the key parameters you used. This helps reproduce the context of your experience.

*   **Analysis Task:**
    *   `NUM_RECENT_ALBUMS`:
    *   Any other configuration used for clustering.
*   **Clustering Task:**
    *   `ENABLE_CLUSTERING_EMBEDDINGS`: (True/False)
    *   `CLUSTER_ALGORITHM`: (e.g., kmeans, gmm, dbscan)
    *   `NUM_CLUSTERS_MIN / MAX`:
    *   `CLUSTERING_RUNS`:
    *   `SCORE_WEIGHT_DIVERSITY` / `SCORE_WEIGHT_PURITY`:
    *   `AI_MODEL_PROVIDER`: (e.g., NONE, OLLAMA, GEMINI)
    *   Any other configuration used for clustering.


**5. Any Other Thoughts or Ideas?**
Use this space for any other comments, suggestions, or general impressions you have.