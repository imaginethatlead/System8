# System8

System 8 is a clean-room pipeline for clustering TikTok-10M metadata using text-only evidence. The primary entrypoint is:

```bash
python -m sys8.run <input_parquet_or_hf_name> --output-dir outputs
```

Artifacts are written to the chosen output directory and include prepared evidence, embeddings, clusters, QA reports, and operator cards (when integrity gates pass). See `SYS8_OPERATOR_GUIDE.md` for detailed operator steps.
