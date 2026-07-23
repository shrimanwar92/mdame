python profiling.py

python cleaning/generate_domain_policy.py 

python cleaning/run_clustering_pipeline.py

python clean_profiler.py

python feature_engg/generate_feature_engg_strategy.py

python feature_engg/run_clustering_pipeline.py

python training/train_clustering.py


## Inference steps
cleaning_pipeline = joblib.load(CLEANING_PIPELINE_PATH)

feature_pipeline = joblib.load(FEATURE_PIPELINE_PATH)

model = joblib.load(BEST_MODEL_PATH)

raw_review = pd.DataFrame([review])

silver = cleaning_pipeline.transform(raw_review)

gold = feature_pipeline.transform(silver)

cluster = model.predict(gold)