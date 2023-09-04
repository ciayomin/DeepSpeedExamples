import datasets

ds = datasets.load_dataset("openai/summarize_from_feedback", "comparisons", cache_dir="/disk1/datasets/temp")
ds.save_to_disk("/disk1/datasets/summarize_from_feedback")



