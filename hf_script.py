from huggingface_hub import HfApi
import os

api = HfApi()
repo = "synthesis-II/tst_translation/checkpoint-83301"
dirs = os.listdir(repo)

for file in dirs:
    if ".git" not in file:
        api.upload_file(
            path_or_fileobj="synthesis-II/tst_translation/checkpoint-83301/" + str(file),
            path_in_repo=file.split("/")[-1],
            repo_id="mustaphounii04/marianNMT_automobiles",
            repo_type="model",
        )

