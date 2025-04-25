from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
#path_or_fileobj="marianNMT_automobiles",
path_in_repo="marianNMT_automobiles/",
folder_path="marianNMT_automobiles/",
repo_id="mustaphounii04/marianNMT_automobiles",
token="basurita",
repo_type='model'
)
