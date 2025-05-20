

from huggingface_hub import list_repo_files, hf_hub_download

YOUR_FOLDER = r'/media/vieunite/New Volume1/data'

REPO_ID = "quchenyuan/360x_dataset_HR"
TARGET_FOLDER = "panoramic"
token = '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'
files = list_repo_files(REPO_ID, repo_type='dataset', token=token)

target_files = [f for f in files if f.startswith(TARGET_FOLDER)]

# 批量下载
downloaded_files = []
for file in target_files:
    file_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type='dataset',
        filename=file,
        cache_dir=YOUR_FOLDER,
        token=token
    )
    downloaded_files.append(file_path)
    print(f"已下载: {file}")

print(f"所有文件已下载到: {downloaded_files}")