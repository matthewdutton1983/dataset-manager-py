import shutil
from datasets import load_dataset, load_from_disk


class HuggingFaceDatasetLoader:
    def __init__(self):
        self.dataset = None

    def load_from_hub(self, dataset_name):
        self.dataset = load_dataset(dataset_name)
        return self.dataset
    
    def load_from_disk(self, path):
        self.dataset = load_from_disk(path)
        return self.dataset
    
    def save_to_disk(self, path):
        if self.dataset is not None:
            self.dataset.save_to_disk(path)
        else:
            raise ValueError("No dataset loaded, cannot save.")
        
    def archive_dataset(self, dataset_dir, archive_path, archive_format="zip"):
        """Archives the dataset directory into a zip file or tarball.
        
        Args:
            dataset_dir (str): Path to the dataset directory.
            archive_path (str): Path where the archive will be saved.
            archive_format (str): Format of the archive, "zip" or "tar".
        """
        if archive_format not in ["zip", "tar"]:
            raise ValueError("Invalid archive format. Use 'zip' or 'tar'.")
        
        shutil.make_archive(
            base_name=archive_path,
            format=archive_format,
            root_dir=dataset_dir
        )
        
