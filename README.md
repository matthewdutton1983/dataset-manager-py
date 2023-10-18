# huggingface-dataloader-py

Python wrapper over the HuggingFace datasets library that makes it easier to load and convert datasets.

```python
# Import the HuggingFaceDatasetLoader class
from dataloaders.huggingface import HuggingFaceDatasetLoader

# Instantiate a new HuggingFaceDatasetLoader object
loader = HuggingFaceDatasetLoader()

# Download a dataset from the HuggingFace Hub
dataset = loader.load_from_hub(dataset_name="cuad")

# Calling dataset will print out the top-level detail about the dataset
dataset

DatasetDict({
    train: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 22450
    })
    test: Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 4182
    })
})

# You can also save the dataset to disk
loader.save_to_disk(path="cuad-dataset")

# And reload the dataset from disk
reloaded_dataset = loader.load_from_disk(path="cuad-dataset")

# It's also possible to compress the dataset into either a zip file or a tarball
# Defaults to the 'zip' format
loader.archive_dataset(dataset_dir="cuad-dataset", archive_path=".", archive_format="zip")
```
