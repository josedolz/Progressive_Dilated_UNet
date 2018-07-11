# UNet Progressive dilated convolutions: application to bladder multi-region segmentation (2D)


## Requirements

- The code has been written in Python (3.5.2) and requires [PyTorch](https://pytorch.org) (Version 0.4.0)
- You should also have installed different modules, such as numpy, torchvision, or some others (if executing the code complains, just install the missing module)
## Running the code

## Training

### How do I train my own architecture from scratch?

To start with your own architecture, you have to modify the file "main.py" according to your requirements.

- Line 86. Define the path to your images (i.e.->root_dir). Images should be saved as .png, and the structures should be:
 - Training images: ../path/train-->Img
 - Training GT: ../path/train-->GT
 - Validation images: ../path/val-->Img
 - Validation GT: ../path/val-->GT
- Line 87. Change your model name. This is the name to save the trained model, as well as all its statistics during training/validation.


Then you simply have to write in the command line (from the code folder):

```
CUDA_VISIBLE_DEVICES=X python main.py
```

Where X indicates the GPU where to run the training.

After some epochs (over 100) typically top performance is achieved, and the best model is saved in model/ folder.


## Testing

### How can I use a trained model?

Once you are satisfied with your training, you can evaluate it by writing this in the command line:

```
CUDA_VISIBLE_DEVICES=X python Inference.py ./model/Best_Your_ModelName.pkl FolderToSaveResults_Name
```
Where X indicates the GPU where to run the inference.

Images as .png, as well as .mat files are saved. Remember these images are in 2D, so they will have to be reconstructed back.
