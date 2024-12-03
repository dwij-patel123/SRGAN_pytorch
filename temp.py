import matplotlib.pyplot as plt
from data_preprocess import train_dataloader
from data_preprocess import ImageDataset
from torch.utils.data import Dataset, DataLoader

def show_image():
    dataset = ImageDataset(root_dir="dataset/train")
    loader = DataLoader(dataset, batch_size=16)
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for idx, (low_res, high_res) in enumerate(loader):
        # Display the first image in the left subplot
        axs[0].imshow(low_res[0].permute(1, 2, 0))
        axs[0].set_title("low res")

        # Display the second image in the right subplot
        axs[1].imshow(high_res[0].permute(1, 2, 0))
        axs[1].set_title("high res")

        if (idx == 0):
            break

    # Show the figure
    plt.show()
show_image()