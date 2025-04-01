import os
import random
import argparse

class shuffle_images():
    def __init__(self,Args) -> None:
        self.Args = Args

    def GetImages(self) -> list:
        images = []
        for filename in os.listdir(self.Args.InputFolder):
            if self.Args.format in filename:
                images.append(filename)

        return images

    def shuffle(self, original_images) -> list:
        random.shuffle(original_images)
        num_selected_images = int(len(original_images)*self.Args.percentage)

        return original_images[:num_selected_images]

    def main(self):
        images = self.GetImages()
        selected_images = self.shuffle(images)
        for filename in selected_images:
            os.system(f"cp {self.Args.InputFolder}/{filename} {self.Args.OutputFolder}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-InputFolder", "--InputFolder", dest="InputFolder", required=True, type=str, help="Input Folder contains images")
    parser.add_argument("-format", "--format", dest="format", required=False, default=".nii.gz", type=str, help="the format of the image")
    parser.add_argument("-percentage", dest="percentage", required=False, default=0.25, type=float, help="the percentage of images to be selected")
    parser.add_argument("-OutputFolder", "--OutputFolder", dest="OutputFolder", required=True, type=str, help="Output directory")
    args = parser.parse_args()

    shuffle_images(args).main()