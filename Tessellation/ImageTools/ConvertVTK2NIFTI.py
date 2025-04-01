import os
import argparse
import vtk
import nibabel as nib
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy


class ConvertVTK2NIFTI():
    """Converts vtk image formats (.vti, .vtk) into NIFTI image formats (.nii, .nii.gz)
    """
    def __init__(self,Args) -> None:
        self.Args = Args

    def GetImages(self) -> list:
        """Gets all of the vtk images within the input directory

        Returns:
            list:a list of str 
        """
        vtk_Images = []
        
        for filename in os.listdir(self.Args.InputFolder):
            if filename.endswith('vti') or filename.endswith('vtk'):
                vtk_Images.append(filename)
        
        return vtk_Images
    
    def ReadVTKImage(self,path) -> vtk.vtkImageData:
        """Reads the vtk image and returns its scalar data

        Args:
            path (str): path to the vtk image

        Returns:
            vtk.vtkImageData: Image scalars (pixel values)
        """

        if path.endswith('.vti'):
            reader = vtk.vtkXMLImageDataReader()
        elif path.endswith('.vtk'):
            reader = vtk.vtkStructuredPointsReader()
        
        reader.SetFileName(path)
        reader.Update()
        
        return reader.GetOutput()

    def vtk2numpy(self,Image) -> np.array:
        """Converts vtk image scalar data into a numpy array with the same dimensions.

        Args:
            Image (vtk.vtkImageData): The output of ReadVTKImage

        Returns:
            np.array: numpy array of the image scalars
        """
        
        vtk_data = Image.GetPointData().GetScalars()
        dims = Image.GetDimensions()
        numpy_data = vtk_to_numpy(vtk_data)
        numpy_data = numpy_data.reshape(dims,order='F')
        
        return numpy_data
    
    def numpy2NIFTI(self,numpy_data, nfimage_path, vtkImage) -> None:
        """Converts a numpy array into a nifti image and saves the nifti file within the provided path

        Args:
            numpy_data (np.array): output of the vtk2numpy
            nfimage_path (str): output nifti image path
        """

        spacing = vtkImage.GetSpacing()
        origin = vtkImage.GetOrigin()

        affine = np.eye(4) #no scaling, rotation, or translation is applied
        #construct affin transform matrix based on the spacing and origin of the initial image
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]

        affine[0, 0] *= -1  # Flip the X axis
        affine[1, 1] *= -1  # Flip the Y axis

        vtk_origin_ras = [-origin[0], -origin[1], origin[2]]
        affine[:3, 3] = vtk_origin_ras

        nifti_Image = nib.Nifti1Image(numpy_data, affine)
        nib.save(nifti_Image, nfimage_path)

    def main(self) -> None:
        """Takes the input folder and reads all of the vtk images inside
        and converts them into NIFTI images and saves in the output directory.
        """
        output_dir = "ImageNF"
        if output_dir not in os.listdir(self.Args.InputFolder):
            os.system(f"mkdir {self.Args.InputFolder}/{output_dir}")
        self.output_dir = f"{self.Args.InputFolder}/{output_dir}"


        vtk_Images = self.GetImages()
        vtk_Images.sort()

        for image in vtk_Images:
            image_path = f"{self.Args.InputFolder}/{image}"
            nfimage, _ = os.path.splitext(image)
            nfimage_path = f"{self.output_dir}/{nfimage}{self.Args.Nformat}"

            print(f'---Converting vtk to NIFTI: {image}')
            Image = self.ReadVTKImage(image_path)
            numpy_data = self.vtk2numpy(Image)
            self.numpy2NIFTI(numpy_data, nfimage_path, Image)

            
            

if __name__=="__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("-InputFolder", "--InputFolder", type= str, default="./", dest= "InputFolder", required=True, help="the folder containing vtk format images")
    Parser.add_argument("-Nformat", "--Nformat", type= str, default=".nii.gz", dest="Nformat", required= False, help="NIFTI supported file formats: .nii or .nii.gz")
    args = Parser.parse_args()

    ConvertVTK2NIFTI(args).main()