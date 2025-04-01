import vtk
#import nibabel as nib
import numpy as np
import os
import argparse
from vtk.util.numpy_support import vtk_to_numpy
from ConvertVTK2NIFTI import ConvertVTK2NIFTI

class ConvertModel2LabelMap():
    """Takes the surface model (.vtp) as input and returns a binary label map segmentation in vtk (.vtk) and NIFTI (.nii.gz) formats
    """
    def __init__(self, Args) -> None:
        """setting the input arguments and the path of the output image.

        Args:
            Args (NameSpace): Input arguments
        """
        self.Args = Args
        output_dir, InputSurfaceName = os.path.split(self.Args.InputSurface)
        InputSurfaceName = os.path.splitext(InputSurfaceName)[0]
        _, ImageName = os.path.split(self.Args.InputImage)
        ImageName = os.path.splitext(ImageName)[0]
        self.output_file_path = os.path.join(output_dir,ImageName + "_" +InputSurfaceName)

    def VTPReader(self,path: str) -> vtk.vtkPolyData:
        """Reads the surface model file

        Args:
            path (str): path to the surface model file

        Returns:
            vtk.vtkPolyData: surface model dataset
        """
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput()
    
    def CreateLabeledImage(self, Enclosed:vtk.vtkUnsignedCharArray, Image:vtk.vtkImageData) -> vtk.vtkStructuredGrid:
        """ takes the array of the binary label map and projects it into an image with same origin and spacing and dimensions of the input image

        Args:
            Enclosed (vtk.vtkUnsignedCharArray): binary label map array
            Image (vtk.vtkImageData): input image

        Returns:
            vtk.vtkStructuredGrid: binary labeled image
        """
        
        bounds = Image.GetBounds()
        origin = Image.GetOrigin()
        spacing = Image.GetSpacing()

        LabeledImage = vtk.vtkImageData()
        LabeledImage.SetSpacing(spacing)
        LabeledImage.SetOrigin(origin)

        dims = [
            int((bounds[1]-bounds[0]) / spacing[0]),
            int((bounds[3]-bounds[2]) / spacing[1]),
            int((bounds[5]-bounds[4]) / spacing[2])
        ]

        LabeledImage.SetDimensions(dims)

        LabeledImage.AllocateScalars(vtk.VTK_FLOAT, 1)
        
        for i in range(Image.GetNumberOfPoints()):
            (x,y,z) = Image.GetPoint(i)
            scalar = Enclosed.GetValue(i)
            iX = int(round((x-origin[0])/spacing[0]))
            iY = int(round((y-origin[1])/spacing[1]))
            iZ = int(round((z-origin[2])/spacing[2]))

            if 0 <= iX < dims[0] and 0 <= iY < dims[1] and 0 <= iZ < dims[2]:
                LabeledImage.SetScalarComponentFromFloat(iX,iY,iZ,0,scalar)

        return LabeledImage


    def LabelEnclosedPoints(self, Surface: vtk.vtkPolyData, Image:vtk.vtkImageData) -> vtk.vtkUnsignedCharArray:
        """Takes the input image and input surface and returns the labeled array where 1 are pixels that are enclosed the surface

        Args:
            Surface (vtk.vtkPolyData): Input vtp surface
            Image (vtk.vtkImageData): Input vtk image

        Returns:
            vtk.vtkUnsignedCharArray: binary labeled array
        """
        
        PointsVTK=vtk.vtkPoints()
        PointsVTK.SetNumberOfPoints(Image.GetNumberOfPoints())

        #structuredgrid = vtk.vtkImageData()

        for i in range(Image.GetNumberOfPoints()):
            PointsVTK.SetPoint(i,Image.GetPoint(i))
            #structuredgrid.GetPointData().Setpoint(i,Image.GetPoint(i))
		
        print ("--- Converting Image Points into a Polydata")
		#Convert into a polydata format
        pdata_points = vtk.vtkPolyData()
        pdata_points.SetPoints(PointsVTK)

        #labels = np.zeros(num_points, dtype=np.uint8)
        selectEnclosed = vtk.vtkSelectEnclosedPoints()
        selectEnclosed.SetInputData(pdata_points) #Points in the Image
        selectEnclosed.SetSurfaceData(Surface) #Surface Model
        selectEnclosed.SetTolerance(0.000000001)
        selectEnclosed.Update()
    
        return selectEnclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
    
    def WriteVTK(self, Image: vtk.vtkImageData) -> None:
        writer = vtk.vtkDataSetWriter()#vtkXMLImageDataWriter()
        writer.SetFileName(self.output_file_path + ".vtk")
        writer.SetInputData(Image)
        writer.Write()

    def main(self):
        Surface = self.VTPReader(self.Args.InputSurface)
        print("type(Surface) = ", type(Surface))
        class_arguments = argparse.Namespace(InputFolder=None, Nformat=".nii.gzz")
        Image = ConvertVTK2NIFTI(class_arguments).ReadVTKImage(self.Args.InputImage)
        print("type(Image) = ", type(Image))
        Enclosed = self.LabelEnclosedPoints(Surface, Image)
        Labeled_Image = self.CreateLabeledImage(Enclosed, Image)
        print("type(Labeled_Image) = ",type(Labeled_Image))
        self.WriteVTK(Labeled_Image)
        np_array = ConvertVTK2NIFTI(class_arguments).vtk2numpy(Labeled_Image)
        ConvertVTK2NIFTI(class_arguments).numpy2NIFTI(np_array, self.output_file_path + ".nii.gz", Image)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-InputSurface", "--InputSurface", required=True, dest="InputSurface", type=str)
    parser.add_argument("-InputImage", "--InputImage", required=True, dest="InputImage", type=str)
    args = parser.parse_args()

    ConvertModel2LabelMap(args).main()