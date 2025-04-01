import vtk
import argparse

class vtk_image_gradient():
    def __init__(self, args):
        self.Args = args
    
    def gradient_filter(self, vtk_image):
        gradient_filter = vtk.vtkImageGradient()
        gradient_filter.SetInputData(vtk_image)
        gradient_filter.SetDimensionality(3)
        gradient_filter.Update()

        return gradient_filter.GetOutput()
    
    def define_borders(self, gradient_image):
        magnitude_filter = vtk.vtkImageMagnitude()
        magnitude_filter.SetInputData(gradient_image)
        magnitude_filter.Update()

        return magnitude_filter.GetOutput()
    
    def Write_vtk_image(self,Image):
        writer = vtk.vtkDataSetWriter()#vtkXMLImageDataWriter()
        writer.SetFileName("./Gradient_Image.vtk")
        writer.SetInputData(Image)
        writer.Write()

    def Read_vtk_image(self, image_location):
        reader = vtk.vtkStructuredPointsReader()
        reader.SetFileName(image_location)
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        return reader.GetOutput()

    def main(self):
        input_image = self.Read_vtk_image(self.Args.InputImage)
        gradient_image = self.gradient_filter(input_image)
        magnitude_gradient_image = self.define_borders(gradient_image)
        self.Write_vtk_image(magnitude_gradient_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-InputImage", "--InputImage", dest= "InputImage", type= str, required=True)
    args = parser.parse_args()

    vtk_image_gradient(args).main()