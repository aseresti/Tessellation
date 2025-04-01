import vtk
import os
import argparse
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utilities import WriteVTUFile, ReadVTUFile, ThresholdInBetween, ClosestPoint



class ExtractSpatialCorrespondance():
    def __init__(self, Args):
        self.Ischemic = ReadVTUFile(Args.Ischemic)
        self.Territory = ReadVTUFile(Args.Territory)
        #self.Ischemic = ThresholdInBetween(self.Myocardium, "ImageScalars", 0, Args.LowerThreshold)
        OutputFileName = os.path.splitext(os.path.basename(Args.Ischemic))[0]
        self.OutputFileName1 = f"{OutputFileName}_Territory_Overlap.vtu"
        self.OutputFileName2 = f"{OutputFileName}_Ischemic_Overlap.vtu"

    def ExtractOverlap(self):
        OverlapTag_Ischemic = np.zeros(self.Ischemic.GetNumberOfPoints())
        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(self.Territory)
        point_locator.BuildLocator()

        tolerance = 0.01
        dist2 = vtk.reference(0.0)
        Mean_dist = 0
        counter = 0
        for i in range(self.Ischemic.GetNumberOfPoints()):
            point = self.Ischemic.GetPoint(i)
            closest_point_id = point_locator.FindClosestPointWithinRadius(tolerance, point, dist2)
            if closest_point_id != -1:
                OverlapTag_Ischemic[i] = 1
            else:
                counter += 1
                point_ = self.Territory.GetPoint(i)
                Mean_dist += vtk.vtkMath.Distance2BetweenPoints(point, point_) ** 0.5
            
        Mean_dist /= counter

                
        OverlapTagVTK = numpy_to_vtk(OverlapTag_Ischemic, deep=True)
        OverlapTagVTK.SetName("Overlap")
        self.Ischemic.GetPointData().AddArray(OverlapTagVTK)
        self.Ischemic.Modified()

        WriteVTUFile(self.OutputFileName2, self.Ischemic)

        OverlapTag_Territory = np.zeros(self.Territory.GetNumberOfPoints())
        point_locator = vtk.vtkPointLocator()
        point_locator.SetDataSet(self.Ischemic)
        point_locator.BuildLocator()

        tolerance = 0.01
        dist2 = vtk.reference(0.0)
        for i in range(self.Territory.GetNumberOfPoints()):
            point = self.Territory.GetPoint(i)
            closest_point_id = point_locator.FindClosestPointWithinRadius(tolerance, point, dist2)
            if closest_point_id != -1:
                OverlapTag_Territory[i] = 1
                
        OverlapTagVTK = numpy_to_vtk(OverlapTag_Territory, deep=True)
        OverlapTagVTK.SetName("Overlap")
        self.Territory.GetPointData().AddArray(OverlapTagVTK)
        self.Territory.Modified()

        WriteVTUFile(self.OutputFileName1, self.Territory)

        return Mean_dist




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-Ischemic", type=str, required=True, dest="Ischemic")
    parser.add_argument("-Territory", type=str, required=True, dest="Territory", help="Territory Regions")
    #parser.add_argument("-LowerThreshold", type=float, required=True, dest="LowerThreshold")
    args = parser.parse_args()

    Spatial_Correspondance = ExtractSpatialCorrespondance(args).ExtractOverlap()
    print("Spatial Correspondance: ", Spatial_Correspondance)