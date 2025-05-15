import os
import sys
import glob
import vtk
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from utilities import ReadVTPFile, WriteVTPFile, GetCentroid, ThresholdByUpper, ExtractSurface, PrintProgress

TessellationPath = os.path.abspath(f"{sys.path[0]}/..")
sys.path.append(TessellationPath)

from SimVascularTools.ConvertPathtoVTP import ConvertPath2VTP

class CreatePolarPlot():
    def __init__(self, args):
        slice_base = os.path.join(args.InputFolder, args.SliceBase)
        slice_apex = os.path.join(args.InputFolder, args.SliceApex)
        myocardium = os.path.join(args.InputFolder, args.Myocardium)
        pathfolder = os.path.join(args.InputFolder, args.PathFolder)
        OutputFolder = os.path.join(args.InputFolder, "PolarMap")
        InputFiles = glob.glob(f"{args.InputFolder}/*")

        if slice_base in InputFiles:
            self.centeroid_base = GetCentroid(ReadVTPFile(slice_base))
        else:
            print(f"{args.SliceBase} Not Found in the InputFolder")

        if slice_apex in InputFiles:
            self.centeroid_apex = GetCentroid(ReadVTPFile(slice_apex))
        else:
            print(f"{args.SliceApex} Not Found in the InputFolder")

        if myocardium in InputFiles:
            self.Myocardium = ReadVTPFile(myocardium)
        else:
            print(f"{args.Myocardium} Not Found in the InputFolder")

        if pathfolder in InputFiles:
            self.PathFolder = pathfolder
        else:
            print(f"{args.PathFolder} directory Not Found in the InputFolder")

        if OutputFolder in InputFiles:
            self.OutputFolder = OutputFolder
        else:
            os.system(f"mkdir {OutputFolder}")
            self.OutputFolder = OutputFolder

        self.R_max = args.PlotRadius

    def Line(self, point1, point2, res):
        line = vtk.vtkLineSource()
        line.SetPoint1(point1)
        line.SetPoint2(point2)
        line.SetResolution(res)
        line.Update()

        return line.GetOutput()
    
    def BoldLine(self, line):
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(line)
        tube_filter.SetRadius(0.05)
        tube_filter.SetNumberOfSides(50) 
        tube_filter.CappingOn()
        tube_filter.Update()
        
        return tube_filter.GetOutput()
    
    def SliceWPlane(self, Volume, Origin, Norm):
        plane=vtk.vtkPlane()
        plane.SetOrigin(Origin)
        plane.SetNormal(Norm)
        Slice=vtk.vtkCutter()
        Slice.GenerateTrianglesOff()
        Slice.SetCutFunction(plane)
        Slice.SetInputData(Volume)
        Slice.Update()
        
        return Slice.GetOutput()
    
    def DefineMyocardiumCenterLine(self, centeroid_base, centeroid_apex, resolution):
        CL_dir = np.array([
            centeroid_base[0] - centeroid_apex[0],
            centeroid_base[1] - centeroid_apex[1],
            centeroid_base[2] - centeroid_apex[2],
        ])
        CL_direction = CL_dir/np.linalg.norm(CL_dir)

        point0 = np.array([
            centeroid_base[0] + CL_direction[0]*5,
            centeroid_base[1] + CL_direction[1]*5,
            centeroid_base[2] + CL_direction[2]*5
        ])

        point1 = np.array([
            centeroid_apex[0] - CL_direction[0]*1.5,
            centeroid_apex[1] - CL_direction[1]*1.5,
            centeroid_apex[2] - CL_direction[2]*1.5
        ])

        CenterLine = self.Line(point0, point1, resolution)
        WriteVTPFile(os.path.join(self.OutputFolder, "MyocardiumCenterLine.vtp"), self.BoldLine(CenterLine))

        return CL_direction, CenterLine
    
    def CopyProfileToPolyData(self, old_polydata, Array, new_coords, ArrayNameDestination):
        new_points = vtk.vtkPoints()
        for pt in new_coords:
            new_points.InsertNextPoint(pt)

        new_polydata = vtk.vtkPolyData()
        new_polydata.SetPoints(new_points)
        ProfileCopy = vtk.vtkFloatArray()
        ProfileCopy.SetName(ArrayNameDestination)
        ProfileCopy.SetNumberOfComponents(1)
        ProfileCopy.SetNumberOfTuples(old_polydata.GetNumberOfPoints())

        for j in range(old_polydata.GetNumberOfPoints()):
            ProfileCopy.SetTuple(j, Array.GetTuple(j))

        new_polydata.GetPointData().AddArray(ProfileCopy)

        return new_polydata

    def MeshPolyData(self, polydata):
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(polydata)
        delaunay.Update()

        return delaunay.GetOutput()
    
    def ConvertSurfaceToVolume(self, Surface):
        AppendFilter = vtk.vtkAppendFilter()
        AppendFilter.AddInputData(Surface)
        AppendFilter.Update()

        return AppendFilter.GetOutput()
    
    def CalculateAngleBetweenCentroids(self, IschemicRegionMap, VesselTerritoryMap):
        CenteroidIschemic = GetCentroid(IschemicRegionMap)
        CenteroidVesselTerritory = GetCentroid(VesselTerritoryMap)

        Centeroid_CircularMap = [0, 0, 0]

        Line1 = self.Line(Centeroid_CircularMap, CenteroidIschemic, 10)
        Line2 = self.Line(Centeroid_CircularMap, CenteroidVesselTerritory, 10)

        WriteVTPFile(os.path.join(self.OutputFolder, "LineIschemicCenter.vtp"), self.BoldLine(Line1))
        WriteVTPFile(os.path.join(self.OutputFolder, "LineTerritoryCenter.vtp"), self.BoldLine(Line2))

        direction1 = [CenteroidIschemic[0] - Centeroid_CircularMap[0],
                    CenteroidIschemic[1] - Centeroid_CircularMap[1],
                    CenteroidIschemic[2] - Centeroid_CircularMap[2]
                    ]

        direction1 /= np.linalg.norm(direction1)

        direction2 = [CenteroidVesselTerritory[0] - Centeroid_CircularMap[0],
                    CenteroidVesselTerritory[1] - Centeroid_CircularMap[1],
                    CenteroidVesselTerritory[2] - Centeroid_CircularMap[2]
                    ]

        direction2 /= np.linalg.norm(direction1)

        dot_product = np.clip(np.dot(direction1, direction2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_degree = np.degrees(angle_rad)

        return angle_degree

    def FindClosestPointToArray(self, array, point):
        point = np.array(point)
        array = np.asarray(array)
        differences = np.linalg.norm(array - point, axis=1)
        idx = np.argmin(differences)
        
        return idx
    
    def ReorderCoronaryMapBasedOnCenterLine(self, arbitrary_array, VesselCenterLine, new_points):
        idx_array = []
        for point in arbitrary_array:
            idx_array.append(self.FindClosestPointToArray(VesselCenterLine, point))
        sorted_idx = np.argsort(np.array(idx_array))
        new_points = np.array(new_points)
        
        return new_points[sorted_idx]
    
    def BullsEye(self):
        CL_direction, CenterLine = self.DefineMyocardiumCenterLine(self.centeroid_base, self.centeroid_apex, 1000)
        rotation, _ = R.align_vectors([[0, 0, 1]], [CL_direction])

        print("- Mapping Myocardium onto Polar Plot")
        Npoints  = CenterLine.GetNumberOfPoints()
        R_map = [i * self.R_max/Npoints for i in range(Npoints, 0, -1)]

        for i in range(self.Myocardium.GetPointData().GetNumberOfArrays()):
            array_name_ = self.Myocardium.GetPointData().GetArrayName(i)
            if "Ischemic" in array_name_:
                IschemicArrayName = array_name_
                IschemicArrayNumber = i
            elif "TerritoryLabels" in array_name_:
                TerritoryArrayName = "SelectedTerritory"
                TerritoryArrayNumber = i
            else:
                continue

        append_filter_ischemic = vtk.vtkAppendPolyData()
        append_filter_territory = vtk.vtkAppendPolyData()
        Center = []
        progress_ = 0
        for i in range(Npoints):
            progress_ = PrintProgress(i, Npoints, progress_)
            
            point = CenterLine.GetPoint(i)
            slice_ = self.SliceWPlane(self.Myocardium, point, CL_direction)
            if slice_.GetNumberOfPoints() == 0:
                Center.append(point)
                continue

            IschemicProfile = slice_.GetPointData().GetArray(IschemicArrayNumber)
            TerritoryProfile = slice_.GetPointData().GetArray(TerritoryArrayNumber)
            pts_np = np.array([slice_.GetPoint(j) for j in range(slice_.GetNumberOfPoints())])
            center = pts_np.mean(axis=0)
            Center.append(center)
            new_coords = []
            for j in range(slice_.GetNumberOfPoints()):
                coord_ = slice_.GetPoint(j)
                aligned_coord_ = rotation.apply(coord_ - center)
                angle = np.arctan2(aligned_coord_[1], aligned_coord_[0])
                new_coords.append([R_map[i]*np.cos(angle), R_map[i]*np.sin(angle), 0])
            
            append_filter_ischemic.AddInputData(self.CopyProfileToPolyData(slice_, IschemicProfile, new_coords, IschemicArrayName))
            append_filter_territory.AddInputData(self.CopyProfileToPolyData(slice_, TerritoryProfile, new_coords, TerritoryArrayName))
        append_filter_ischemic.Update()
        append_filter_territory.Update()

        print("- Writing Territory and Ischemic Maps and Regions")
        IschemicMap = self.MeshPolyData(append_filter_ischemic.GetOutput())
        TerritoryMap = self.MeshPolyData(append_filter_territory.GetOutput())

        WriteVTPFile(os.path.join(self.OutputFolder, "MyocardiumMap_Ischemic.vtp"), IschemicMap)
        WriteVTPFile(os.path.join(self.OutputFolder, "MyocardiumMap_Territory.vtp"), TerritoryMap)

        IschemicRegionMap = ThresholdByUpper(self.ConvertSurfaceToVolume(IschemicMap), IschemicArrayName, 1)
        TerritoryRegionMap = ThresholdByUpper(self.ConvertSurfaceToVolume(TerritoryMap), TerritoryArrayName, 1)

        WriteVTPFile(os.path.join(self.OutputFolder, "MyocardiumMap_IschemicRegion.vtp"), ExtractSurface(IschemicRegionMap))
        WriteVTPFile(os.path.join(self.OutputFolder, "MyocardiumMap_TerritoryRegion.vtp"), ExtractSurface(TerritoryRegionMap))

        AngleBetweenCenters = self.CalculateAngleBetweenCentroids(IschemicRegionMap, TerritoryRegionMap)
        print("Calculating the angle between centroids of ischemic and vessel territory regions:")
        print(f"\t{AngleBetweenCenters} degrees")
        outputfile = os.path.join(self.OutputFolder, "Angle.dat")
        with open(outputfile, 'w') as ofile:
            ofile.writelines(f"The angle between centroids of ischemic and vessel territory regions = {AngleBetweenCenters}$degrees$")
        
        print("- Mapping Coronaries Into Polar Plot:")
        args = argparse.Namespace()
        args.InputFolder = self.PathFolder
        Path2Point = ConvertPath2VTP(args)
        for path in Path2Point.filenames:
            VesselCenterline, _ = Path2Point.pth_to_points(path)
            file_ = Path2Point.points_to_vtp(VesselCenterline)
            new_points = []
            arbitrary_points = []
            for i in range(Npoints):
                point = CenterLine.GetPoint(i)
                slice_ = self.SliceWPlane(file_, point, CL_direction)
            
                if slice_.GetNumberOfPoints() == 0:
                    continue
                for j in range(slice_.GetNumberOfPoints()):
                    arbitrary_points.append(slice_.GetPoint(j))
                    centered_point = [
                        slice_.GetPoint(j)[0] - Center[i][0],
                        slice_.GetPoint(j)[1] - Center[i][1],
                        slice_.GetPoint(j)[2] - Center[i][2]
                    ]
                    alignPoints = rotation.apply(centered_point)
                    angle = np.arctan2(alignPoints[1], alignPoints[0])
                    new_points.append([R_map[i]*np.cos(angle), R_map[i]*np.sin(angle), 0])

            
            new_points_sorted = self.ReorderCoronaryMapBasedOnCenterLine(arbitrary_points, VesselCenterline, new_points)
            vessel_polydata = Path2Point.points_to_vtp(new_points_sorted)
            VesselName = os.path.splitext(os.path.basename(path))[0]
            print(f"--- Converting {VesselName}")
            Vesseldir = os.path.join(self.OutputFolder, f"CoronaryMap_{VesselName}.vtp")
            WriteVTPFile(Vesseldir, self.BoldLine(vessel_polydata))




if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument("-InputFolder", "--InputFolder", dest= "InputFolder", required=True, type=str)
    Parser.add_argument("-SliceApex", "--SliceApex", dest= "SliceApex", required=False, type=str, default="SliceApex.vtp")
    Parser.add_argument("-SliceBase", "--SliceBase", dest= "SliceBase", required=False, type=str, default="SliceBase.vtp")
    Parser.add_argument("-Myocardium", "--Myocardium", dest= "Myocardium", required=False, type=str, default="MyocardiumSurface.vtp")
    Parser.add_argument("-PathFolder", "--PathFolder", dest= "PathFolder", required= False, type= str, default="Paths")
    Parser.add_argument("-PlotRadius", "--PlotRadius", dest= "PlotRadius", default=12.0, type= float, required= False)
    args = Parser.parse_args()

    CreatePolarPlot(args).BullsEye()