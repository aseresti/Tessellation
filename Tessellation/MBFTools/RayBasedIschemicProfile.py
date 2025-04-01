import os
import glob
import vtk
import argparse
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utilities import ReadVTUFile, ReadVTPFile, WriteVTPFile, GetCentroid, ThresholdPointsByUpper, LargestConnectedRegion, PrintProgress

class RayBasedIschemicProfile():
    def __init__(self, Args):
        self.Args = Args
        slice_apex = f"{self.Args.InputFolder}/Slice_Apex.vtp"
        slice_base = f"{self.Args.InputFolder}/Slice_Base.vtp"
        
        SliceApex = ReadVTPFile(slice_apex)
        SliceBase = ReadVTPFile(slice_base)

        base_centeroid = GetCentroid(SliceBase)
        self.apex_centeroid = GetCentroid(SliceApex)

        AnnotationPoints = [base_centeroid, self.apex_centeroid]
        
        centerline_axis = np.array([AnnotationPoints[1][0] - AnnotationPoints[0][0], 
                                        AnnotationPoints[1][1] - AnnotationPoints[0][1], 
                                        AnnotationPoints[1][2] - AnnotationPoints[0][2]])
        
        self.CL_axis = centerline_axis/np.linalg.norm(centerline_axis)

        self.MBF = ReadVTUFile(self.Args.InputMBFBase)
        self.Apex = ReadVTUFile(self.Args.InputMBFApex)

        """
        Myocardium_centeroid = GetCentroid(self.MBF)
        point1 = np.array([Myocardium_centeroid[0] + self.CL_axis[0]*60, Myocardium_centeroid[1] + self.CL_axis[1]*60, Myocardium_centeroid[2] + self.CL_axis[2]*60])
        point2 = np.array([Myocardium_centeroid[0] - self.CL_axis[0]*50, Myocardium_centeroid[1] - self.CL_axis[1]*50, Myocardium_centeroid[2] - self.CL_axis[2]*50])
        """

        self.Ischemic = LargestConnectedRegion(ReadVTUFile(self.Args.InputIschemic))

        self.CenterLine = self.Line(AnnotationPoints[0], AnnotationPoints[1], self.Args.NSection)

        #Parameters:
        self.Radius = 35


    def Line(self, point1, point2, res):
        line = vtk.vtkLineSource()
        line.SetPoint1(point1)
        line.SetPoint2(point2)
        line.SetResolution(res)
        line.Update()
        
        return line.GetOutput()
    
    def BoldLine(self, centerline):
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(centerline)
        tube_filter.SetRadius(0.2)  # Adjust this value to change thickness
        tube_filter.SetNumberOfSides(50)  # Higher = smoother tube
        tube_filter.CappingOn()  # Close tube ends
        tube_filter.Update()
        
        return tube_filter.GetOutput()
    
    def SliceWPlane(self, Volume,Origin,Norm):
        plane=vtk.vtkPlane()
        plane.SetOrigin(Origin)
        plane.SetNormal(Norm)
        Slice=vtk.vtkCutter()
        Slice.GenerateTrianglesOff()
        Slice.SetCutFunction(plane)
        Slice.SetInputData(Volume)
        Slice.Update()
        
        return Slice.GetOutput()
    
    def ClipWPlane(self, surface, center, axis):
        plane = vtk.vtkPlane()
        plane.SetOrigin(center)
        plane.SetNormal(axis)

        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(surface)
        clipper.SetClipFunction(plane)
        clipper.InsideOutOff()
        #clipper.GetOutputInformation(1)
        clipper.Update()

        return clipper.GetOutput()
    
    def RotateVector(self, ray, normal, angle):
        angle = np.radians(angle)
        normal /= np.linalg.norm(normal)
        ray_rot = (ray*np.cos(angle) + np.cross(normal, ray) * np.sin(angle) + normal * np.dot(normal, ray) * (1 - np.cos(angle)))
        
        return ray_rot
    
    def ProbeFilter(self, InputData, SourceData):
        probe = vtk.vtkProbeFilter()
        probe.AddInputData(InputData)
        probe.SetSourceData(SourceData)
        probe.Update()
        
        return probe.GetOutput()
    
    def Cylinder(self):
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(self.CenterLine)
        tube_filter.SetRadius(self.Radius)
        tube_filter.SetNumberOfSides(2*self.Args.NRaySection)
        tube_filter.CappingOff()
        tube_filter.Update()
        
        return tube_filter.GetOutput()
    
    def Hemisphere(self):
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(self.apex_centeroid)
        sphere.SetRadius(self.Radius)
        sphere.SetPhiResolution(4*self.Args.NRaySphere)
        sphere.SetThetaResolution(4*self.Args.NRaySphere)
        sphere.Update()

        Hemisphere = self.ClipWPlane(sphere.GetOutput(), self.apex_centeroid, self.CL_axis)

        return Hemisphere
    
    def fibonacci_sphere(self, samples=100):
        """Generates evenly distributed unit vectors over a sphere"""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # Radius at given y
            theta = phi * i  # Golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])

        return np.array(points)
    
    def CastRaysVisualizations(self, CLPoints):
        
        #>>> Ray Casting across the Myocardium Cylinder 
        slices = vtk.vtkAppendPolyData()

        NRays_sample = 20
        ray_ = np.array([-self.CL_axis[2], 0, self.CL_axis[0]])
        res = 50
        angles = np.linspace(0, 360, NRays_sample, endpoint= False)
        Rays = vtk.vtkAppendPolyData()
        
        for k in range(0,len(CLPoints), int(len(CLPoints)/(self.Args.NSection/10))):
            Myo_slice = self.SliceWPlane(self.MBF, CLPoints[k], self.CL_axis)
            slices.AddInputData(Myo_slice)

            origin = CLPoints[k]
            slice_rays = vtk.vtkAppendPolyData()
            for i in range(NRays_sample):
                ray_new = self.RotateVector(ray_, self.CL_axis, angles[i])
                point2 = np.array([origin[0]+ 100*ray_new[0], origin[1] + 100*ray_new[1], origin[2] + 100*ray_new[2]])
                ray = self.Line(origin, point2, res)
                ray_projected = self.ProbeFilter(ray, self.Ischemic)
                
                ischemic_profile = vtk_to_numpy(ray_projected.GetPointData().GetArray("ImageScalars"))
                is_ischemic = ischemic_profile[ischemic_profile > 0]
                if is_ischemic.size > 0:
                    ischemic_profile = numpy_to_vtk(np.array([1 for _ in range(ray_projected.GetNumberOfPoints())]))
                else:
                    ischemic_profile = numpy_to_vtk(np.array([0 for _ in range(ray_projected.GetNumberOfPoints())]))

                ray_projected = self.ProbeFilter(ray_projected, self.MBF)
                
                ischemic_profile.SetName("IschemicProfile")
                ray_projected.GetPointData().AddArray(ischemic_profile)

                territory_profile = vtk_to_numpy(ray_projected.GetPointData().GetArray("TerritoryMaps"))
                territory = territory_profile[territory_profile > 0]
                
                if territory.size > 0:
                    territory_tag = np.bincount(territory).argmax()
                    territory_profile = numpy_to_vtk(np.array([territory_tag for _ in range(ray_projected.GetNumberOfPoints())]))
                else:
                    territory_profile = numpy_to_vtk(np.array([-1 for _ in range(ray_projected.GetNumberOfPoints())]))
                
                territory_profile.SetName("TerritoryProfile")
                ray_projected.GetPointData().AddArray(territory_profile)
                slice_rays.AddInputData(self.BoldLine(ray_projected))
        
            slice_rays.Update()
        
            Rays.AddInputData(slice_rays.GetOutput())


        #>>> Ray Casting across the Apex Hemisphere
        NRays_sample = 100
        directions = self.fibonacci_sphere(NRays_sample)
        origin = self.apex_centeroid
        Rays_ = vtk.vtkAppendPolyData()

        for i in range(NRays_sample):
            ray_new = directions[i]
            point2 = np.array([origin[0]+ 50*ray_new[0], origin[1] + 50*ray_new[1], origin[2] + 50*ray_new[2]])
            ray = self.Line(origin, point2, res)
            ray_projected = self.ProbeFilter(ray, self.Ischemic)
            
            ischemic_profile = vtk_to_numpy(ray_projected.GetPointData().GetArray("ImageScalars"))
            is_ischemic = ischemic_profile[ischemic_profile > 0]
            if is_ischemic.size > 0:
                ischemic_profile = numpy_to_vtk(np.array([1 for _ in range(ray_projected.GetNumberOfPoints())]))
            else:
                ischemic_profile = numpy_to_vtk(np.array([0 for _ in range(ray_projected.GetNumberOfPoints())]))

            ray_projected = self.ProbeFilter(ray_projected, self.Apex)
            
            ischemic_profile.SetName("IschemicProfile")
            ray_projected.GetPointData().AddArray(ischemic_profile)

            territory_profile = vtk_to_numpy(ray_projected.GetPointData().GetArray("TerritoryMaps"))
            territory = territory_profile[territory_profile > 0]
            
            if territory.size > 0:
                territory_tag = np.bincount(territory).argmax()
                territory_profile = numpy_to_vtk(np.array([territory_tag for _ in range(ray_projected.GetNumberOfPoints())]))
            else:
                territory_profile = numpy_to_vtk(np.array([-1 for _ in range(ray_projected.GetNumberOfPoints())]))
            
            territory_profile.SetName("TerritoryProfile")
            ray_projected.GetPointData().AddArray(territory_profile)
            Rays_.AddInputData(self.BoldLine(ray_projected))


        Rays_.Update()
        Rays_ = self.ClipWPlane(Rays_.GetOutput(), self.apex_centeroid, self.CL_axis)
        Rays.AddInputData(Rays_)
        Rays.Update()
        ray_path = f"{self.Args.InputFolder}/Rays.vtp"
        WriteVTPFile(ray_path, Rays.GetOutput())
    

        slices.Update()
        slice_path = f"{self.Args.InputFolder}/MyoSlices.vtp"
        WriteVTPFile(slice_path, slices.GetOutput())

    def RayCasting(self, CLPoints):

        NRays = self.Args.NRaySection
        ray_ = np.array([-self.CL_axis[2], 0, self.CL_axis[0]])
        res = 50
        angles = np.linspace(0, 360, NRays, endpoint= False)
        Rays = vtk.vtkAppendPolyData()
        
        print("------ Ray Casting across Sections along Myocardium:")
        progress_old = -1
        for k in range(0,len(CLPoints)):
            progress = PrintProgress(k,len(CLPoints),progress_old)
            progress_old = progress

            origin = CLPoints[k]
            slice_rays = vtk.vtkAppendPolyData()
            
            for i in range(NRays):
                ray_new = self.RotateVector(ray_, self.CL_axis, angles[i])
                point2 = np.array([origin[0]+ 100*ray_new[0], origin[1] + 100*ray_new[1], origin[2] + 100*ray_new[2]])
                ray = self.Line(origin, point2, res)
                ray_projected = self.ProbeFilter(ray, self.Ischemic)
                
                ischemic_profile = vtk_to_numpy(ray_projected.GetPointData().GetArray("ImageScalars"))
                is_ischemic = ischemic_profile[ischemic_profile > 0]
                if is_ischemic.size > 0:
                    ischemic_profile = numpy_to_vtk(np.array([1 for _ in range(ray_projected.GetNumberOfPoints())]))
                else:
                    ischemic_profile = numpy_to_vtk(np.array([0 for _ in range(ray_projected.GetNumberOfPoints())]))

                ray_projected = self.ProbeFilter(ray_projected, self.MBF)
                
                ischemic_profile.SetName("IschemicProfile")
                ray_projected.GetPointData().AddArray(ischemic_profile)

                territory_profile = vtk_to_numpy(ray_projected.GetPointData().GetArray("TerritoryMaps"))
                territory = territory_profile[territory_profile > 0]
                
                if territory.size > 0:
                    territory_tag = np.bincount(territory).argmax()
                    territory_profile = numpy_to_vtk(np.array([territory_tag for _ in range(ray_projected.GetNumberOfPoints())]))
                else:
                    territory_profile = numpy_to_vtk(np.array([-1 for _ in range(ray_projected.GetNumberOfPoints())]))
                
                territory_profile.SetName("TerritoryProfile")
                ray_projected.GetPointData().AddArray(territory_profile)
                slice_rays.AddInputData(self.BoldLine(ray_projected))
        
            slice_rays.Update()
        
            Rays.AddInputData(slice_rays.GetOutput())
            Rays.Update()

        directions = self.fibonacci_sphere(self.Args.NRaySphere)
        origin = self.apex_centeroid
        Rays_Hemisphere = vtk.vtkAppendPolyData()

        print("------ Ray Casting across the Apex Hemisphere:")
        progress_old = -1
        for i in range(self.Args.NRaySphere):
            progress = PrintProgress(i, self.Args.NRaySphere, progress_old)
            progress_old = progress

            ray_new = directions[i]
            point2 = np.array([origin[0]+ 50*ray_new[0], origin[1] + 50*ray_new[1], origin[2] + 50*ray_new[2]])
            ray = self.Line(origin, point2, res)
            ray_projected = self.ProbeFilter(ray, self.Ischemic)
            
            ischemic_profile = vtk_to_numpy(ray_projected.GetPointData().GetArray("ImageScalars"))
            is_ischemic = ischemic_profile[ischemic_profile > 0]
            if is_ischemic.size > 0:
                ischemic_profile = numpy_to_vtk(np.array([1 for _ in range(ray_projected.GetNumberOfPoints())]))
            else:
                ischemic_profile = numpy_to_vtk(np.array([0 for _ in range(ray_projected.GetNumberOfPoints())]))

            ray_projected = self.ProbeFilter(ray_projected, self.Apex)
            
            ischemic_profile.SetName("IschemicProfile")
            ray_projected.GetPointData().AddArray(ischemic_profile)

            territory_profile = vtk_to_numpy(ray_projected.GetPointData().GetArray("TerritoryMaps"))
            territory = territory_profile[territory_profile > 0]
            
            if territory.size > 0:
                territory_tag = np.bincount(territory).argmax()
                territory_profile = numpy_to_vtk(np.array([territory_tag for _ in range(ray_projected.GetNumberOfPoints())]))
            else:
                territory_profile = numpy_to_vtk(np.array([-1 for _ in range(ray_projected.GetNumberOfPoints())]))
            
            territory_profile.SetName("TerritoryProfile")
            ray_projected.GetPointData().AddArray(territory_profile)
            Rays_Hemisphere.AddInputData(self.BoldLine(ray_projected))


        Rays_Hemisphere.Update()
        #print("------ Clipping Sphere")
        #Rays_ = self.ClipWPlane(Rays_.GetOutput(), self.apex_centeroid, self.CL_axis)
        

        return Rays.GetOutput(), Rays_Hemisphere.GetOutput()

    def SpatialCorrespondance(self, Cylinder_Projected):
        with open(f"{self.Args.InputFolder}/MBF_Territories_Labels.dat",'r') as infile:
            infile.readline()
            TerritoryLabels=[]
            TerritoryNames = ""
            for LINE in infile:
                line=LINE.split()
                for tag in self.Args.TerritoryTag:
                    if line[1].find(tag)>=0: 
                        TerritoryLabels.append(int(line[0]))
                        TerritoryNames += tag + "_"
        print(TerritoryNames)

        ThresholdArray = np.zeros(Cylinder_Projected.GetNumberOfPoints())
        for i in range(Cylinder_Projected.GetNumberOfPoints()):
            if int(Cylinder_Projected.GetPointData().GetArray("TerritoryProfile").GetValue(i)) in TerritoryLabels:
                ThresholdArray[i] = 1
        
        ThresholdArrayVTK = numpy_to_vtk(ThresholdArray, deep=True)
        ThresholdArrayVTK.SetName(f"TerritoryLabels_{TerritoryNames}")
        Cylinder_Projected.GetPointData().AddArray(ThresholdArrayVTK)
        Cylinder_Projected.Modified()

        TerritoryRegions = ThresholdPointsByUpper(Cylinder_Projected, f"TerritoryLabels_{TerritoryNames}", 1)
        WriteVTPFile(self.Args.InputFolder + f"/Projected_Territories_{TerritoryNames}.vtp", TerritoryRegions)

        IschemicRegions = ThresholdPointsByUpper(Cylinder_Projected, "IschemicProfile", 1)
        ischemic_surface_path = os.path.splitext(os.path.basename(self.Args.InputIschemic))[0]
        WriteVTPFile(self.Args.InputFolder + f"/{ischemic_surface_path}.vtp", IschemicRegions)

        TerritoryRegionsPoints=vtk_to_numpy(TerritoryRegions.GetPoints().GetData())
        IschemicRegionsPoints=vtk_to_numpy(IschemicRegions.GetPoints().GetData())


        TerritoryTesselationString=[]
        TerritoryIschemicString=[]
        for i in range(TerritoryRegions.GetNumberOfPoints()): TerritoryTesselationString.append(str(TerritoryRegionsPoints[i]))
        for i in range(IschemicRegions.GetNumberOfPoints()): TerritoryIschemicString.append(str(IschemicRegionsPoints[i]))
        TotalPointsString=TerritoryTesselationString+TerritoryIschemicString
        UniquePointsString=set(TotalPointsString)
        OverlapCounter=len(TotalPointsString)-len(UniquePointsString)

        DiceScore = (2*OverlapCounter)/(TerritoryRegions.GetNumberOfPoints() + IschemicRegions.GetNumberOfPoints())
        print(DiceScore)
        return Cylinder_Projected

    
    def main(self):
        
        print("--- Extracting the Centerline of the Myocardium")
        cl_file = f"{self.Args.InputFolder}/CenterLine.vtp"
        WriteVTPFile(cl_file, self.BoldLine(self.CenterLine))

        print("--- Visualizing Ray Casting")
        CLPoints = self.CenterLine.GetPoints()
        CLPointsArray = np.array([CLPoints.GetPoint(i) for i in range(CLPoints.GetNumberOfPoints())])
        self.CastRaysVisualizations(CLPointsArray)

        print("--- Ray Casting Across the Myocardium")
        Rays_Myocardium, Rays_Hemisphere = self.RayCasting(CLPointsArray)
        
        print("--- Creating the Output Surface")
        Cylinder = self.Cylinder()
        Hemisphere = self.Hemisphere()

        print("--- Projecting the Rays onto the Output Surface")
        #OSurface_Projected = self.ProbeFilter(OutputSurface.GetOutput(), Rays)
        Cylinder_Projected = self.ProbeFilter(Cylinder, Rays_Myocardium)
        Hemisphere_Projected = self.ProbeFilter(Hemisphere, Rays_Hemisphere)

        print("--- Writing the Output Surface")
        OutputSurface = vtk.vtkAppendPolyData()
        OutputSurface.AddInputData(Cylinder_Projected)
        OutputSurface.AddInputData(Hemisphere_Projected)
        OutputSurface.Update()


        print("--- Calculating the Spatial Correspondeance")
        OSurface_Projected = self.SpatialCorrespondance(OutputSurface.GetOutput())

        output_path = f"{self.Args.InputFolder}/Output_Projected.vtp"
        WriteVTPFile(output_path, OSurface_Projected)







if __name__ == "__main__":

    Parser = argparse.ArgumentParser()
    Parser.add_argument("-InputMBFBase", "--InputMBFBase", required=True, dest="InputMBFBase", type=str)
    Parser.add_argument("-InputMBFApex", "--InputMBFApex", required=True, dest="InputMBFApex", type=str)
    Parser.add_argument("-InputIschemic", "--InputIschemic", required=True, dest="InputIschemic", type=str)
    Parser.add_argument("-InputFolder", "--InputFolder", required=True, dest="InputFolder", type=str)
    Parser.add_argument("-NRaySection", "--NRaySection", required=False, default=800, type=int, dest="NRaySection")
    Parser.add_argument("-NRaySphere", "--NRaySphere", required=False, default=5000, type=int, dest="NRaySphere")
    Parser.add_argument("-NSection", "--NSection", required=False, default= 60, type=int, dest="NSection")
    Parser.add_argument("-TerritoryTag", "--TerritoryTag", nargs='+', required=True, dest= "TerritoryTag", type=str)

    args = Parser.parse_args()

    RayBasedIschemicProfile(args).main()