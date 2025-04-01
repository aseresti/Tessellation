import SimpleITK as sitk
import argparse

def Convertnrrd2NIFTI(ipath, opath):
    # Read the NRRD image
    nrrd_image = sitk.ReadImage(ipath)
    
    # Save the image as NIfTI
    sitk.WriteImage(nrrd_image, opath)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-InputFolder", "--InputFolder", type=str, required=True, dest="InputFolder")
    parser.add_argument("-OutputFolder", "--OutPutFolder", type=str, required=True, dest="OutputFolder")
    args = parser.parse_args()

    Convertnrrd2NIFTI(args.InputFolder, args.OutputFolder)