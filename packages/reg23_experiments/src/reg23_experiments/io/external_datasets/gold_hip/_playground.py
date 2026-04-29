import SimpleITK as sitk


def main():
    img = sitk.ReadImage("archive_download/researchdata/data/3d_dataset/ct/ct_1mm.mhd")
    print(img)

    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    print("spacing:", spacing)
    print("origin:", origin)
    print("direction:", direction)

    arr = sitk.GetArrayFromImage(img)
    print(arr.shape)


if __name__ == "__main__":
    main()
