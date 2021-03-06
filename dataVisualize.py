import vtk

def visualize_3d(reader):
    """visualize 3d volume using vtk volume rendering

    Arguments:
        reader : vtk imageReader
        in our case, specially the vtkMetaImageReader

    Returns:
        just display the visualized volume no returns
    """
    # The volume will be displayed by ray-cast alpha compositing.
    # A ray-cast mapper is needed to do the ray-casting, and a
    # compositing function is needed to do the compositing along the ray.
    volumeMapper = vtk.vtkGPUVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    volumeMapper.SetBlendModeToComposite()

    # The opacity transfer function is used to control the opacity
    # of different tissue types.
    volumeScalarOpacity = vtk.vtkPiecewiseFunction()
    volumeScalarOpacity.AddPoint(0, 0.00)
    volumeScalarOpacity.AddPoint(500, 0.15)
    volumeScalarOpacity.AddPoint(1000, 0.15)
    volumeScalarOpacity.AddPoint(1150, 0.85)

    # The color transfer function maps voxel intensities to colors.
    # It is modality-specific, and often anatomy-specific as well.
     # The goal is to one color for flesh (between 500 and 1000)
    # and another color for bone (1150 and over).
    # volumeColor = vtk.vtkColorTransferFunction()
    # volumeColor.AddRGBPoint(0,    0.0, 0.0, 0.0)
    # volumeColor.AddRGBPoint(500,  1.0, 0.5, 0.3)
    # volumeColor.AddRGBPoint(1000, 1.0, 0.8, 0.5)
    # volumeColor.AddRGBPoint(1150, 1.0, 1.0, 0.9)
    # The gradient opacity function is used to decrease the opacity
    # in the "flat" regions of the volume while maintaining the opacity
    # at the boundaries between tissue types.  The gradient is measured
    # as the amount by which the intensity changes over unit distance.
     # For most medical data, the unit distance is 1mm.
    volumeGradientOpacity = vtk.vtkPiecewiseFunction()
    volumeGradientOpacity.AddPoint(0,   0.0)
    volumeGradientOpacity.AddPoint(90,  0.5)
    volumeGradientOpacity.AddPoint(100, 1.0)

    # The VolumeProperty attaches the color and opacity functions to the
    # volume, and sets other volume properties.  The interpolation should
    # be set to linear to do a high-quality rendering.  The ShadeOn option
    # turns on directional lighting, which will usually enhance the
    # appearance of the volume and make it look more "3D".  However,
    # the quality of the shading depends on how accurately the gradient
    # of the volume can be calculated, and for noisy data the gradient
    # estimation will be very poor.  The impact of the shading can be
    # decreased by increasing the Ambient coefficient while decreasing
    # the Diffuse and Specular coefficient.  To increase the impact
    # of shading, decrease the Ambient and increase the Diffuse and Specular.
    volumeProperty = vtk.vtkVolumeProperty()
    # volumeProperty.SetColor(volumeColor)
    volumeProperty.SetScalarOpacity(volumeScalarOpacity)
    volumeProperty.SetGradientOpacity(volumeGradientOpacity)
    # print(volumeProperty.GetScalarOpacity(), volumeProperty.GetGradientOpacity())
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.4)
    volumeProperty.SetDiffuse(0.6)
    volumeProperty.SetSpecular(0.2)

    # Create the renderer, the render window, and the interactor. The renderer
     # draws into the render window, the interactor enables mouse- and
    # keyboard-based interaction with the scene.

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    # The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
    # and orientation of the volume in world coordinates.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    # Finally, add the volume to the renderer
    ren.AddViewProp(volume)
    # Set up an initial view of the volume.  The focal point will be the
    # center of the volume, and the camera position will be 400mm to the
    # patient's left (whis is our right).
    camera = ren.GetActiveCamera()
    c = volume.GetCenter()
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.SetPosition(c[0] + 400, c[1], c[2])
    camera.SetViewUp(0, 0, -1)

    # Increase the size of the render window
    renWin.SetSize(640, 480)
    # Interact with the data.
    iren.Initialize()
    renWin.Render()
    iren.Start()


def vtkImageReader(file, segmentation=False):
    """
    using vtkMetaImageReader to read mhd file
    :param file: the path of the mhd file (note: using '/' instead of '\' in absolute path)
    :param segmentation: assign true when file is partial volume segmentation
    :return: updated reader
    """
    image = vtk.vtkMetaImageReader()
    image.SetFileName(file)
    image.Update()
    print(image.GetOutput())
    if segmentation:
        # turn datatype to ushort
        # and increase the voxel value by multiply a constant if it is a partial volume segmentation file
        cast = vtk.vtkImageCast()
        cast.SetInputConnection(image.GetOutputPort())
        cast.SetOutputScalarTypeToUnsignedShort()
        cast.Update()

        math = vtk.vtkImageMathematics()
        math.SetOperationToMultiplyByK()
        math.SetConstantK(1150.0)
        math.SetInputConnection(cast.GetOutputPort())
        math.Update()
        return math
    else:
        return image


def display_substract(file1,file2):
    source1 = vtk.vtkMetaImageReader()
    source1.SetFileName(file1)
    source1.Update()
    source2 = vtk.vtkMetaImageReader()
    source2.SetFileName(file2)
    source2.Update()
    substract = vtk.vtkImageMathematics()
    substract.SetOperationToSubtract()
    substract.SetInput1Data(source1.GetOutput())
    substract.SetInput2Data(source2.GetOutput())
    substract.Update()
    visualize_3d(substract)


if __name__ == "__main__":
    path1 = 'E:/ctaDatasets/train/original/challenge000/cta000l.mhd'
    path2 = 'E:/ctaDatasets/train/original/challenge100/cta100r.mhd'
    # sitk_read(file_train)
    reader_train = vtkImageReader(path1)
    visualize_3d(reader_train)
    # reader2 = vtkImageReader(path2)
    # visualize_3d(reader2)


    # print(mx_train[45:50,105:110,76:80])
    # reader_test = vtkImageReader(file_test)
    # visualize_3d(reader_test)
    # mx_test = image2matrix(reader_test)
    # print(mx_test[45:50,105:110,76:80])
    # print( np.max(mx_test))
