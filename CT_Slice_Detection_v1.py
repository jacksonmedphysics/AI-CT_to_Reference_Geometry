import os
from tensorflow.keras.models import load_model
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os
from scipy import ndimage
import pandas as pd
from scipy.interpolate import interp1d

#predict list of reference z-axis locations associated with each image slice using 'get_zeds(im)'

#ct_dir_to_coords will read a directory (1 series per dir) and create a dataframe w/z-reference coordinates as well as centers-of-mass in x-y plane for each slice
#also returns ct as sitk image object

#register_cts will co-register CT images in two directories (uses function above to read in) by AI initial alignment and rigid refinement by sitk
#assisted by automated body mask from threshold above air
#option to provide PET/SPECT image which will be transformed in the process.
#also options to plot z-axis predictions coronal view slices at start, after AI alignment, and final refinement to quickly assess accuracy

def get_zeds(im): #'im' variable denotes sitk image object throughout
    #Runs Z-axis image recognition AI on each CT slice and returns 1D array of predicted locations on ref geometry
    model_name='all_datatrain_model_folded_256_v2_allcase_.100-14.456.hdf5'
    model = load_model(model_name, compile=False)
    input_shape=model.get_input_shape_at(0)
    vol=sitk.GetArrayFromImage(im)
    vol=ndimage.zoom(vol,(1,input_shape[1]/vol.shape[1],input_shape[2]/vol.shape[2]),order=1)
    ar=np.expand_dims(vol,-1)
    ar[ar<-200]=-200
    ar[ar>200]=200
    pred=model.predict(ar)
    del model
    ave=np.array(pred[:,0])
    return ave

def get_xy_coms(im):
    #Convert CT image into approx linear attenuation map to calculate X/Y centre-of-mass for each slice
    vol=sitk.GetArrayFromImage(im)
    vol[vol<-1000]=-1000 #clip values below -1000
    #to convert to linear attenuation coef
    #Ut=HU*Uw/1000+Uw
    vol=vol*0.206/1000+0.206
    x=[]
    y=[]
    for i in range(vol.shape[0]):
        com=ndimage.center_of_mass(vol[i,...])
        x.append(com[1])
        y.append(com[0])
    return x,y

def ct_dir_to_coords(ct_dir):
    #loads CT and creates dataframe with coordinate information, X&Y centres-of-mass, Z according to ref geometry
    reader=sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(ct_dir))
    ct=reader.Execute()
    origin=ct.GetOrigin()
    spacing=ct.GetSpacing()
    #sitk.WriteImage(ct,ct_dir+'.nii.gz')
    df=pd.DataFrame(columns=['orig_x','orig_y','orig_z','com_x','com_y','ref_z'])
    x,y=get_xy_coms(ct)
    z_locs=get_zeds(ct)
    for i in range(len(z_locs)):
        orig_x=origin[0]
        orig_y=origin[1]
        orig_z=origin[2]+i*spacing[2]
        com_x=x[i]*spacing[0]+origin[0]
        com_y=y[i]*spacing[1]+origin[1]
        ref_z=z_locs[i]
        row=[orig_x,orig_y,orig_z,com_x,com_y,ref_z]
        df.loc[i]=row
    return ct,df

def match_coord_dfs(fixed_df,moving_df,moving_ct,plot_z_locations=False):
    min_val=max([min(fixed_df.ref_z.values),min(moving_df.ref_z.values)])
    max_val=min([max(fixed_df.ref_z.values),max(moving_df.ref_z.values)])
    z_range=np.arange(min_val,max_val,1)
    f_fixed=interp1d(fixed_df.ref_z.values,fixed_df.orig_z.values)
    f_moving=interp1d(moving_df.ref_z.values,moving_df.orig_z.values)
    z_fixed=f_fixed(z_range)
    z_moving=f_moving(z_range)
    z_shift=np.average(z_fixed-z_moving)
    f_fixed=interp1d(fixed_df.ref_z.values,fixed_df.com_x.values)
    f_moving=interp1d(moving_df.ref_z.values,moving_df.com_x.values)
    x_fixed=f_fixed(z_range)
    x_moving=f_moving(z_range)
    x_shift=np.average(x_fixed-x_moving)
    f_fixed=interp1d(fixed_df.ref_z.values,fixed_df.com_y.values)
    f_moving=interp1d(moving_df.ref_z.values,moving_df.com_y.values)
    y_fixed=f_fixed(z_range)
    y_moving=f_moving(z_range)
    y_shift=np.average(y_fixed-y_moving)
    print('Translation:',x_shift,y_shift,z_shift)
    if plot_z_locations:
        plt.figure(figsize=[10,8])
        plt.plot(fixed_df.orig_z.values,fixed_df.ref_z.values,label='Fixed Image')
        plt.plot(moving_df.orig_z.values,moving_df.ref_z.values,label='Moving Image')
        plt.scatter((z_moving+z_shift),z_range,label='Moving Image post-alignment (overlapping region)',s=0.5,c='r')
        plt.title('Z-axis Location Detection and Shift: '+str(round(z_shift,2))+' mm')
        plt.xlabel('DICOM Coordinate (mm)')
        plt.ylabel('Reference Geometry Value (mm)')
        plt.legend()
        plt.grid()
        plt.show()
    return x_shift,y_shift,z_shift

def mask_image(image,mask_range=[-980,3000]):
    ar=sitk.GetArrayFromImage(image)
    mask=np.logical_and((ar>mask_range[0]),(ar<mask_range[1]))
    mask_im=sitk.GetImageFromArray(mask.astype('uint8'))
    mask_im.SetSpacing(image.GetSpacing())
    mask_im.SetOrigin(image.GetOrigin())
    mask_im.SetDirection(image.GetDirection())
    mask_im=sitk.Cast(mask_im,sitk.sitkInt8)
    return mask_im

def register_cts(fixed_ct_path,moving_ct_path,moving_pet_path=None,plot_z_locations=False,plot_alignment=False):
    #Runs on fixed and moving ct paths, if pet path is provided will return transformed PET as well
    #flags plot reference z-axis coordinates and shift as well as coronal slice to confirm reg accuracy, both default to False
    fixed_ct,fixed_df=ct_dir_to_coords(fixed_ct_path)
    moving_ct,moving_df=ct_dir_to_coords(moving_ct_path)
    fixed_mask=mask_image(fixed_ct,[-970,4000]) #HU range to include in mask
    moving_mask=mask_image(moving_ct,[-970,4000])
    print('Aligning by reference geometry...')
    x_shift,y_shift,z_shift=match_coord_dfs(fixed_df,moving_df,moving_ct,plot_z_locations=plot_z_locations)
    tx=sitk.Euler3DTransform()
    tx.SetParameters((0.,0.,0.,-x_shift,-y_shift,-z_shift))
    ai_ct=sitk.Resample(moving_ct,fixed_ct,tx,sitk.sitkLinear,-1000,sitk.sitkFloat32)
    R=sitk.ImageRegistrationMethod() #set image registration method parameters
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(3.0,.001,200)
    R.SetOptimizerScalesFromIndexShift()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetMetricFixedMask(fixed_mask)
    R.SetMetricMovingMask(moving_mask)
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    fixed_ct=sitk.Cast(fixed_ct,sitk.sitkFloat32) #need to be floats for registration
    moving_ct=sitk.Cast(moving_ct,sitk.sitkFloat32)
    print('Performing Rigid Registration')
    transform=R.Execute(fixed_ct,moving_ct) #calculate registration...
    print(transform.GetParameters())
    moving_ct_rs=sitk.Resample(moving_ct,fixed_ct,transform, sitk.sitkLinear,-1000,sitk.sitkFloat32)
    if plot_alignment:
        plot_coronal(fixed_ct,moving_ct,ai_ct,moving_ct_rs)
    if moving_pet_path: #check if PET path has been declared
        reader=sitk.ImageSeriesReader()
        reader.SetFileNames(reader.GetGDCMSeriesFileNames(moving_pet_path))
        moving_pet=reader.Execute()
        moving_pet_rs=sitk.Resample(moving_pet,fixed_ct,transform,sitk.sitkLinear,0,sitk.sitkFloat32)
        return fixed_ct,ai_ct,moving_ct_rs, moving_pet_rs
    else:
        return fixed_ct,ai_ct,moving_ct_rs

def plot_coronal(fixed_ct,moving_ct,moving_ct_ai,moving_ct_reg):
    rs=sitk.ResampleImageFilter()
    rs.SetReferenceImage(fixed_ct)
    rs.SetDefaultPixelValue(-1000)
    moving_ct=rs.Execute(moving_ct)
    ar_moving=sitk.GetArrayFromImage(moving_ct)
    ar_fixed=sitk.GetArrayFromImage(fixed_ct)
    ar_ai=sitk.GetArrayFromImage(moving_ct_ai)
    ar_reg=sitk.GetArrayFromImage(moving_ct_reg)
    sp=fixed_ct.GetSpacing()
    aspect=sp[2]/sp[0]
    diff_moving=ar_fixed-ar_moving
    diff_ai=ar_fixed-ar_ai
    diff_reg=ar_fixed-ar_reg
    center_y=int(ar_fixed.shape[1]/2)
    plt.figure(figsize=[12,6])
    plt.subplot(131)
    plt.imshow(np.flipud(diff_moving[:,center_y,:]),aspect=aspect,cmap='gray')
    plt.axis('off')
    plt.title('Original Image Locations')    
    plt.subplot(132)
    plt.imshow(np.flipud(diff_ai[:,center_y,:]),aspect=aspect,cmap='gray')
    plt.axis('off')
    plt.title('Initial AI Alignment')
    plt.subplot(133)
    plt.imshow(np.flipud(diff_reg[:,center_y,:]),aspect=aspect,cmap='gray')
    plt.title('SITK Rigid Refinement')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return
