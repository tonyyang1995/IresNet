import nibabel as nib

template1 = nib.load('../AAL3v1_1mm.nii.gz')
template = template1.get_data()

a = nib.load('BA8.nii.gz')
a = a.get_data()

idx = (template == 152) # acc sup l
a[idx] = 0.555

s = nib.Nifti1Image(a, affine=template1.affine, header=template1.header)
nib.save(s, 'BA8_t3.nii.gz')
