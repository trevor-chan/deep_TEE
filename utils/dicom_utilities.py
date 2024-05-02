import matplotlib.pyplot as plt
from pydicom import dcmread
import pylibjpeg
from pydicom.tag import Tag
from pydicom.pixel_data_handlers import convert_color_space


modifiedTags = [
  {'tagLoc':'0x00100010','tagValue':'nameAnonymized'},
  {'tagLoc':'0x00100020','tagValue':'mrnAnonymized'},
  {'tagLoc':'0x00100030','tagValue':'19710101'},
  {'tagLoc':'0x00081048','tagValue':'PhysicianOfRecord'},
  {'tagLoc':'0x00081050','tagValue':'PerformingPhysician'},
  {'tagLoc':'0x00321032','tagValue':'RequestingPhysician'},
  {'tagLoc':'0x00101000','tagValue':'PatientID'},
  {'tagLoc':'0x00080090','tagValue':'Physician'},
  {'tagLoc':'0x00102160','tagValue':'Ethnicity'},
  {'tagLoc':'0x00080080','tagValue':'InstitutionName'},
  {'tagLoc':'0x00080020','tagValue':'19710101'},
  {'tagLoc':'0x00080021','tagValue':'19710101'},
  {'tagLoc':'0x00080022','tagValue':'19710101'},
  {'tagLoc':'0x00080023','tagValue':'19710101'},
  {'tagLoc':'0x00400244','tagValue':'19710101'}
]

acqdatetimeTags = [
  {'tagLoc':'0x0008002A','tagValue':'19710101'},
  {'tagLoc':'0x00400253','tagValue':'19710101'}
]

def deidentify (dicom):					
    for tag in modifiedTags:
        try:
            if Tag(tag['tagLoc']) in dicom:
                dicom[tag['tagLoc']].value = tag['tagValue']
                # print(f"Modified tag {tag['tagLoc']} to {tag['tagValue']}")
        except Exception as e:
            print(e)

    for tag in acqdatetimeTags:
        try:
            if Tag(tag['tagLoc']) in dicom:
                tag_acqdatetime = dicom[tag['tagLoc']].value
                newtag_acqdatetime = tag['tagValue'] + tag_acqdatetime[8:]
                dicom[tag['tagLoc']].value = newtag_acqdatetime
                # print(f"Modified tag {tag['tagLoc']} to {newtag_acqdatetime}")
        except Exception as e:
            print(e)
  
    return dicom

def read_dicom (filename):
    # return deidentify(dcmread(filename))
    return dcmread(filename)

def get_dicom_image_data (dicom):
    rgb = convert_color_space(dicom.pixel_array, "YBR_FULL", "RGB", per_frame=True)
    return rgb

def save_fig (filename, pixel_array):
    w = 8
    h = 6
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # ax.imshow(pixel_array, cmap=plt.cm.gray)
    ax.imshow(pixel_array) #save color image
    plt.savefig(filename, pad_inches=0, bbox_inches=None, dpi=pixel_array.shape[1] / w)
    plt.close(fig)