from PIL import Image
from streamlit_drawable_canvas import st_canvas
import base64
import streamlit as strlt
import pathlib
import numpy as np
import cv2
import io

#Within the streamlit static asset directory, we make a downloads directory and publish output files to it.
STRLT_STATIC_PATH = pathlib.Path(strlt.__path__[0]) / 'static'
PATH_FOR_DOWNLOAD = (STRLT_STATIC_PATH / "downloads")

if PATH_FOR_DOWNLOAD.is_dir() is not True:
    PATH_FOR_DOWNLOAD.mkdir()
else:
    print('dont create a folder')



# Once we get the corner points for the documents, you only need the destination coordinates to perform perspective
# transform and align the documents. 
def find_dest(points):
    (tl, tr, br, bl) = points
    # Find the maximum height. 
    heightA = np.sqrt(((tr[0] - br[0]) * (tr[0] - br[0])) + ((tr[1] - br[1]) * (tr[1] - br[1])))
    heightB = np.sqrt(((tl[0] - bl[0]) * (tl[0] - bl[0])) + ((tl[1] - bl[1]) * (tl[1] - bl[1])))
    intheightA = int(heightA)
    intheightB = int(heightB)
    maximumHeight = max(intheightA, intheightB)
    # Find the maximum width. 
    widthA = np.sqrt(((br[0] - bl[0]) * (br[0] - bl[0])) + ((br[1] - bl[1]) *(br[1] - bl[1])))
    widthB = np.sqrt(((tr[0] - tl[0]) * (tr[0] - tl[0])) + ((tr[1] - tl[1]) * (tr[1] - tl[1])))
    intwidthA = int(widthA)
    intwidthB = int(widthB)
    maximumWidth = max(int(intwidthA), int(intwidthB))
    # Final destination co-ordinates.
    dest_corners = [[0, 0], [maximumWidth, 0], [maximumWidth, maximumHeight], [0, maximumHeight]]

    return order_points(dest_corners)


def order_points(points):
   #This function is for making a conventional order of points 
   #    starting from top-left to bottom-left(top-left, top-right, bottom-right, bottom-left)
    rectangular = np.zeros((4, 2), dtype='float32')
    points = np.array(points)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    smallest = np.argmin(s)
    largest= np.argmax(s)
    smallest_diff = np.argmin(diff)
    largest_diff = np.argmax(diff)
    
    # Top-left
    rectangular[0] = points[smallest]
    # Top-right
    rectangular[1] = points[smallest_diff]
    # Bottom-right
    rectangular[2] = points[largest]
    # Bottom-left
    rectangular[3] = points[largest_diff]
    
    ordered = rectangular.astype('int')
    # ordered coordinates
    return ordered.tolist()

def scan(img):
    # Resize the image to dim_limit size
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    orig_img = img.copy()
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    msk = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, msk, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((msk == 2) | (msk == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    
    # Blurring and turning to grey
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
   
    edge = cv2.Canny(gray, 0, 200)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, level = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

  
    if len(page) == 0:
        return orig_img
    # Using the Douglas-Peucker algorithm, this function approximates a curve or a polygon with another curve/polygon, with fewer vertices. 
    for c in page:
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        if len(corners) == 4: 
            break
    corners = sorted(np.concatenate(corners).tolist())
    corners = order_points(corners)

    dest_corners = find_dest(corners)

    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(dest_corners))
    result = cv2.warpPerspective(orig_img, M, (dest_corners[2][0], dest_corners[2][1]),
                                flags=cv2.INTER_LINEAR)
    return result


# Create a link to download a file
def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    decodingCode = base64.b64encode(buffered.getvalue())
    img_dcd = decodingCode.decode()
    link = f'<a href="data:file/txt;base64,{img_dcd}" download="{filename}">{text}</a>'
    return link

# Set title.
strlt.sidebar.title('onScanner: Document Scanning Application')

# canvas parameters in application
image = None
result = None
uploaded_file = strlt.sidebar.file_uploader("Upload the Document in png or jpg form", type=["png", "jpg"])
col2, col1 = strlt.columns(2)

if uploaded_file !=  None:

    # Convert the file to an opencv image.
    fileRead = bytearray(uploaded_file.read())
    file_bytes = np.asarray(fileRead, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    
    h, w = image.shape[:2]
    hh, w2 = int(h * 400 / w), 400

    
    
    with col2:
            strlt.title('Your document')
            strlt.image(image, channels='BGR', use_column_width=True)
    with col1:
            strlt.title('Scanned document')
            result = scan(image)
            strlt.image(result, channels='BGR', use_column_width=True)
    
    if result is not None:
        # Display link.
        result = Image.fromarray(result[:, :, ::-1])
        strlt.sidebar.markdown(get_image_download_link(result, 'scannedDocument.png', 'Download ' + 'Scanned Document'),
                            unsafe_allow_html=True)
    else:
        ''