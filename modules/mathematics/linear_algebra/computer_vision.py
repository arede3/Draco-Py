import numpy as np
import cv2 as cv
import sklearn as skl
from linear_algebra import *
from matplotlib import pyplot as plt

testPath = 'C://Users//irona//Desktop//Cyberbotics Lab//Computer Vision//OpenCVLogo.PNG'

# GRAY SCALE <--> BGR/RGB SCALE
BGR2GRAY = cv.COLOR_BGR2GRAY
RGB2GRAY = cv.COLOR_RGB2GRAY
GRAY2BGR = cv.COLOR_GRAY2BGR
GRAY2RGB = cv.COLOR_GRAY2RGB

# XYZ SCALE <--> BGR/RGB SCALE
BGR2XYZ = cv.COLOR_BGR2XYZ
RGB2XYZ = cv.COLOR_RGB2XYZ
XYZ2BGR = cv.COLOR_XYZ2BGR
XYZ2RGB = cv.COLOR_XYZ2RGB

# YCrCb SCALE <--> BGR/RGB SCALE
BGR2YCrCb = cv.COLOR_BGR2YCrCb
RGB2YCrCb = cv.COLOR_RGB2YCrCb
YCrCb2BGR = cv.COLOR_YCrCb2BGR
YCrCb2RGB = cv.COLOR_YCrCb2RGB

# HSV SCALE <--> BGR/RGB SCALE
BGR2HSV = cv.COLOR_BGR2HSV
RGB2HSV = cv.COLOR_RGB2HSV
HSV2BGR = cv.COLOR_HSV2BGR
HSV2RGB = cv.COLOR_HSV2RGB

# HSV SCALE <--> BGR/RGB SCALE
BGR2HSV = cv.COLOR_BGR2HSV
RGB2HSV = cv.COLOR_RGB2HSV
HSV2BGR = cv.COLOR_HSV2BGR
HSV2RGB = cv.COLOR_HSV2RGB

# HLS SCALE <--> BGR/RGB SCALE
BGR2HLS = cv.COLOR_BGR2HLS
RGB2HLS = cv.COLOR_RGB2HLS
HLS2BGR = cv.COLOR_HLS2BGR
HLS2RGB = cv.COLOR_HLS2RGB

# HLS SCALE <--> BGR/RGB SCALE
BGR2HLS = cv.COLOR_BGR2HLS
RGB2HLS = cv.COLOR_RGB2HLS
HLS2BGR = cv.COLOR_HLS2BGR
HLS2RGB = cv.COLOR_HLS2RGB

# CIE L*a*b SCALE <--> BGR/RGB SCALE
BGR2Lab = cv.COLOR_BGR2Lab
RGB2Lab = cv.COLOR_RGB2Lab
Lab2BGR = cv.COLOR_Lab2BGR
Lab2RGB = cv.COLOR_Lab2RGB

# CIE L*u*v SCALE <--> BGR/RGB SCALE
BGR2Luv = cv.COLOR_BGR2Luv
RGB2Luv = cv.COLOR_RGB2Luv
Luv2BGR = cv.COLOR_Luv2BGR
Luv2RGB = cv.COLOR_Luv2RGB

# Bayer SCALE <--> BGR/RGB SCALE
BayerBG2BGR = cv.COLOR_BayerBG2BGR
BayerBG2RGB = cv.COLOR_BayerBG2RGB
BayerGB2BGR = cv.COLOR_BayerGB2BGR
BayerGB2RGB = cv.COLOR_BayerGB2RGB
BayerRG2BGR = cv.COLOR_BayerRG2BGR
BayerRG2RGB = cv.COLOR_BayerRG2RGB
BayerGR2BGR = cv.COLOR_BayerGR2BGR
BayerGR2RGB = cv.COLOR_BayerGR2RGB

class Image(object):

    def __init__(self,*readArgs):
        self.image = Matrix(cv.imread(*readArgs),dtype=np.uint8,ndarrayType='tensor')
        n = len(self.image.shape)

        self.imgHeight = None
        self.imgWidth = None
        self.numChannels = None
        
        if n == 3:
            [self.imgHeight,self.imgWidth,self.numChannels] = self.image.shape
        else:
            [self.imgHeight,self.imgWidth] = self.image.shape
        
    def displayImage(self,waitVal=0):
        k = cv.waitKey(waitVal) & 0xFF
        cv.imshow('image',self.image.data)

        if k == ord('q'):
            cv.destroyAllWindows()

    def saveImage(self,*writeArgs):
        cv.imwrite(*writeArgs)

    def convertImageColor(self,cvImageColor,ndarrayType='matrix'):
        self.image = Matrix(cv.cvtColor(self.image.data,cvImageColor),np.uint8,ndarrayType)
        
    def convertImageFormat(self,cvImageFormat):
        pass

    def getImageDims(self):
        return [self.imgHeight,self.imgWidth]

    def __str__(self):
        return 'Image:\n%s' % (str(self.image))

    def __repr__(self):
        return str(self)
    
class Video(object):

    def __init__(self,*vidCapArgs):
        self.video = cv.VideoCapture(*vidCapArgs)
        self.frameList = []
        self.retList = []

    def initializeFrameInfo(self):
        while self.video.isOpened():
            ret,frame = self.video.read()
            self.frameList.append(Matrix(frame))
            self.retList.append(ret)

            if frame.empty():
                break

        self.video.release()
        cv.destroyAllWindows()
        
    def displayVideo(self):
        while self.video.isOpened():
            ret,frame = self.video.read()

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv.destroyAllWindows()

    def saveVideoFromCamera(self,*saveArgs):
        [vidCapArgs,writeArgs] = saveArgs
        
        video = cv.VideoCapture(*vidCapArgs)
        out = cv.VideoWriter(*writeArgs)

        while video.isOpened():
            ret, frame = video.read()

            if ret == True:
                out.write(frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        video.release()
        out.release()
        cv.destroyAllWindows()
        
    def convertFrameColor(self,frame,cvImageColor):
        pass

    def convertFrameFormat(self,frame,cvImageFormat):
        pass

    def getFrame(self,frameIndex):
        return [self.retList[frameIndex],self.frameList[frameIndex]]

    def getFramesList(self):
        return self.frameList

    def getRetList(self):
        return self.retList

    def __str__(self):
        return 'Video:\n%s' % (str(self.frameList))

    def __repr__(self):
        return str(self)

class DrawingTool(object):

    def __init__(self,oldImage):
        self.originalImage = oldImage
        self.oldImage = oldImage
        self.geometriesList = []
        self.newImage = None

    def drawShape(self,shapeName):
        pass

    def line(self,pt1,pt2,color,thickness=1,lineType=cv.LINE_8,shift=0):
        self.newImage = cv.line(self.oldImage.data,pt1,pt2,color,thickness,lineType,shift)
        self.oldImage = Matrix(self.newImage)

    def polylines(self,pts,isClosed,color,thickness=1,lineType=cv.LINE_8,shift=0):
        self.newImage = cv.polylines(self.oldImage.data,pts,isClosed,color,thickness,lineType,shift)
        self.oldImage = Matrix(self.newImage)

    def rectangle(self,pt1,pt2,color,thickness=1,lineType=cv.LINE_8,shift=0):
        self.newImage = cv.rectangle(self.oldImage.data,pt1,pt2,color,thickness,lineType,shift)
        self.oldImage = Matrix(self.newImage)

    def rectangle2(self,rec,color,thickness=1,lineType=cv.LINE_8,shift=0):
        self.newImage = cv.rectangle(self.oldImage.data,rec,color,thickness,lineType,shift)
        self.oldImage = Matrix(self.newImage)
    
    def circle(self,center,radius,color,thickness=1,lineType=cv.LINE_8,shift=0):
        self.newImage = cv.circle(self.oldImage.data,center,radius,color,thickness,lineType,shift)
        self.oldImage = Matrix(self.newImage)

    def ellipse(self,center,axes,angle,startAngle,endAngle,color,thickness=1,lineType=cv.LINE_8,shift=0):
        self.newImage = cv.ellipse(self.oldImage.data,center,axes,angle,startAngle,endAngle,color,thickness,lineType,shift)
        self.oldImage = Matrix(self.newImage)

    def ellipse2(self,box,color,thickness=1,lineType=cv.LINE_8):
        self.newImage = cv.ellipse(self.oldImage.data,box,color,thickness,lineType)
        self.oldImage = Matrix(self.newImage)
        
    def ellipse2Poly(self,*args):
        return self.ellipse2Poly(*args)

    def fillConvexPoly(self,pts,color,lineType=cv.LINE_8,shift=0):
        self.newImage = cv.fillConvexPoly(self.oldImage.data,pts,color,lineType,shift)
        self.oldImage = Matrix(self.newImage)

    def fillPoly(self,pts,color,lineType=cv.LINE_8,shift=0):
        self.newImage = cv.fillPoly(self.oldImage.data,pts,color,lineType,shift)
        self.oldImage = Matrix(self.newImage)
        
    def drawMarker(self,position,color,markerType=cv.MARKER_CROSS,markerSize=20,thickness=1,lineType=8):
        self.newImage = cv.drawMarker(self.oldImage.data,position,color,markerType,markerSize,thickness,lineType)
        self.oldImage = Matrix(self.newImage)

    def getFontScaleFromHeight(fontFace,pixelHeight,thickness=1):
        return cv.getFontScaleFromHeight(fontFace,pixelHeight,thickness)

    def getTextSize(text,fontFace,fontScale,thickness):
        return cv.getTextSize(text,fontFace,fontScale,thickness)

    def putText(self,text,org,fontFace,fontScale,color,thickness=1,lineType=cv.LINE_8,bottomLeftOrigin=False):
        self.newImage = cv.putText(self.oldImage.data,text,org,fontFace,fontScale,color,thickness,lineType,bottomLeftOrigin)
        self.oldImage = Matrix(self.newImage)
        
    def arrowedLine(self,pt1,pt2,color,thickness=1,lineType=8,shift=0,tipLength=0.1):
        self.newImage = cv.arrowedLine(self.oldImage.data,pt1,pt2,color,thickness,lineType,shift,tipLength)
        self.oldImage = Matrix(self.newImage)

    def clipLine(self,imgRect,pt1,pt2):
        [retVal,pt1,pt2] = cv.clipLine(imgRect,pt1,pt2)
        return [retVal,pt1,pt2]

    def drawContours(self,contours,countourIdx,color,thickness=1,lineType=cv.LINE_8):
        self.newImage = cv.drawContours(self.oldImage.data,contours,contourIdx,color,thickness,lineType)
        self.oldImage = Matrix(self.newImage)

######## TEST FUNCTIONS FOR EXPERIMENTING WITH DIFFERENT IMAGE FILTERS        
class ImageFilter(object):

    def __init__(self,*filterArgs):
        self.filterArgs = filterArgs
        self.filterType = None

    def setFilterType(self,filterType):
        self.filterType = filterType

    def getFilterType(self):
        return self.filterType

    def convolution(self):
        [image,kernel] = self.filterArgs

        [imgHeight,imgWidth] = image.shape

        [kHeight,kWidth] = kernel.shape
        imagePad = None
        
        n = len(image.shape)

        if n == 3:
            imagePad = np.pad(image.data, pad_width = ((kernel_h//2,kernel_h//2),(kernel_w//2,kernel_w//2),(0,0)),mode='constant',constant_values=0).astype(np.float64)
        elif n == 2:
            imagePad = np.pad(image.data, pad_width = ((kernel_h//2,kernel_h//2),(kernel_w//2,kernel_w//2)),mode='constant',constant_values=0).astype(np.float64)
        else:
            raise Exception('Input data is incorrect!')

        h = kernel_h // 2
        w = kernel_w // 2

        imageConv = np.zeros(imagePad.shape)

        for i in range(h, imagePad.shape[0]-h):
            for j in range(w, imagePad.shape[1]-w):
                x = imagePad[i-h:i-h+kernel_h,j-w:j-w+kernel_w]
                x = x.flatten()*kernel.flatten()
                imageConv[i][j] = x.sum()
        h_end = -h
        w_end = -w

        if h == 0:
            return imageConv[h:,w:w_end]
        if w == 0:
            return imageConv[h:h_end,w:]

        return Matrix(imageConv[h:h_end,w:w_end])

    def gaussianFilter(self, sigma):
        [image,kernel] = self.filterArgs
        filterSize = 2 * int(4*sigma + 0.5) + 1
        gaussian_filter = np.zeros((filter_size,filter_size), np.float64)
        m = filter_size//2
        n = filter_size//2

        for x in range(-m, m+1):
            for y in range(-n, n+1):
                x1 = 2*np.pi*(sigma**2)
                x2 = np.exp(-(x**2 + y**2)/(2*sigma**2))
                gaussian_filter[x+m,y+n] = (1/x1)*x2

        im_filtered = np.zeros_like(image.data, dtype=np.float64)
        for c in range(3):
            im_filtered[:, :, c] = convolution(image[:,:,c],gaussian_filter)

        return Matrix(im_filtered.astype(np.uint8))
                        
        
    
        
