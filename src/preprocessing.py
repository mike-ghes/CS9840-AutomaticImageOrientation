import numpy as np
import cv2
import os
import random
import itertools

#change directory to VOC2007/JPEGImages or specificy full file name
voc2012 = ".\\datasets\\VOC2012\\JPEGImages"
voc2007 = ".\\datasets\\VOC2007\\JPEGImages"
easy = ".\\datasets\\easy_mode"


def perImage(fname, img_bgr, conversion, N_cm=7, N_edh=0):
    """
    Feature selection that will be executed per image
    :params fname: Filename to be used as the description in csv file
    :params img_bgr: jpeg image that has already been read from file
    :params conversion: cv2 Color conversion code. 
    :params N_cm: Number of blocks to use for color moments
    :params N_edh: Number of blocks to use for edge detection (diabled)
    :returns: One row of a csv file for the image. 
    """

    # Image conversion and rotation ########################################
    
    #Convert to L*u*v (or HSV or LAB ...) for CM
    img = cv2.cvtColor(img_bgr, conversion)
    
    #And grayscale for edge detection
    #img_bw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Choose random rotation
    rot = random.randint(0,3)
    
    # Rotate images
    img = np.rot90(img, rot)
    #img_bw2 = np.rot90(img_bw, rot)    
    
    # Image size 
    h, w = img.shape[0], img.shape[1]
        
    # CM extraction ########################################################
        
    #block sizes
    hb_cm = h / N_cm
    wb_cm = w / N_cm
    cm_mat = np.zeros((N_cm,N_cm,6))  
    
    # Loop through block indices: x={0...N}, y={0..N}
    for x,y in itertools.product(range(N_cm), repeat=2):
        
        #Extract only the relevant block
        block_cm = img[x*hb_cm:(x+1)*hb_cm, y*wb_cm:(y+1)*wb_cm]
        
        #CM: mean L, var L, mean U, var U, mean V, var V
        for z, (i, moment) in zip(range(6), itertools.product(range(3), [np.mean, np.var])):
            cm_mat[x,y,z] = moment(block_cm[:,:,i])
    
    
    # EDH extraction #######################################################
    
    # Code throws an error if the matrix (img_bw2) has been rotated. This appears
    # to be a known bug in OpenCV. Relevant code has been commented out.
    
    ##Not quite accurate. 
    #hog = cv2.HOGDescriptor()
    #grad0, grad1 = hog.computeGradient(img_bw2)
    #
    ##Set up block size and matrix for EDH
    #hb_edh = h / N_edh
    #wb_edh = w / N_edh
    #edh_mat = np.zeros((N_edh,N_edh,4))   
    #
    ##Loop through block indices                          
    #for x,y in itertools.product(range(N_edh), repeat=2): 
    #    
    #    #Extract only the relevant block            
    #    block_edh1 = grad0[x*hb_edh:(x+1)*hb_edh, y*wb_edh:(y+1)*wb_edh]
    #    block_edh2 = grad1[x*hb_edh:(x+1)*hb_edh, y*wb_edh:(y+1)*wb_edh]
    #    
    #    #Sum the gradients in each direction?
    #    for z, (block, i) in zip(range(4), itertools.product([block_edh1, block_edh2], [0,1])):
    #        edh_mat[x,y,z] = sum(block[:,:,i].flatten())
    
    
    # String Construction ##################################################
    
    # Convert to 1 dimensional array 
    cm_vector  = cm_mat.flatten()
    #edh_vector = edh_mat.flatten()
    
    #Concatenate everything together
    #return ", ".join(['"'+fname+'"', str(rot)] + map(str,cm_vector) + map(str, edh_vector)) + "\n"
    return ", ".join(['"'+fname+'"', str(rot)] + map(str,cm_vector)) + "\n"
    

def preprocess_VOC(jpgPath=voc2012, csvPath = 'processed_VOC2012_LUV.csv', conversion = cv2.COLOR_BGR2LUV):
    """
    Process the VOC (2007, 2012, easy) datasets.
    :params jpgPath: path name to the JPEGImages folder
    :params csvPath: output csv file path.
    :params conversion: cv2 Color conversion code.
    :returns: None
    """
    
    #All images are in root folder
    _, _, fnames = os.walk(jpgPath).next()

    # Open output file
    with open(csvPath, 'w') as outfile:
        
        #Loop over all jpg's only. 
        for f in fnames:
            if f[-3:] == 'jpg':
                
                #Image is in BGR
                img_bgr = cv2.imread(jpgPath +"\\"+ f)
                
                #Write line to csv
                outfile.write(perImage(f, img_bgr, conversion))
                

def preprocess_Indoor(csvPath = 'processed_IndoorScenes_LUV.csv', conversion = cv2.COLOR_BGR2LUV):
    """
    Process the VOC (2007, 2012, easy) datasets.
    :params jpgPath: path name to the JPEGImages folder
    :params csvPath: output csv file path.
    :params conversion: cv2 Color conversion code.
    :returns: None
    """
    # Open log file    
    with open('datasets\\logs\\' + csvPath[:-3] + 'log', 'w') as logfile:
        
        # Open csv file
        with open('datasets\\' + csvPath, 'w') as outfile:
            
            # Images are in subfolders
            _, subfolders, _ = os.walk(".\\datasets\\IndoorScenes").next()
            
            # Loop through all subfolders
            for folder in subfolders:
                
                # Takes a while so let the user know how far along we are
                fpath = ".\\datasets\\IndoorScenes\\" + folder
                print fpath
                
                #Loop through all images in subfolder
                _, _, imgs = os.walk(fpath).next()
                for f in imgs:
                
                    # IndoorScenes contains some mislabelled gif images. 
                    # If a problem is encountered with imread, skip the image and write to log file.
                    try:
                        if f[-3:] == 'jpg':
                            
                            #Image is in BGR
                            img_bgr = cv2.imread(fpath + "\\" + f)
                        
                            # Write line to csv file
                            outfile.write(perImage(f, img_bgr, conversion))
                    except:
                        
                        # Print the skipped image to both console and log file
                        logfile.write(fpath + f + " skipped.\n")
                        print f, "skipped"