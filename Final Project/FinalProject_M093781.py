import os
import cv2
import xml.etree.ElementTree as ET
import lxml.etree as etree

# ------ Find anomaly part ----- #
def findAnomaly(normal_gray, anomaly_gray):
    '''
    This section try to find the anomaly part by using normal gray - anomaly gray.
    Due to pepper salt noise, use for loop to ignore the extremum (>245 or < 15).
    Finally use median blur to reduce the rest noise, return anomaly map.
    Con: O(n^2), take time to delete extremum.
    '''
    if anomaly_type_upper in ['Mouse_bite', 'Open_circuit', 'Missing_hole']:
        anomaly = anomaly_gray - normal_gray
    elif anomaly_type_upper in ['Spur', 'Short', 'Spurious_copper']:
        anomaly = normal_gray - anomaly_gray

    anomaly[anomaly > 245] = 0
    anomaly[anomaly < 10] = 0
    anomaly = cv2.medianBlur(anomaly, 3)
    _, thres = cv2.threshold(anomaly, 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite('error.jpg', thres)
    return thres

# ------ Find the coordinate of anomalies ---- #
def annotate(anomaly_map, anomaly_img, xmlpath, normal_img):
    '''
    This section find the x_min, x_max, y_min, y_max of anomalies from anomaly map.
    '''
    contours, _ = cv2.findContours(anomaly_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edge = 20
    x_minList = []
    x_maxList = []
    y_minList = []
    y_maxList = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 30:
            (x,y,w,h) = cv2.boundingRect(contour)
            x_minList.append(x-edge)
            x_maxList.append(x+w+edge)
            y_minList.append(y-edge)
            y_maxList.append(y+h+edge)

            cropAnomaly(anomaly_img, normal_img, y_minList[i], y_maxList[i], x_minList[i], x_maxList[i])
            ### classify the anomaly and return 'class' ###
        else:
            x_minList.append(None)
            x_maxList.append(None)
            y_minList.append(None)
            y_maxList.append(None)
    ### write xml ###
    writeXml('{}'.format(anomaly_type_lower), x_minList, x_maxList, y_minList, y_maxList, xmlpath)

def cropAnomaly(anomaly_img, normal_img, y_min, y_max, x_min, x_max):
    global an_count
    offset = 0
    try:
        cropimg = anomaly_img[y_min+offset:y_max-offset,
                              x_min+offset:x_max-offset]
        cv2. imwrite('anomalies/{}/{}_anomaly{}.jpg'.format(anomaly_type_upper, anomaly_type_lower, an_count), cropimg)
        cropimg = normal_img[y_min + offset:y_max - offset,
                             x_min + offset:x_max - offset]
        cv2.imwrite('normal/{}/{}_normal{}.jpg'.format(anomaly_type_upper, anomaly_type_lower, an_count), cropimg)
        an_count += 1
    except:
        print(y_min, x_min, "cannot crop.")

# ------ Write xml ------ #
def writeXml(class_name, x_minList, x_maxList, y_minList, y_maxList, xmlpath):
    annotation = ET.Element('annotation')
    for i in range(len(x_minList)):
        if x_minList[i] != None:
            x_min = x_minList[i]
            x_max = x_maxList[i]
            y_min = y_minList[i]
            y_max = y_maxList[i]
            # -- build object branch
            object = ET.SubElement(annotation, 'object')
            # -- in branch2: object
            name = ET.SubElement(object, 'name')
            name.text = str(class_name)
            pose = ET.SubElement(object, 'pose')
            pose.text = "Unspecified"
            truncated = ET.SubElement(object, 'truncated')
            truncated.text = "0"
            difficult = ET.SubElement(object, 'difficult')
            difficult.text = "0"
            bndbox = ET.SubElement(object, 'bndbox')
            # -- in branch3: bndbox
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(x_min)
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(x_max)
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(y_min)
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(y_max)

    tree = ET.ElementTree(annotation)
    tree.write(xmlpath)
    root = etree.parse(xmlpath)
    f = open(xmlpath, 'w')
    f.write(etree.tostring(root, pretty_print=True, encoding="unicode"))
    f.close()

if __name__ == '__main__':
    an_count = 0
    PCB_USED = '/Users/zhaoyuhan/Documents/OneDrive - 清華大學/碩一下/Image Processing/Final Project/PCB_DATASET/PCB_USED/'
    global anomaly_type_upper
    global anomaly_type_lower
    anomaly_type_upper = 'Spurious_copper'
    anomaly_type_lower = 'spurious_copper'

    # single image
    # anomaly_img_path = '/Users/zhaoyuhan/Documents/OneDrive - 清華大學/碩一下/Image Processing/Final Project/PCB_DATASET/images/Spurious_copper/01_spurious_copper_03.jpg'
    # normal = PCB_USED + '01.JPG'
    #
    # anomaly_img = cv2.imread(anomaly_img_path)
    # anomaly_gray = cv2.cvtColor(anomaly_img, cv2.COLOR_BGR2GRAY)
    # normal_img = cv2.imread(normal)
    # normal_gray = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)
    #
    # anomaly_map = findAnomaly(normal_gray, anomaly_gray)
    # annotate(anomaly_map, anomaly_img, 'ann/' + 'annotation.xml', normal_img)

    # loop folder
    anomaly_img_folder = '/Users/zhaoyuhan/Documents/OneDrive - 清華大學/碩一下/Image Processing/Final Project/PCB_DATASET/images/{}/'.format(anomaly_type_upper)
    allFileList = os.listdir(anomaly_img_folder)
    for file in allFileList:
        print(file)
        anomaly_img_path = anomaly_img_folder + file
        board = str(file[:2])
        normal = PCB_USED + '{}.JPG'.format(board)

        anomaly_img = cv2.imread(anomaly_img_path)
        anomaly_gray = cv2.cvtColor(anomaly_img, cv2.COLOR_BGR2GRAY)
        normal_img = cv2.imread(normal)
        normal_gray = cv2.cvtColor(normal_img, cv2.COLOR_BGR2GRAY)

        anomaly_map = findAnomaly(normal_gray, anomaly_gray)
        annotate(anomaly_map, anomaly_img, 'ann/{}/'.format(anomaly_type_upper) + file[:-4] + '.xml', normal_img)
