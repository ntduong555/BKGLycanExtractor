import fitz, sys, os, cv2,shutil, pdfplumber, time, ntpath
from .submit import *
from .glycanExtractor import compare2img, countcolors, extractGlycanTopology,buildglycan
import numpy as np
from shutil import *
##########  this script take both pdfplumber to extract coordinate and fitz to manipulate pdf file
##########  "Example command: py abbitatePDF.py <pdf file name> <training set>"  #####################


def checkpath(workdir):
    # check to see if test and pages exist
    check_path = f"{workdir}/test/pages/"
    isdir = os.path.isdir(check_path)
    #print(f"Checking \"{check_path}\" exist? {isdir}")
    if isdir == True:
        #print("There are files in \"test\" folder proceed to delete them.")
        os.makedirs(check_path)
    else:
        #print("Created path: test/pages.")
        os.makedirs(check_path)


def extract_img_obj(path):
    pdf_file = pdfplumber.open(path)
    array=[]
    count=0
    #print("Loading pages:")
    for i, page in enumerate(pdf_file.pages):
        page_h = page.height
        #print(f"-- page {i} --")
        for j, image in enumerate(page.images):
            #print(image)
            box = (image['x0'], page_h - image['y1'], image['x1'], page_h - image['y0'])
            #image_id=f"p{image['page_number']}-{image['name']}-{image['x0']}-{image['y0']}"
            image_id =f"iid_{count}"
            image_xref=image['stream'].objid
            image_page = image['page_number']
            array.append((image_page,image_id,image_xref,box))
            count+=1

    return array


def findglycans(image_path,workdir,base_configs):
    #extract location of all glycan from image file path
    array = []
    base = os.getcwd()
    #coreyolo = r"/home/nduong/demo/BKGLycanExtractor/configs/coreyolo.cfg"
    #weight = r"/home/nduong/demo/BKGLycanExtractor/configs/Glycan_300img_5000iterations.weights"
    weight=base_configs+"Glycan_300img_5000iterations.weights"
    coreyolo=base_configs+"coreyolo.cfg"
    #print(f"weight1: type: {type(weight)} and: {weight}")
    #print(f"weight2: type: {type(weight2)} and: {weight2}")
    net = cv2.dnn.readNet(weight, coreyolo)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #colors = np.random.uniform(0, 255, size=(1, 3))

    #blue = (255, 0, 0)
    #green = (0, 255, 0)
    #red = (0, 0, 255)
    count=0

    image = cv2.imread(image_path)
    origin_image = image.copy()
    height, width, channels = image.shape
    ############################################################################################
    #fix issue with
    ############################################################################################
    white_space = 200
    bigwhite = np.zeros([image.shape[0] +white_space, image.shape[1] +white_space, 3], dtype=np.uint8)
    bigwhite.fill(255)
    bigwhite[0:image.shape[0], 0:image.shape[1]] = image
    image = bigwhite.copy()
    detected_glycan = image.copy()
    #cv2.imshow("bigwhite", bigwhite)
    #cv2.waitKey(0)

    ############################################################################################
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # loop through results and print them on images
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # filter only confidence higher than 60%
            if confidence > 0.6:
                center_x = int(detection[0] * (width+white_space))
                center_y = int(detection[1] * (height+white_space))
                w = int(detection[2] * (width+white_space))
                h = int(detection[3] * (height+white_space))

                # Rectangle coordinates
                x = int(center_x - w / 2)-int(0.2*w)
                y = int(center_y - h / 2)-int(0.2*h)

                w=int(1.4*w)
                h = int(1.4 * h)
                # fix bug that prevent the image to be crop outside the figure object
                if x<=0:
                    x=0
                if y <=0:
                    y=0

                if x+w >= (width):
                    w=int((width)-x)
                if y+h >= (height):
                    h=int((height)-y)

                p1 = (x,y)
                p2 = (x+w,y+h)
                #print(p1,p2)
                cv2.rectangle(detected_glycan,p1,p2,(0,255,0),3)


                boxes.append([x, y, w, h, confidence])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # cv.dnn.NMSBoxesRotated(	bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]]	)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(f"\nGlycan detected: {len(boxes)}")
    #print(f"Path: {image_path}")
    #cv2.imshow("Image", detected_glycan)
    #cv2.waitKey(0)
    image=origin_image.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h, confidence = boxes[i]
            #fix some minorbugs when y<0
            if y<=0:
                y=0


            #print(y, y + h, x, x + w)
            aux_cropped = image[y:y+h,x:x+w].copy()
            #print(y,y+h,x,x+w)
            #aux_crop=cv2.resize(aux_crop,None,fx=1,fy=1)


            #cv2.imshow("Image", aux_cropped)
            #cv2.waitKey(0)


            count+=1
            count_dictionary,final,origin,mask_dict, return_contours = countcolors(aux_cropped,base_configs)
            save_origin=origin.copy()
            mono_dict, a, b = extractGlycanTopology(mask_dict, return_contours, origin)
            if mono_dict!={}:
                glycoCT = buildglycan(mono_dict)
            else:
                glycoCT = "GlycoCT error. buildglycan"

            #print("glycoCT is", buildglycan(mono_dict))
            all_masks = list(mask_dict.keys())
            all_masks.remove("black_mask")
            all_masks = sum([mask_dict[a] for a in all_masks])

            #print(image_path)
            basename = ntpath.basename(image_path).split('.')[0]
            try:
                os.makedirs(f"{workdir}/test/{basename}-{str(count)}/")
            except FileExistsError:
                pass
            cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/save_origin.png", save_origin)
            cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/final.png", final)
            cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/-black_mask.png", mask_dict["black_mask"])
            cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/-all_mask.png", all_masks)
            cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/a.png", a)
            cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/b.png", b)
            array.append((x/width,y/height,(x+w)/width,(y+h)/height,confidence,count_dictionary,basename+"-"+str(count),glycoCT))

    #if image.shape[0] > 1500 or image.shape[1] > 1200:
    #    image = cv2.resize(image, None, fx=0.5, fy=0.5)  # , None, fx=1, fy=1)
    #cv2.imshow("Image", image)
    #key = cv2.waitKey(0)
    return array #%of xy coordinate and with/height percentage



def annotatePDFGlycan(work_dict):
    token=work_dict["token"]
    path=work_dict["infilename"]
    workdir=work_dict["workdir"]
    outfilename=work_dict["outfilename"]
    base_configs=work_dict["base_configs"]


    try:
        checkpath(workdir)
    except PermissionError:
        checkpath(workdir)
    image_names = []
    annotate_log=open(outfilename.replace(".pdf","_log.txt"),"a")
    annotate_log.write(f"{token}\n{path}\n{outfilename}")
    # get image name list
    #stream = open(path,"rb").read()
    doc = fitz.open(path)

    image_array = extract_img_obj(path)
    #print(f"Found {len(image_array)} Figures.")
    annotate_log.write(f"\nFound {len(image_array)} Figures.")
    #rest
    for p,page in enumerate(doc.pages()):
        img_list = [image for image in image_array if image[0]==(p+1)]
        #print(f"##### {page} found figures: {len(img_list)}")
        annotate_log.write(f"\n##### {page} found figures: {len(img_list)}")
        for img in img_list:
            xref = img[2]
            img_name = f"{p}-{img[1]}"
            x0, y0, x1, y1 = img[3]
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
            h = y1 - y0
            w = x1 - x0
            rectangle = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
            #print(f"@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")
            annotate_log.write(f"\n@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")

            pixel = fitz.Pixmap(doc, xref)
            #page.drawRect(rectangle, color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=False)
            if h > 60 and w > 60:
                pixel.writePNG(rf"{workdir}/test/p{p}-{xref}.png")  # xref is the xref of the image
                #print(f" save image to {workdir}test/p{p}-{xref}.png")
                annotate_log.write(f"\n save image to {workdir}test/p{p}-{xref}.png")

                #time.sleep(0.1)
                array = findglycans(rf"{workdir}/test/p{p}-{xref}.png",workdir,base_configs) # find glycan using object classification algorihtm
                #print(f"Glycans Found")
                annotate_log.write(f"\nGlycans Found")

                for glycan_bbox in array:
                    g_x0,g_y0,g_x1,g_y1,confidence,count_dictionary,glycan_id,glycoCT=glycan_bbox
                    #print(count_dictionary)
                    imgcoordinate_page= (x0+g_x1*float(x1-x0),  y0+g_y1*float(y1-y0))

                    total_count = count_dictionary['Glc']+count_dictionary['GlcNAc']+count_dictionary['GalNAc']+count_dictionary['NeuAc']+count_dictionary['Man']+count_dictionary['Gal']+count_dictionary['Fuc']

                    #predict accession
                    if glycoCT!="Error in glycan structure" and total_count <= int(glycoCT.count("\n")/2):
                        #print("submitting:", [glycoCT])
                        annotate_log.write(f"\nsubmitting:{[glycoCT]}")

                        accession = searchGlycoCT(glycoCT)
                        #print("found:", accession)
                        annotate_log.write(f"\nfound: {accession}")
                    else:
                        accession="not found"
                        #print("found:", accession)
                        annotate_log.write(f"\nfound: {accession}")

                    ############annotation of glycan image here#############################
                    uri_base="https://gnome.glyomics.org/StructureBrowser.html?"
                    if accession=="not found":
                        glycan_uri=uri_base+f"Glc={count_dictionary['Glc']}&GlcNAc={count_dictionary['GlcNAc']}&GalNAc={count_dictionary['GalNAc']}&NeuAc={count_dictionary['NeuAc']}&Man={count_dictionary['Man']}&Gal={count_dictionary['Gal']}&Fuc={count_dictionary['Fuc']}"
                    else:
                        glycan_uri =uri_base+"focus="+accession
                    page.insertLink({'kind': 2, 'from': fitz.Rect(x0+g_x0*float(x1-x0),y0+g_y0*float(y1-y0),x0+g_x1*float(x1-x0),y0+g_y1*float(y1-y0)), 'uri': glycan_uri})
                    comment = f"Glycan id: {glycan_id} found with {str(confidence * 100)[:5]}% confidence."  # \nDebug:{count_dictionary}|"+str(imgcoordinate_page)+f"|{str(x1-x0)},{g_x1},{y1-y0},{g_y1}"
                    comment += f'\nPredicted accession:\n{accession}'
                    comment += f'\nPredicted glycoCT:\n{glycoCT}'
                    page.addTextAnnot(imgcoordinate_page, comment, icon="Note")
                    #page.addFileAnnot(imgcoordinate_page, comment, icon="Note")
                    page.drawRect((g_x0, g_y0, g_x1,g_y1), color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=True)
                if array!=[]:
                    page.addTextAnnot((x0,y0),f"Found {len(array)} glycans\nObj: {img_name} at coordinate: {x0, y0, x1, y1} ", icon="Note")
                    page.drawRect(rectangle, color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=False)
    doc.save(outfilename)
    #doc.save(f"{workdir}/output/Result.pdf")
    annotate_log.close()
    return True

#main!
if __name__ =="__main__":
    try:
        checkpath()
    except PermissionError:
        time.sleep(2)
        checkpath()

    doc = fitz.open()
    #dicitonary = extract_img_obj(path)
    print("#########################annotatePDFGlycan#######################")
    #path = sys.argv[1]
    path = "PIIS0091674914002905.pdf"
    path = "1-s2.0-S187439191830246X-main.pdf"
    path = "temp/test.pdf"
    path = "temp/Supplementary Figure 1.pdf"
    #path="temp/Doc1.pdf"
    #p#ath = "temp/test2.pdf"
    #path = "temp/jpgtest.pdf"
    #path = "acs.jproteome.5b00756.pdf"
    weight_v="configs/Glycan_300img_5000iterations.weights"
    print("Loaded file:",path,type(path))
    annotatePDFGlycan(path,weight_v)
    #findglycans("test/1-s20-S0065230X14000141-f08-02-9780128013816.png")
    print("Scrip Finished annotated:",path)
    os.startfile(r"test\\0000page.pdf")
