import openslide
import xml.etree.ElementTree as ET
import csv	
import numpy as np
from skimage import filters, color
from skimage.morphology import disk
from skimage.morphology import opening, closing, dilation
import pickle
import os
from PIL import Image
import xmltodict

class WSI(object):

    def __init__(self, wsi_file,mag0,mask,mask_dir,annotation,annotations_file,annotation_dir,max_per_class,anno_threshold,ignore_bg, patchesDir):
		"""
        
        	An object for load WSI, compute tissue mask and annotation mask.
        	:param wsi_file: whole slide image file path (ex. '/nobackup/alabdullatif/barretts/images/test/10790.svs').
        	:param mag0: the highest magnification for the WSI (ex. images in my research have 40X as the highest magnification).
        	:param mask: True if generating a tissue mask is needed, and False otherwise
        	:param mask_dir: The location where to save the generated tissue mask.
        	:param annotation: True if generating a annotation mask is needed, and False otherwise.
  			:param annotations_file: The file that contains the WSI annotations.
  			:param annotation_dir: The location where to save the generated annotation mask.
  			:param xml_dir: The location where to save the generated annotations in xml form.
  			:param max_per_class: maximum number of the sampled pat ches per class.
  			:param anno_threshold: percentage of  accepted tissue to background.
  			:param ignore_bg: True if we want to ignor background during the sampling false otherwise.
  			:param patchesDir: The location where to save the sampledd patches.
    	"""
    	im=glob.glob(wsi_file)
    	
    	#check if the file exist
    	if len(im)==0:
    		print('The file does not exist')
    	else:
			#Load the WSI
			slide=openslide.OpenSlide(im[0])
			#store the file name
			x=im[0].split('.')[0].split('/')
			self.ID=x[len(x)-1]
			#Get the available levels of the WSI(Aperio ex. 1,4,16,64)
			self.levels=slide.level_downsamples
			#Get the available magnifications of the WSI(ex. 40X,10X,2.5X,0.6X)
			mag=float(mag0)
			self.mags=[mag/l for l in levels]
			
			#Compute tissue mask
			if mask:
				#The desired magnification to process the tissue mask generation at(2.5X in this research)
				mag2=2.5
				#Get the closest magnification to the specified one.
				diffs = [abs(mag2 - self.mags[i]) for i in range(len(mags))]
    			minimum = min(diffs)
    			level= diffs.index(minimum)
    			#Generate and save tissue mask
				self.tissueMask = self._generate_tissue(slide,level)		
				filename = os.path.join(mask_dir, self.ID + '_TissueMask.png')
    			plt.imsave(filename, self.tissueMask, cmap=cm.gray)
			#Compute annotation mask
			if annotation:
				self._generate_annotation_mask(annotations_file, xml_dir, slide, annotation_dir )

			#Find all tissue in boxes
			self._Find_tissue_in_boxes(level,annotation_dir,256)

			#Sample patches
			self._Sample_patches(wsi, level,max_per_class,anno_threshold,ignore_bg, patchesDir,annotation_dir)

	
	def _generate_tissue_mask(wsi, level):
        """
        To generate a tissue mask for the WSI.
        This is achieved by Otsu thresholding on the saturation channel, then to remove the noise
        morphological closing and opening operations are applied.

        :param wsi: openSlide object for the WSI
        :param level:the level to process the mask generation at
        :return: the generated mask
        """

        temp = wsi.read_region(location=(0, 0), level=level, size=wsi.level_dimensions[level])
        # Convert to Hue-Saturation-Value.
        hsv = color.convert_colorspace(temp, 'RGB', 'HSV')  
        # Get saturation channel.
        saturation = hsv[:, :, 1] 
        # Otsu threshold.
        threshold = filters.threshold_otsu(saturation)  
        # Tissue is 'high saturation' region.
        mask = (saturation > threshold)  

        # Morphological operations-----------------
        # radius of disk for morphological operations.
        disk_r = 10  
        disk_o = disk(disk_r)
        # remove pepper noise
        mask = closing(mask, disk_o)  
        # remove salt noise
        mask = opening(mask, disk_o)  
        return mask
    
    
    
        
    def _generate_annotation_mask(annotations_file, xml_dir, wsi, annotation_dir ):
    	"""
    	Generates annotation mask 
    	:param annotations_file: The file that contains the WSI annotations.
  		:param xml_dir: The location where to save the generated annotations in xml form.
  		:param wsi: openSlide object for the WSI
  		:param annotation_dir: The location where to save the generated annotation mask.
    	"""
    	
    	#Generate annotation xml file
		self._generate_annotation_xml(annotations_file, xml_dir)
		
		#create annotation mask with size similar to the tissue mask image.
    	mask = np.ones((self.tissueMask.shape[0], self.tissueMask.shape[1], 3), dtype=np.uint8)*255
    	ref_factor = float(mask.shape[1]) / float(wsi.level_dimensions[0][0])
    
    	# Get sorted annotations coordinates and classes
    	polygons, classes = self._Get_annotations_from_xml(os.path.join(xml_dir,self.ID+'.xml'))
  
    	# Iterate over polygons and populate the mask
    	for i, pts in enumerate(polygons):
        
        	pts *=ref_factor
        	
        	if classes[i]=='#00ff00':
        		#green for negative grade dysplasia
        		code = (0,255,0)
        	elif classes[i]=='#0000ff':
        		#blue for low grade dysplasia
        		code = (0,0,255)
        	else:
        		#Red for high grade dysplasia
        		code = (255,0,0)
        
        	rr, cc = polygon(pts[:,0], pts[:,1], mask.shape)
        	for s in range(rr.size):
            	if x[rr[s],cc[s]]:
                	mask[rr[s], cc[s], :] = code
        
        #remove the background from the annotation mask
        for i in range(self.tissueMask.shape[0]):
			for c in range(self.tissueMask.shape[1]):
				if not(self.tissueMask[i,c]):
					mask[i,c]=(255,255,255)
					
    	# Save the mask
    	filename = os.path.join(annotation_dir, self.ID + '_AnnotationMask.png')
    	plt.imsave(filename, mask, cmap=cm.gray)
    
    
    
    
    def _generate_annotation_xml(annotations_file, xml_dir):
        """
        To generate xml for the annotations.
        
        :param annotations_file: .csv file that has the annotations as http link with the form
        http\\...\wsi_name.svs?x_coordinateAt40X+y_coordinateAt40X+height_atTheMagnificationOfTheAnnotation+width_atTheMagnificationOfTheAnnotation+TheMagnificationOfTheAnnotation+quality
        :param xml_dir:folder where to save the xml file
        """
    	allannotations=ET.Element('All_Annotations')
		annotations= ET.SubElement(allannotations,'Annotations')
		annotationGroups= ET.SubElement(allannotations,'AnnotationGroups')
		#add more colors based on the number of classes in the problem(ex this research has 3 classes)
		color=['#00ff00','#0000ff','#ff0000']
		classes=[]
		counter=0
		with open(annotations_file) as csvfile:
			readCSV = csv.reader(csvfile)
			for row in readCSV:
				annotation=ET.SubElement(annotations,'Annotation')
				annotation.set('Color','F4FA58')
				annotation.set('Name','Annotation '+str(counter))
				annotation.set('PartOfGroup',row[1])
				annotation.set('Type','Polygon')
				if not(row[1] in classes):
					classes.append(row[1])
				counter+=1	
				coordinates=ET.SubElement(annotation,'Coordinates')
				cmag=float(40)/float(row[0].split('?')[1].split('+')[4])
				h1=float(row[0].split('?')[1].split('+')[3])*(float(40)/cmag)
				w1=float(row[0].split('?')[1].split('+')[2])*(float(40)/cmag)
				h=[0.0,h1,h1,0.0]
				w=[0.0,0.0,w1,w1]
				for c in range(4):
					coordinate=ET.SubElement(coordinates,'Coordinate')
					coordinate.set('Order',str(c))
					coordinate.set('X',str(float(row[0].split('?')[1].split('+')[0])+h[c]))
					coordinate.set('Y',str(float(row[0].split('?')[1].split('+')[1])+w[c]))
	
		for n in range(len(classes)):
			group=ET.SubElement(annotationGroups,'Group')
			if classes[n]=='G1':
				group.set('Color',color[0])
			elif classes[n]=='G3':
				group.set('Color',color[1])
			else:
				group.set('Color',color[2])
			group.set('Name',classes[n])
			group.set('PartOfGroup','None')


		mydata = ET.tostring(allannotations)
		myfile = open(os.path.join(xml_dir, self.ID+'.xml'), "w")
		myfile.write(mydata)
		myfile.close()
	
	
	
		
	def _Get_annotations_from_xml(xmlFile):
		"""
        Convert xml annotations to numpy arrays.
        :param xmlFile: path to xml file
        :return: list of  annotations in coordinates form and a list of their classes
        """

        with open(xmlFile) as xml_file:
            xml = xmltodict.parse(xml_file.read())

        name_to_color = {}
        
        ##for multi classes cases-------------------
        #for d in xml['All_Annotations']['AnnotationGroups']['Group']:
            #name_to_color[d['@Name']] = d['@Color']
        ##---------------------------------------- 
           
        #for one class cases------------------------
        d=xml['All_Annotations']['AnnotationGroups']['Group']
        name_to_color[d['@Name']] = d['@Color']
        #----------------------------------------
        
        anno = []
        classes = []
        
        #for multi region----------------------
        for polygon_dict in xml['All_Annotations']['Annotations']['Annotation']:
            
            color = name_to_color[polygon_dict['@PartOfGroup']]
            
            coords = polygon_dict['Coordinates']['Coordinate'] # list of ordered
            coords = np.array([[float(_['@Y']), float(_['@X'])] for _ in coords])
            anno.append(coords)
            classes.append(color)
        #----------------------------------------
            
        ##for one region only---------------------  
        #polygon_dict = xml['All_Annotations']['Annotations']['Annotation']
        #color = name_to_color[polygon_dict['@PartOfGroup']]
            
        #coords = polygon_dict['Coordinates']['Coordinate'] # list of ordered
        #coords = np.array([[float(_['@Y']), float(_['@X'])] for _ in coords])
        #anno.append(coords)
        #classes.append(color) 
        ##----------------------------------------
        
        # sort annotations and their classes
        areas = np.zeros(len(anno))
    	for i, pts in enumerate(anno):
        	areas[i] = anno_area(pts[:,1], pts[:,0])
    	sorted = np.argsort(-areas)
    	anno_sorted = [anno[i] for i in sorted]
    	classes_sorted = [classes[i] for i in sorted]
    
    	return anno_sorted, classes_sorted
     
     
     
        
	def _Find_tissue_in_boxes(level,annotation_dir,patch_size):
		"""
        Find all tissue and prepare them for sampling
        :param level: the magnification to process at
        :param annotation_dir: the folder where to save the mask
        :param patch_size: the size of the sampled patches (ex. in this research 256)
        """
        
 		mask=np.zeros(self.tissueMask.shape)
		for i in range(self.tissueMask.shape[0]):
			for c in range(self.tissueMask.shape[1]):
				if(self.tissueMask[i,c]):
					mask[i,c]=1
			
		mask1 = cv2.normalize(src=mask, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		_ , contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour_mask = np.zeros_like(mask1)
		cv2.drawContours(contour_mask, contours,255, 255)
		for i in range(len(contours)):
			(x,y,w,h) = cv2.boundingRect(contours[i])
			factor=self.mags[level]/self.mags[1]
			n=int((patch_size*factor)+0.5)
			w=int(w)+((w/n)-int(w/n))*n
			h=int(h)+((h/n)-int(h/n))*n
			cv2.rectangle(contour_mask, (x,y), (x+w,y+h), (255, 255, 0), -1)   
			
		filename = os.path.join(annotation_dir, self.ID + '_findTissueMask.png')
    	plt.imsave(filename, contour_mask, cmap=cm.gray)
    	
    
    
    
    def _class_c_patch_i(wsi,class_list,class_seeds, c, i,anno_threshold,patch_size,mag,mag0):
        """
        Try and get the ith patch of class c. If we reject return None.
        :param wsi: openSlide object for the WSI
        :param class_list: list of the patches' classes 
        :param class_seeds: list of patches' coordinates
        :param c: class
        :param i: index
        :param anno_threshold: percentage of  accepted tissue to background
        :param patch_size: the size of the sampled patches (ex. in this research 256)
        :param mag: the desired magnification to get the patch
        :param mag0: the highest magnification for the WSI (ex. images in my research have 40X as the highest magnification). 
        :return:  info about accepted patch or None if it is rejected.
        """
        idx = class_list.index(c)
        h, w = class_seeds[idx][i]
       

        tissue_mask_patch = self.tissue_mask[w:w+patch_size, h:h+patch_size]
        if np.sum(tissue_mask_patch) / np.prod(tissue_mask_patch.shape) < anno_threshold:
            return None

        info = {
            'w': w,
            'h': h,
            'parent':self.ID,
            'size': patch_size,
            'mag': mag,
            'class': c,
            'id': i,
            'lvl0': mag0
        }
		
		if c==1:
            pixel=(0,255,0)
        elif c==2:
            pixel=(0,255,0)
        elif c==3:
            pixel=(0,255,0)
        elif c==4:
            pixel=(0,255,0)
        else:
            pixel=(255,255,255)
            	
        # If no annotation we're done.
        if self.annotation is None:
            return patch, info

        annotation_patch = wsi.read_region((w, h), mag, patchsize) 
        annotation_patch = np.asarray(annotation_patch)
        mask = (annotation_patch == pixel)
        if np.sum(mask) / np.prod(mask.shape) < self.anno_threshold:
            return None

        return  info
    
    
    
    
    def _Sample_patches(wsi, level,max_per_class,anno_threshold,ignore_bg, patchesDir,annotation_dir):
    	'''
    	Sample patches from the whole slide image
    	:param max_per_class: maximum number of the sampled pat ches per class.
  		:param anno_threshold: percentage of  accepted tissue to background.
  		:param ignore_bg: True if we want to ignor background during the sampling false otherwise.
  		:param patchesDir: The location where to save the sampledd patches.
  		:param annotation_dir: The location where to save the generated annotation mask.
    	'''
    	size = wsi.level_dimensions[level]
    	filename = os.path.join(annotation_dir, self.ID + '_findTissueMask.png')
        low_res = Image.open(filename,'r')
        low_res_np = np.asarray(low_res)
        low_res_np.setflags(write=1)
        # Convert to labels
        unique = np.unique(low_res_np.reshape(-1, low_res_np.shape[2]), axis=0)
        for i in range(unique.shape[0]):
            pixel = unique[i,:]
            mask = ( low_res_np == pixel).all(axis=2)
            
            if tuple(pixel)==(0,255,0):
            	label=1
            elif tuple(pixel)==(0,0,255):
            	label=2
            elif tuple(pixel)==(255,0,0):
            	label=3
            elif tuple(pixel)==(255,255,0):
            	label=4
            else:
            	label=0
            
            low_res_np[mask] = [label, label, label]
         
    	annotation_low_res = low_res_np[:,:,0]
        classes = sorted(list(np.unique(annotation_low_res)))

        
		class_list=[]
		class_seeds=[]
        for c in classes:
            mask = (annotation_low_res == c)
            nonzero = np.nonzero(mask)
            N = nonzero[0].size
            coordinates = [(int(nonzero[0][i] * factor), int(nonzero[1][i] * factor)) for i in range(N)]
            shuffle(coordinates)
            class_list.append(c)
            class_seeds.append(coordinates)	
        
        #------------------------------------------------   
         
        frame = pd.DataFrame(data=None, columns=['id', 'w', 'h', 'class', 'mag', 'size', 'parent', 'lvl0'])
        if ignore_bg:
            clist = class_list[1:]
        else:
            clist = class_list
        for c in clist:
            index = class_list.index(c)
            seeds = class_seeds[index]
            count = 0
            for j, seed in enumerate(seeds):
                info = self._class_c_patch_i(wsi,class_list,class_seeds, c, j,anno_threshold,patch_size,wsi.level_downsamples[level],40.0)
                if info is not None:
                    frame = frame.append(info, ignore_index=1)
                if isinstance(max_per_class, int):
                    # If not rejected increment count
                    if info is not None:
                        count += 1
                    if count >= (max_per_class - 1):
                        break


        filename = os.path.join(patchesDir, self.ID + '_patchframe.pickle')
        frame.to_pickle(filename)
        