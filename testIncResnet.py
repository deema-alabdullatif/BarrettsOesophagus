from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D,normalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
#from keras.utils import multi_gpu_model
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import scipy.misc
import csv

g1=[0,255,0]
g3=[0,0,255]
g5=[255,0,0]

epoch_num=2
bs=1
output_path2='/nobackup/alabdullatif/results_DEC/'
classes=3

# create the base pre-trained model

# this is the model we will train
base_model = load_model('/nobackup/alabdullatif/supervisedinceptionresnetv2/temp2/stage2finetune2.h5')
test_im=['10586','10790','10829','10857','11013','11014','11035','11040','11054','11063','13083','13154','13239','13303','13348']
#test_im=['10790','10829','10857','11013','11014','11035','11040','11054','11063','13083','13154','13239','13303','13348']
for i in range(len(test_im)):
	train_datagen= ImageDataGenerator(preprocessing_function=preprocess_input) 
	path='/nobackup/alabdullatif/l40roi_patches/'+test_im[i]+'/'
	test_generator = train_datagen.flow_from_directory(directory=path,color_mode="rgb",batch_size=16,class_mode=None,shuffle=False,seed=42)	
	STEP_SIZE_VAL=test_generator.n//test_generator.batch_size	
	test_generator.reset()
	pred=base_model.predict_generator(test_generator, steps=STEP_SIZE_VAL, verbose=0)
	pre=np.argmax(pred,axis=1)
	filenames=test_generator.filenames
	out_file = open(output_path2+test_im[i]+'.csv','w')
	predictions_format_str = ('%d	%s	%s	%s\n')
	for q in range(len(filenames)):
		
		if pre[q] == 0:
			x='G1'
		elif pre[q] == 1:
			x='G3'
		else:
			x='G5'
		out_file.write(predictions_format_str%(q+1, filenames[q], x,pred[q,pre[q]]))
		out_file.flush()
	out_file.close()            
	'''
	cvfile = open(output_path2+'predict/'+test_im[i]+'.csv','r')
	csv_reader = csv.reader(cvfile)
	pt=np.load('/nobackup/alabdullatif/testL9Part/results/needed/'+test_im[i]+'_p.npy')
	rr=np.load('/nobackup/alabdullatif/testL9Part/results/needed/'+test_im[i]+'_r.npy')
	cc=np.load('/nobackup/alabdullatif/testL9Part/results/needed/'+test_im[i]+'_c.npy')
	ref_factor=np.load('/nobackup/alabdullatif/testL9Part/results/needed/'+test_im[i]+'_ref.npy')
	mask=np.load('/nobackup/alabdullatif/testL9Part/results/needed/'+test_im[i]+'_mask.npy')
	p2=[0]*len(pt)
	l=[0]*len(pt)

             
	for line in csv_reader:
		index=int(line[0].split('_class')[0].split('p')[1])
		temp5=line[0].split('\t')
		p2[index]=temp5[1]
		l[index]=temp5[3]

	cvfile.close()
	for s in range(len(rr)):
		if l[s]=='G1':
			code=g1
		elif l[s]=='G3':
			code=g3
		else:
			code=g5
		xx=int(rr[s]*ref_factor)
		yy=int(cc[s]*ref_factor)
		mask[xx:xx+int(256*ref_factor)+1, yy:yy+int(256*ref_factor)+1, :] = code
	filename = '/nobackup/alabdullatif/testL9Part/results/mask/'+image+'_predictionMask.png'
	scipy.misc.toimage(mask, cmin=0.0, cmax=255.0).save(filename)
	'''