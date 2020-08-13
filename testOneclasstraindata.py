from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D,normalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
import pandas as pd

from keras.preprocessing.image import ImageDataGenerator




epoch_num=1
bs=128
features_out=768
output_path2='/nobackup/alabdullatif/10790/'
classes=3
def original_loss(y_true, y_pred):
	lc = 1/(classes*bs) * bs**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((bs-1)**2)
	return lc 


#base_model = load_model('/nobackup/alabdullatif/OneClassAnomaly_outputs/epoch1/0modelr.h5')
#base_model.layers.pop()
base_model = load_model('/nobackup/alabdullatif/OneClassAnomaly_outputs/epoch1/0modelt.h5',custom_objects={'original_loss': original_loss})



# PATCH MOMENTUM - START
import json
conf = json.loads(base_model.to_json())
for l in conf['config']['layers']:
    if l['class_name'] == 'BatchNormalization':
        l['config']['momentum'] = 0.5


m = Model.from_config(conf['config'])
for l in base_model.layers:
    m.get_layer(l.name).set_weights(l.get_weights())

model = m
# PATCH MOMENTUM - END


train_datagen= ImageDataGenerator(preprocessing_function=preprocess_input) 
    	

    	

test_generator = train_datagen.flow_from_directory(
    	directory=r"/nobackup/alabdullatif/barretts/patches/SVDD1/TRAIN_DIR/",
    	color_mode="rgb",
    	batch_size=128,
    	class_mode=None,
    	shuffle=False,
    	seed=42)    	

    	
STEP_SIZE_VAL=test_generator.n//test_generator.batch_size	

test_generator.reset()
pred=model.predict_generator(test_generator, steps=STEP_SIZE_VAL, verbose=1)

np.save(output_path2+'predicttrainset.npy',pred)
filenames=test_generator.filenames



out_file = open(output_path2+"outfilename2trainset.csv",'w')

predictions_format_str = ('%d	%s	%s\n')


for i in range(len(filenames)):
	x='G1'
	if filenames[i].split('class_')[1].split('_from')[0] == '3':
		x='G3'
	elif filenames[i].split('class_')[1].split('_from')[0] == '7':
		x='G5'
	else:
		x='G1'
		
	out_file.write(predictions_format_str%(i+1, filenames[i], x))
	out_file.flush()
out_file.close()            
