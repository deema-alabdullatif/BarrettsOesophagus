import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D,normalization
from keras import backend as K
#from keras.engine.network import Network
import matplotlib.pyplot as plt
#from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

epoch_num=2
bst=126
bsv=181
features_out=768
output_path='/nobackup/alabdullatif/OneClassAnomaly_outputs/'
classes=2
bs=126

def original_loss(y_true, y_pred):
	lc = 1/(classes*bs) * bs**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((bs-1)**2)
	return lc 

def main():
	
	
	base_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
	
	# PATCH MOMENTUM - START
	import json
	conf = json.loads(base_model.to_json())
	for l in conf['config']['layers']:
		if l['class_name'] == 'BatchNormalization':
			l['config']['momentum'] = 0.5


	m = Model.from_config(conf['config'])
	for l in base_model.layers:
		m.get_layer(l.name).set_weights(l.get_weights())

	base_model = m
	
	# PATCH MOMENTUM - END
	
	tflag=False
	for layer in base_model.layers:
		#if layer.name == "block8_9_mixed":
			#tflag=True
		layer.trainable = tflag

	x = Dense(features_out, activation='relu')(base_model.output)
	
	model_t = Model(inputs=base_model.input, outputs=x)
	#model_r = Network(inputs=model_t.input, outputs=model_t.output,name="shared_layer")
                      
	predictions = Dense(classes, activation='softmax')(model_t.output)
	#model_r = multi_gpu_model(Model(inputs=model_r.input,outputs=predictions))
	#model_r = multi_gpu_model(Model(inputs=model_t.input,outputs=predictions),gpus=4)
	model_r = Model(inputs=model_t.input,outputs=predictions)
	optimizer = Adam(lr=5e-5)
	model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
	model_t.compile(optimizer=optimizer, loss=original_loss)
	#model_t.summary()
	#model_r.summary()



	
	
	print('load train1')
	train_datagen= ImageDataGenerator(preprocessing_function=preprocess_input,
    	rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	horizontal_flip=True) 
	train_generator = train_datagen.flow_from_directory(
    	directory=r"/nobackup/alabdullatif/barretts/patches/SVDD1/TRAIN_DIR/",
    	color_mode="rgb",
    	batch_size=bst,
    	class_mode="categorical",
    	shuffle=True,
    	seed=42)
	print('load val1')
	val_datagen= ImageDataGenerator(preprocessing_function=preprocess_input,
    	rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	horizontal_flip=True) 
	val_generator = val_datagen.flow_from_directory(
    	directory=r"/nobackup/alabdullatif/barretts/patches/SVDD1/VALIDATION_DIR/",
    	color_mode="rgb",
    	batch_size=bsv,
    	class_mode="categorical",
    	seed=42)
	print('load train2')	
	trainR_datagen= ImageDataGenerator(preprocessing_function=preprocess_input,
    	rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	horizontal_flip=True) 
	trainR_generator = trainR_datagen.flow_from_directory(
    	directory=r"/nobackup/alabdullatif/pcam/train/",
    	target_size=(256, 256),
    	color_mode="rgb",
    	batch_size=bst,
    	class_mode="categorical",
    	shuffle=True,
    	seed=42)
	print('load val2')
	valR_datagen= ImageDataGenerator(preprocessing_function=preprocess_input,
    	rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	horizontal_flip=True) 
	valR_generator = valR_datagen.flow_from_directory(
    	directory=r"/nobackup/alabdullatif/pcam/val/",
    	target_size=(256, 256),
    	color_mode="rgb",
    	batch_size=bsv,
    	class_mode="categorical",
    	seed=42)
    	
	STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
	STEP_SIZE_VAL=val_generator.n//val_generator.batch_size
	STEP_SIZE_TRAINR=trainR_generator.n//trainR_generator.batch_size
	STEP_SIZE_VALR=valR_generator.n//valR_generator.batch_size
	
	loss, loss_c, valloss, valloss_c = [], [], [], []
	print('--------------Training------------')
	
	for epochnumber in range(epoch_num):
		lc, ld, lcVal, ldVal = [], [], [], []
		print("epoch:",epochnumber+1)
		counter=0
		
		for [x, y],[xR,yR] in zip(train_generator,trainR_generator):
			bs=126
			if (counter > STEP_SIZE_TRAIN):
				break
			lc.append(model_t.train_on_batch(x, np.zeros((bs, features_out))))
			ld.append(model_r.train_on_batch(xR, yR))
			counter+=1
			print('network t and r train losses are '+str(np.mean(lc))+' and '+str(np.mean(ld)))
		
		# Store the model on disk
		model_r.save(output_path+str(epochnumber)+'modelr.h5')
		model_t.save(output_path+str(epochnumber)+'modelt.h5')	
		loss.append(np.mean(ld))
		loss_c.append(np.mean(lc))	
		print("Descriptive Loss Training:", loss[-1])
		print("Compact Loss Training", loss_c[-1])
        
		counter=0
		for [xv, yv],[xRv,yRv] in zip(val_generator,valR_generator):
			bs=181
			if (counter > STEP_SIZE_VAL):
				break
			lcVal.append(model_t.test_on_batch(xv, np.zeros((bs, features_out))))
			ldVal.append(model_r.test_on_batch(xRv, yRv))
			counter+=1
			print('network t and r validation losses are '+str(np.mean(lcVal))+' and '+str(np.mean(ldVal)))
			
		valloss.append(np.mean(ldVal))
		valloss_c.append(np.mean(lcVal))
		print("Descriptive Loss Validation:", valloss[-1])
		print("Compact Loss Validation", valloss_c[-1])

		
	
		np.save(output_path+'trainDL.npy',np.asarray(loss))
		np.save(output_path+'valDL.npy',np.asarray(valloss))
		np.save(output_path+'trainCL.npy',np.asarray(loss_c))
		np.save(output_path+'valCL.npy',np.asarray(valloss_c))

	
if __name__ == '__main__':
	main()    	