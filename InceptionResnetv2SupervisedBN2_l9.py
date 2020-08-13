from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D,normalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
#from keras.utils import multi_gpu_model
from keras.optimizers import Adam


from keras.preprocessing.image import ImageDataGenerator




epoch_num=10
bs=128
features_out=768

output_path2='/nobackup/alabdullatif/supervisedinceptionresnetv2_Level9/out5/'
classes=3

# create the base pre-trained model

# this is the model we will train
base_model = load_model('/nobackup/alabdullatif/supervisedinceptionresnetv2_Level9/out4/stage2finetune2e2_l9.h5')

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


es=EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, 
	mode='auto')
lr=ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=1, verbose=1, 
	mode='auto')

train_datagen= ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	horizontal_flip=True) 
    	
val_datagen= ImageDataGenerator(preprocessing_function=preprocess_input) 
    	
train_generator = train_datagen.flow_from_directory(
    	directory=r"/nobackup/alabdullatif/level9_data/validate/",
    	color_mode="rgb",
    	batch_size=bs,
    	class_mode="categorical",
    	shuffle=True,
    	seed=42)
    	
val_generator = val_datagen.flow_from_directory(
    	directory=r"/nobackup/alabdullatif/level9_data/train/",
    	color_mode="rgb",
    	batch_size=bs,
    	class_mode="categorical",
    	seed=42)
    	
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size 
STEP_SIZE_VAL=val_generator.n//val_generator.batch_size	

trainFlag=False
for layer in model.layers:
	if layer.name=='block8_9_mixed':
		trainFlag=True
		
	
	layer.trainable = trainFlag



# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

model.compile(optimizer=Adam(lr=0.000001), loss='categorical_crossentropy', metrics=['accuracy'])
#cp=ModelCheckpoint(output_path2, monitor='val_loss', verbose=0, 
	#save_best_only=False, save_weights_only=False, mode='auto', period=1)
tb=TensorBoard(log_dir=output_path2+'/log', histogram_freq=0, batch_size=bs, 
	write_graph=True, write_grads=False, write_images=False)
# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(train_generator, steps_per_epoch=10, epochs=3, verbose=1, 
	#callbacks=[tb,es,lr], validation_data=val_generator, validation_steps=1, shuffle=True)
	

	
model.fit_generator(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=epoch_num, verbose=1, 
	callbacks=[tb,es,lr], validation_data=val_generator, validation_steps=50, shuffle=True)
model.save(output_path2+'stage2finetune2e2_l9val.h5')
