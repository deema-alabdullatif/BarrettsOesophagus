import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras import backend as K
#from keras.engine.network import Network
import matplotlib.pyplot as plt
#from keras.utils import multi_gpu_model
from keras.optimizers import SGD


epoch_num=2
bs=128
features_out=768
output_path='/nobackup/alabdullatif/OneClassAnomaly_outputs/'
classes=2

def original_loss(y_true, y_pred):
lc = 1/(classes*bs) * bs**2 * K.sum((y_pred -K.mean(y_pred,axis=0))**2,axis=[1]) / ((bs-1)**2)
return lc 

def main():


base_model = load_model('/nobackup/alabdullatif/supervisedinceptionresnetv2/temp2/stage2finetune2.h5')
#to remove prediction layer
base_model.pop()

# PATCH MOMENTUM - START
import json
conf = json.loads(base_model.to_json())
for l in conf['config']['layers']:
 if l['class_name'] == 'BatchNormalization':
  l['config']['momentum'] = 0.5


m = Model.from_config(conf['config'])
for l in base_model.layers:
 m.get_layer(l.name).set_weights(l.get_weights())

base_model=m
# PATCH MOMENTUM - END

#x = Dense(features_out, activation='relu')(m.output)

tflag=False
for layer in base_model.layers:
 if layer.name == "block8_9_mixed":
  tflag=True
 layer.trainable = tflag
 


input_t=Input(shape=(256,256,3),name='dataT')
input_R=Input(shape=(256,256,3),name='dataR')

output_t=base_model(input_t)
output_r=base_model(input_r)
#model_t = Model(inputs=base_model.input, outputs=x)

                      
predictions_r = Dense(2, activation='softmax')(output_r)
#model_r = multi_gpu_model(Model(inputs=model_r.input,outputs=predictions))
#model_r = multi_gpu_model(Model(inputs=model_t.input,outputs=predictions),gpus=4)
model_r = Model(inputs=model_t.input,outputs=predictions)
optimizer = SGD(lr=5e-5, decay=0.00005)
model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
model_t.compile(optimizer=optimizer, loss=original_loss)
model_t.summary()
model_r.summary()




print('load train1')
train_datagen= ImageDataGenerator(preprocessing_function=preprocess_input,
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True) 
train_generator = train_datagen.flow_from_directory(
     directory=r"/nobackup/alabdullatif/barretts/patches/SVDD1/TRAIN_DIR/",
     color_mode="rgb",
     batch_size=bs,
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
     batch_size=bs,
     class_mode="categorical",
     shuffle=True,
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
     batch_size=bs,
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
     batch_size=bs,
     class_mode="categorical",
     shuffle=True,
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
  if (counter > STEP_SIZE_TRAIN):
   break
  lc.append(model_t.train_on_batch(x, np.zeros((bs, features_out))))
  ld.append(model_r.train_on_batch(xR, yR))
  counter+=1
  print('network t and r train losses are '+str(np.mean(lc))+' and '+str(np.mean(ld)))
  
 loss.append(np.mean(ld))
 loss_c.append(np.mean(lc)) 
 print("Descriptive Loss Training:", loss[-1])
 print("Compact Loss Training", loss_c[-1])
        
 counter=0
 for [xv, yv],[xRv,yRv] in zip(val_generator,valR_generator):
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

 # Store the model on disk
 model.save(output_path+str(epochnumber+1)+'_oneClassModel.h5')

 np.save(output_path+'trainDL.npy',np.asarray(loss))
 np.save(output_path+'valDL.npy',np.asarray(valloss))
 np.save(output_path+'trainCL.npy',np.asarray(loss_c))
 np.save(output_path+'valCL.npy',np.asarray(valloss_c))
 
plt.plot(loss,label="Training Descriptive Loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig(output_path+'trainDL.png')
  
plt.plot(valloss,label="Validation Descriptive Loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig(output_path+'valDL.png')
  
plt.plot(loss_c,label="Training Compact Loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig(output_path+'trainCL.png')
  
plt.plot(valloss_c,label="Validation Compact Loss")
plt.xlabel("epoch")
plt.legend()
plt.savefig(output_path+'valCL.png')
################################### 

if __name__ == '__main__':
main()     
     
    #from keras.callbacks import EarlyStopping, ModelCheckpoint
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
#model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
#keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
