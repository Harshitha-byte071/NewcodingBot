       =====================IMPORTPACKAGES=====================
import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn import linear_model 
import warnings 
warnings.filterwarnings('ignore') 

===========================DATASELECTION====================

dataframe=pd.read_csv("Bot-Iot.csv") 
print("-------------------------------------------") 
print(" DATA SELECTION ") 
print("-------------------------------------------") 
print() 
print(dataframe.head(15))

#==========================PREPROCESSING=====================

#==== checking missing values ==== 
print("------------------------------------------") 
print(" CHECKING MISSING VALUES ") 
print("------------------------------------------") 
print() 
print(dataframe.isnull().sum()) 
#========= label encoding =========== 
print("------------------------------------------") 
print(" BEFORE LABEL ENCODING ")
print("------------------------------------------") 
print() 
print(dataframe['category'].head(20))label_encoder = preprocessing.LabelEncoder() 
#dataframe['category'] = label_encoder.fit_transform(dataframe['category']) 
#dataframe['proto'] = label_encoder.fit_transform(dataframe['proto']) 
dataframe = dataframe.astype(str).apply(label_encoder.fit_transform) 
print("------------------------------------------") 
print(" AFTER LABEL ENCODING ") 
print("------------------------------------------") 
print()
print(dataframe['category'].head(20)) 
#====== STANDARD SCALAR ====== 
from sklearn.preprocessing import StandardScaler 
scalar = StandardScaler() 
df_scaled = pd.DataFrame(scalar.fit_transform(dataframe), 
columns=dataframe.columns) 
#==== SPLIT X AND Y ==== 
X=dataframe[['pkSeqID','proto','proto_number','pkts','seq','attack']] 
y=dataframe['category'] 
#==== MINMAX SCALAR ======== 
from sklearn.preprocessing import MinMaxScaler 
X_scaled = MinMaxScaler().fit_transform(X)
X_normal_scaled = X_scaled[y == 0] 
X_abnormal_scaled = X_scaled[y == 1] 

#==========================DATASPLITTING=====================

#X=dataframe[['pkSeqID','proto','proto_number','pkts','seq','category']]
#y=dataframe['attack']X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.3, 
random_state=0) 
print("----------------------------------") 
print(" DATA SPLITTING ") 
print("----------------------------------") 
print() 
print("Total no of data :",dataframe.shape[0]) 
print("Total no of test data :",X_test.shape[0]) 
print("Total no of train data :",X_train.shape[0]) 

#======================= FEATURE EXTRACTION ================= 

from sklearn.decomposition import PCA
pca = PCA(n_components = 4) 
X_train1 = pca.fit_transform(X)print("------------------------------------------") 
print(" PRINCIPLE COMPONENT ANALYSIS ") 
print("------------------------------------------") 
print() 
print(" The original features is :", X.shape[1]) 
print() 
print(" The reduced feature is :",X_train1.shape[1 ] ) 
print() 

#===========================AUTOENCODER=====================

from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Input,Dense 
from tensorflow.keras import regularizers 
# ==== INPUT LAYER ==== 
input_layer = Input(shape =(X.shape[1], )) 
# ===== ENCODE THE DATA ====encoded = Dense(100, activation 
='tanh',activity_regularizer = regularizers.l1(10e-5))(input_layer) 
encoded = Dense(50, activation ='tanh',activity_regularizer = 
regularizers.l1(10e-5))(encoded) 
encoded = Dense(25, activation ='tanh',activity_regularizer = 
regularizers.l1(10e5))(encoded) 
encoded = Dense(12, activation ='tanh',activity_regularizer = 
regularizers.l1(10e-5))(encoded) 
encoded = Dense(6, activation ='relu')(encoded) 
# ===== DECODE THE DATA ===== 
decoded = Dense(12, activation ='tanh')(encoded) 
decoded = Dense(25, activation ='tanh')(decoded) 
decoded = Dense(50, activation ='tanh')(decoded)
decoded = Dense(100, activation ='tanh')(decoded) 
# ==== OUTPUT LAYER ==== 
output_layer = Dense(X.shape[1], activation ='relu')(decoded) 
autoencoder = Model(input_layer, output_layer) 
autoencoder.compile(optimizer ="adadelta", loss ="mse") 
print("--------------------------------------")
print(" AUTO ENCODER ") 
print("--------------------------------------") 
print() 
import numpy as np 
==== FITTING THE MODEL ===== 
autoencoder.fit(X_normal_scaled, X_normal_scaled, batch_size = 16, epochs = 
10, shuffle = True, validation_split = 0.20) 

#=========================== CLASSIFICATION =============== 

# === LONG SHORT TERM MEMORY === 
x=np.expand_dims(X_train, axis=2) 
Y=np.expand_dims(y_train,axis=1) 
from tensorflow.keras.layers import Input,Dense 
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras import regularizersfrom sklearn.preprocessing import 
MinMaxScaler 
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import LSTM
model = Sequential() 
model.add(LSTM(input_shape=(6,1), kernel_initializer="uniform", 
return_sequences=True, stateful=False, units=50)) 
model.add(Dropout(0.2)) 
model.add(LSTM(5, kernel_initializer="uniform", 
activation='relu',return_sequences=False)) 
model.add(Dropout(0.2)) 
model.add(Dense(3,kernel_initializer="uniform",activation='relu')) 
model.add(Dense(1, activation='linear')) 
model.compile(loss="binary_crossentropy", 
optimizer='adam',metrics=['accuracy','mae']) 
model.summary() 
print("---------------------------------------------------------") 
print(" LONG SHORT TERM MEMORY ----> LSTM ") 
print("---------------------------------------------------------")
print() 
His=model.fit(x, Y, epochs = 5, batch_size=2, verbose = 2) 
acc_lstm =model.evaluate(x, Y, verbose=2)[1]
acc_lstm=acc_lstm*100 
print("--------------------------------------") 
print("PERFORMANCE ANALYSIS ---> LSTM") 
print("-------------------------------------") 
print() 
print("1. ACCURACY =",acc_lstm,'%' ) 
pred_lstm = model.predict([x]) 
y_pred2 = pred_lstm.reshape(-1) 
y_pred2[y_pred2<0.5] = 0 
y_pred2[y_pred2>=0.5] = 1
y_pred2 = y_pred2.astype('int')cm = metrics.confusion_matrix(y_pred2,Y) 
TP=cm[0][0] 
TN=cm[0][1] 
FP=cm[1][0]
FN=cm[0][1] 
Total=(TP+TN+FP+FN) 
Pre_lstm=TP/(TP+FP)*100 
Sen_lstm=TP/(TP+FN)*100 
f1_lstm=(2*Pre_lstm*Sen_lstm)/(Pre_lstm+Sen_lstm) 
print() 
print("2. PRECISION =",Pre_lstm,'%' ) 
print() 
print("3. SENSITIVITY =",Sen_lstm,'%' ) 
print() 
print("3. F1 SCORE =",f1_lstm,'%' )

# === CONVOLUTIONAL NEURAL NETWORK === 

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Input 
inp = Input(shape=(6,1))
conv = Conv1D(filters=2, kernel_size=2)(inp) 
pool = MaxPool1D(pool_size=2)(conv) 
flat = Flatten()(pool) 
dense = Dense(1)(flat) 
model1 = Model(inp, dense) 
model1.compile(loss='binary_crossentropy', 
optimizer='adam',metrics=['accuracy']) 
print(model.summary()) 
print("------------------------------------------") 
print(" CONVOLUTIONAL NEURAL NETWORK ") 
print("------------------------------------------") 
print()His1=model1.fit(x, Y, epochs = 5, batch_size=2, verbose = 2) 
acc_cnn =model.evaluate(x, Y, verbose=2)[1] 
acc_cnn=acc_cnn*100 
print("--------------------------------------") 
print("PERFORMANCE ANALYSIS ---> CNN") 
print("-------------------------------------") 
print() 
print("1. ACCURACY =",acc_cnn,'%' ) 
pred_cnn = model.predict([x]) 
y_pred2 = pred_cnn.reshape(-1) 
y_pred2[y_pred2<0.5] = 0 
y_pred2[y_pred2>=0.5] = 1 
y_pred2 = y_pred2.astype('int') 
cm = metrics.confusion_matrix(y_pred2,Y)TP=cm[0][0] 
TN=cm[0][1]
FP=cm[1][0] 
FN=cm[0][1]
Total=(TP+TN+FP+FN) 
Pre_cnn=TP/(TP+FP)*100 
Sen_cnn=TP/(TP+FN)*100 
f1_cnn=(2*Pre_cnn*Sen_cnn)/(Pre_cnn+Sen_cnn) 
print() 
print("2. PRECISION =",Pre_cnn,'%' ) 
print() 
print("3. SENSITIVITY =",Sen_cnn,'%' ) 
print() 
print("3. F1 SCORE =",f1_cnn,'%' )

========================== PREDICTION ===========

print("--------------------------------------") 
print("PREDICTION") 
print("-------------------------------------") 
print() 
for i in range(0,10): 
if y_pred2[i]== 0: 
print("============================") 
print() 
print([i],' DDOS ATTACK ') 
else: 
print("============================") 
print() 
print([i],'MALWARE ATTACK ') 




