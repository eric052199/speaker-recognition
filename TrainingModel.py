import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn import mixture
from featureextraction import extract_features
#from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")

#path to training da
# source   = "development_set/"
source   = "Voice_Samples_Training/"   
#source = "clean/"

#path where training speakers will be saved

# dest = "speaker_models/"
# train_file = "development_set_enroll.txt"

dest = "Trained_Speech_Models/"
train_file = "Voice_Samples_Training_Path.txt"        
file_paths = open(train_file,'r')

count = 1
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:    
    #print(path)
    path = path.strip()   
    #print (path)
    
    # read the audio
    sr,audio = read(source+path)
    
    # extract 40 dimensional MFCC & delta MFCC features
    vector   = extract_features(audio,sr)
    
    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 5 files of speaker are concatenated, then do model training
	# -> if count == 5: --> edited below
    if count == 30:
        print(path)    
        gmm = mixture.GaussianMixture(n_components = 105, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        # dumping the trained gaussian model
        picklefile = path.split("/")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)   
        features = np.asarray(())
        count = 0
    count = count + 1