import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# App title
st.title('Maize Leaf Diseases Identifier Web App')

## APP info    
st.write('''

**This Streamlit App utilizes a Deep Learning model to detect diseases(Northern Leaf Blight,Common Rust,Gray Leaf Spot) that attact the corn leaves, based in digital images.**
''')

st.text("Upload an image of a maize leaf to classify its disease.")

## load file
st.sidebar.write("# File Required")
uploaded_image = st.sidebar.file_uploader('', type=['jpg','png','jpeg'])

################### Class Dict and Dataframe of Probabilites #############################
# Map class
map_class = {
        0:'Northern Leaf Blight',
        1:'Common Rust',
        2:'Gray Leaf Spot',
        3:'Healthy'
        }
        
#Dataframe 
dict_class = {
        'Corn Leaf Condition': ['Northern Leaf Blight', 'Common Rust','Gray Leaf Spot','Healthy'],
        'Confiance': [0,0,0,0]
        }
        
df_results = pd.DataFrame(dict_class, columns = ['Corn Leaf Condition', 'Confiance'])
    
def predictions(preds):
    df_results.loc[df_results['Corn Leaf Condition'].index[0], 'Confiance'] = preds[0][0]
    df_results.loc[df_results['Corn Leaf Condition'].index[1], 'Confiance'] = preds[0][1]
    df_results.loc[df_results['Corn Leaf Condition'].index[2], 'Confiance'] = preds[0][2]
    df_results.loc[df_results['Corn Leaf Condition'].index[3], 'Confiance'] = preds[0][3]

    return (df_results)          

########################################### Load the model #########################
#@st.cache
def get_model():

    model = tf.keras.models.load_model("model_accuracy98%(2).h5")
    #model = tf.keras.models.load_model("model/model_accuracy98%(1).h5")
    #model = tf.keras.models.load_model("model/modelwofe.h5")
    #model = tf.keras.models.load_model("model/modelwofe(1).h5")
    #model = tf.keras.models.load_model("model/modelwofe(2).h5")
    #model = tf.keras.models.load_model("model/modelwofe(3).h5")
    return model

if __name__=='__main__':
    
    # Model
    model = get_model()
    
    # Image preprocessing
    if not uploaded_image:
        st.sidebar.write('Please upload an image before preceeding!')
        st.stop()
    else:
        # Decode the image and Predict the class
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)# Encoding image
        st.write("## Maize Leaf Image")
        st.image(image, use_column_width= True) # Display the image
        #img = tf.io.decode_image(image, channels = 3) # Convert image to tensor
        img = tf.image.resize(image,(256,256)) # Resize the image
        #img_arr = tf.keras.preprocessing.image.img_to_array(img) # Convert image to array
        img_arr = tf.expand_dims(img, 0) # Create a bacth

    Genrate_pred = st.button("Detect Result") 
 
    if Genrate_pred:
        st.subheader('Probabilities by Class') 
        preds = model.predict(img_arr)
        preds_class = model.predict(img_arr).argmax()

        st.dataframe(predictions(preds))

        if (map_class[preds_class]=="Northern Leaf Blight") or (map_class[preds_class]=="Common Rust") or (map_class[preds_class]=="Gray Leaf Spot"): 
            st.subheader("The Corn Leaf is infected by {} disease".format(map_class[preds_class]))

        else:
            st.subheader("The Corn Leaf is {}".format(map_class[preds_class]))
