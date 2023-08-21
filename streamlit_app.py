import streamlit as st
from PIL import Image
import ultralytics
from ultralytics import YOLO

st.title('Aloe Vera Plant Disease Analyser')

col1,col2=st.columns(2)
upload_stat=0
with col1:
    
    input_image=st.file_uploader('Upload The Image File Here',type=['jpg','png'])
    
    confirmation=st.button('Submit')
    if confirmation==True:
        st.image(input_image)
        upload_stat=1
        
with col2:
    st.subheader('Detection:')
    if upload_stat==1:
        image_new=Image.open(input_image)
        model=YOLO('best_new_v2.pt')
        results= model(image_new)

        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            st.image(im)  # show image

    
        
