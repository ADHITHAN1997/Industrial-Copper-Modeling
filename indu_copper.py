#import  libraries
import streamlit as st
from streamlit_option_menu import option_menu
import re
import numpy as np
from PIL import Image


#Streamlit part
st.set_page_config(layout="wide")
st.write("""
<div style='text-align:center'>
    <h1 style='color:#FF9900;'>Industrial Copper Modeling Application</h1>
</div>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title="Industrial Copper Modeling Application",
    options = ["🏡Home","🎢Data Prediction","🔚Exit"])
if selected == "🏡Home":
        col1, col2 = st.columns(2)
        with col1:
            st.write("## :orange[**ABOUT**]")
            st.write("### :The project aims to use machine learning models to improve pricing decisions and lead classification for a copper industry client dealing with skewed and noisy sales data. The models will predict selling prices through regression and classify lead status as won or lost through classification. Skills gained include Python, data preprocessing, exploratory data analysis,and optimizing machine learning models for regression and classification.")
            st.write("### :orange[**Predict Selling Price**]")
            st.write("### In this tab, User is able to select the values and predict the selling price based on the given details")
            st.write("### :orange[**Predict Status**]")
            st.write("### In this tab, User is able to select the values and predict the status based on the given details")

        with col2:
            st.image(Image.open("C:/Users/study/Downloads/vscode/copper-mine.webp"),width=500)
            st.image(Image.open("C:/Users/study/Downloads/vscode/copper-market-price-graph.jpg"),width=500)

if selected == "🎢Data Prediction":
    tab1, tab2 = st.tabs(['**_Prediction Selling Price_**', '**_Prediction Status_**'])
    with tab1:
        # Define the possible values for the dropdown menus
        status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised',
                            'Offered', 'Offerable']
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25.,
                                67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
                    '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
                    '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
                    '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
                    '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
            
        # Define the widgets for user input
        with st.form("my_form"):
            col1, col2, col3 = st.columns([5, 2, 5])
            with col1:
                st.write(' ')
                status = st.selectbox("Status", status_options, key=1)
                item_type = st.selectbox("Item Type", item_type_options, key=2)
                country = st.selectbox("Country", sorted(country_options), key=3)
                application = st.selectbox("Application", sorted(application_options), key=4)
                product_ref = st.selectbox("Product Reference", product, key=5)
            with col3:
                st.write(' ')
                st.write(' ')
                st.write(' ')
                st.write(
                    f'<h5 style="color:darkviolet">NOTE: Min & Max given for reference, you can enter any value</h5>',
                    unsafe_allow_html=True)
                quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                width = st.text_input("Enter width (Min:1, Max:2990)")
                customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            st.markdown("""
                            <style>
                            div.stButton > button:first-child {
                                background-color: darkviolet;
                                color: white;
                                width: 100%;
                            }
                            </style>
                        """, unsafe_allow_html=True)


        if submit_button :
            import pickle
            with open(r"C:/Users/study/Downloads/vscode/model.pkl", 'rb') as file:
                loaded_model = pickle.load(file)
            with open(r"C:/Users/study/Downloads/vscode/scaler.pkl", 'rb') as f:
                scaler_loaded = pickle.load(f)

            with open(r"C:/Users/study/Downloads/vscode/t.pkl", 'rb') as f:
                t_loaded = pickle.load(f)

            with open(r"C:/Users/study/Downloads/vscode/s.pkl", 'rb') as f:
                s_loaded = pickle.load(f)

            new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width),
                                    country, float(customer), int(product_ref), item_type, status]])
            new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
            new_sample_ohe2 = s_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, ]], new_sample_ohe, new_sample_ohe2), axis=1)
            new_sample1 = scaler_loaded.transform(new_sample)
            new_pred = loaded_model.predict(new_sample1)[0]
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

    with tab2:
        item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
        country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
        application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25.,
                               67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
        product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
                   '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
                   '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
                   '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
                   '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

        with st.form("my_form1"):
            col1, col2, col3 = st.columns([5, 1, 5])
            with col1:
                cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
                cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
                cwidth = st.text_input("Enter width (Min:1, Max:2990)")
                ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
                cselling = st.text_input("Selling Price (Min:1, Max:100001015)")

            with col3:
                st.write(' ')
                citem_type = st.selectbox("Item Type", item_type_options, key=21)
                ccountry = st.selectbox("Country", sorted(country_options), key=31)
                capplication = st.selectbox("Application", sorted(application_options), key=41)
                cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")
            st.markdown("""<style>
                            div.stButton > button:first-child {
                                background-color: red;
                                color: white;
                                width: 100%;
                            }
                            </style>
                        """, unsafe_allow_html=True)
            

        if csubmit_button :
            import pickle

            with open(r"C:/Users/study/Downloads/vscode/cmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r"C:/Users/study/Downloads/vscode/cscaler.pkl", 'rb') as f:
                cscaler_loaded = pickle.load(f)

            with open(r"C:/Users/study/Downloads/vscode/ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                    np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer),
                                    int(cproduct_ref), citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
            new_sample = cscaler_loaded.transform(new_sample)
            new_pred = cloaded_model.predict(new_sample)
            if new_pred == 1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')

if selected == "🔚Exit":
    st.markdown('<h2 style="color: #FF6666;">OverView</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 18px;">
        By incorporating machine learning in data exploration, preprocessing, regression, and classification, this solution provides a comprehensive approach for the copper industry to improve pricing decisions and lead status assessments. 
        The Streamlit web application is a useful tool that guarantees decision-makers' accessibility and usability, with a focus on the special tasks of <b>Selling Price</b> and <b>Status Lead</b> prediction.
        </div>
        """, 
        unsafe_allow_html=True
    )
    button = st.button("EXIT!")
    if button:
        st.success("**Thank you for utilizing this platform. I hope you have received the predicted price and status for your copper industry😊**")
