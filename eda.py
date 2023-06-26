# importing packages that are useful
# streamlit is used for web application development
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # setting our web page layout as wide view

    st.set_page_config(layout="wide")
    # grouping tasks for the sidebar
    EDA_tasks = ["1.distinguish attributes", "2.data cleaning", "3.plots", "4.relationship analysis"]
    # choosing from the sidebar
    choice = st.sidebar.radio("select tasks:", EDA_tasks)
    # setting our file input format types
    file_format = st.radio('Select file format:', ('csv', 'excel'), key='file_format')
    # setting a file uploader in our web application
    data = st.file_uploader("UPLOAD A DATASET 	:open_file_folder: ")


    if data:
        if file_format == 'csv':
            df = pd.read_csv(data)
        else:
            df = pd.read_excel(data)
        st.dataframe(df.head())

    if 'my_dframe1' not in st.session_state:
        st.session_state.my_dframe1 = pd.DataFrame()
    if 'my_dframe2' not in st.session_state:
        st.session_state.my_dframe2 = pd.DataFrame()
    if 'my_dframe3' not in st.session_state:
        st.session_state.my_dframe3 = pd.DataFrame()

    # if user picks "distinguish attributes" then
    if choice == '1.distinguish attributes':
        # assigning heading to the choice
        st.subheader(" Distinguishing attributes in EDA :1234:")
        da_tasks = ("Show Shape","Show Columns","Summary","Show Selected Columns","show numerical variables","show categorical variables","percentage distribution of unique values in fields")
        da_options = st.sidebar.selectbox("Distinguishing attributes in EDA", da_tasks)
        # creating a checkbox to display shape(rows and columns) in the data
        if da_options == "Show Shape":
            st.subheader("Show Shape")
            if data is not None:
                st.write("rows and columns formate ", df.shape)

        # creating a checkbox to display  columns in the data
        if da_options == "Show Columns":
            st.subheader("Show Columns")
            all_columns = df.columns.to_list()
            st.write(all_columns)

        # creating a checkbox to display  summary of the data
        if da_options == "Summary":
            st.subheader("Summary")
            st.write(df.describe())

        # creating a checkbox to display  selected columns in the data
        if da_options == "Show Selected Columns":
            st.subheader("Show Selected Columns")
            all_columns = df.columns.to_list()
            selected_columns = st.multiselect("Select Columns", all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)
        # creating a checkbox to display  only numerical columns in the data
        if da_options == "show numerical variables":
            st.subheader("Show numerical variables")
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            newdf = df.select_dtypes(include=numerics)
            st.dataframe(newdf)

        # creating a checkbox to display  only categorical columns in the data
        if da_options == "show categorical variables":
            st.subheader("Show categorical variables")
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            newdf = df.select_dtypes(include=numerics)
            df1 = df.drop(newdf, axis=1)
            st.dataframe(df1)

        # creating a checkbox to display  percentage of unique values in as selected column in the data
        if da_options == "percentage distribution of unique values in fields":
            st.subheader("percentage distribution of unique values in fields")
            all_columns = df.columns.to_list()
            sel_cols = st.multiselect("Select Columns", all_columns)
            cd = df[sel_cols].value_counts(normalize=True) * 100
            st.dataframe(cd)
    # if user picks "data cleaning" then
    elif choice == '2.data cleaning':
        st.subheader(" Data cleaning in EDA 🛠️")
        eda_tasks = ("Show the NA values", "Fill the NA values", "Remove duplicate values", "Detect outliers", "Normalisation","Standardisation")
        options = st.sidebar.selectbox("Data cleaning", eda_tasks)
        
        # creating a checkbox to display all the null values present in the data
        if options == "Show the NA values":
            st.subheader("Show the NA values")
            nas = df.isnull().sum()
            st.dataframe(nas)


        # creating a checkbox to fill the missing values with median

        if options == "Fill the NA values":
            st.subheader("Fill the NA values")
            task = ("fill with median", "fill with mean", "fill with zeroes")
            option = st.selectbox("fill the NA values", task)

            if option == 'fill with median':
                all_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                sens = st.multiselect("Select Columns", all_columns)
                selos = df[sens]
                dfmed = df.fillna(selos.median())
                st.dataframe(dfmed)
                st.session_state.my_dframe1 = dfmed
                    # creating a checkbox to fill the missing values with buttermilk
            if option == 'fill with mean':
                st.subheader("Fill with mean")
                all_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                sen = st.multiselect("Select Columns", all_columns)
                selo = df[sen]
                dfmean = st.session_state.my_dframe1.fillna(abs(selo.mean()))
                st.dataframe(dfmean)
                st.session_state.my_dframe2 = dfmean

                    # creating a checkbox to fill the missing values with zeroes
            if option == 'fill with zeroes':
                st.subheader("Fill with zeros")
                all_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                sen = st.multiselect("Select Columns", all_columns)
                se = df[sen]
                dfz = st.session_state.my_dframe2.fillna(0)
                st.dataframe(dfz)
                st.download_button(label='download CSV', data=dfz.to_csv(), mime='text/csv')
                st.session_state.my_dframe3 = dfz

        if options == "Remove duplicate values":
            st.subheader("Remove duplicate values")
            all_columns = st.session_state.my_dframe3.columns.to_list()
            sel_cols = st.multiselect("Select Columns", all_columns)
            dfb = st.session_state.my_dframe3.drop_duplicates(subset=sel_cols)
            st.dataframe(dfb)
            st.download_button(label='Download CSV', data=dfb.to_csv(), mime='text/csv')

        # creating a checkbox to detect the outliers present in the data
        if options == "Detect outliers":
            st.subheader("Detect outliers")
            d_task = ("z-score","IQR-score")
            d_option = st.selectbox("Detect outliers",d_task)
            # creating a checkbox to detect the outliers present in the data by z-score method
            if d_option == "z-score":
                st.subheader("By Z-score")

                all_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                sel_mns = st.multiselect("Select Columns", all_columns)
                sel_nms = st.session_state.my_dframe3[sel_mns]
                mean = np.mean(sel_nms)
                sd = np.std(sel_nms)
                upper = (mean + 3 * sd)
                lower = (mean - 3 * sd)
                upper_arr = np.extract(sel_nms > upper, sel_nms)
                lower_arr = np.extract(sel_nms < lower, sel_nms)
                tota = np.concatenate((upper_arr, lower_arr))
                st.dataframe(tota)
                upp_arr = np.where(st.session_state.my_dframe3[sel_mns] > upper)[0]
                low_arr = np.where(st.session_state.my_dframe3[sel_mns] < lower)[0]
                if st.button("remove outliers"):
                    ddm = st.session_state.my_dframe3.copy(deep=True)
                    ddm.drop(index=upp_arr, inplace=True)
                    ddm.drop(index=low_arr, inplace=True)
                    st.dataframe(ddm)
                    st.download_button(label='download CSV', data=ddm.to_csv(), mime='text/csv')

            if d_option == "IQR-score":
                st.subheader("By IQR score")
                all_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                sel_ns = st.multiselect("Select Columns", all_columns)
                sel_nos = df[sel_ns]
                q1 = np.percentile(sel_nos, 25)
                q3 = np.percentile(sel_nos, 75)
                iqr = abs(q3) - abs(q1)
                st.write(iqr)
                upper = abs(q3) + 1.5 * iqr
                lower = abs(q1) - 1.5 * iqr
                st.write("upper limit (or) 75 percentile  = ", upper)
                st.write("lower limit (or) 25 percentile  = ", lower)
                upper_arr = np.extract(sel_nos >= upper, sel_nos)
                lower_arr = np.extract(sel_nos <= lower, sel_nos)
                tot = np.concatenate((upper_arr, lower_arr))
                st.dataframe(tot)
                upp_arr = np.where(df[sel_ns] >= upper)[0]
                low_arr = np.where(df[sel_ns] <= lower)[0]
                if st.button("remove outliers"):
                    ddf = st.session_state.my_dframe3.copy(deep=True)
                    ddf.drop(index=upp_arr, inplace=True)
                    ddf.drop(index=low_arr, inplace=True)
                    st.dataframe(ddf)
                    st.download_button(label='download CSV', data=ddf.to_csv(), mime='text/csv')

        if options == "Normalisation":
            st.subheader("Normalisation")

            all_columns = st.session_state.my_dframe3.columns.to_list()
            sel_mns = st.multiselect("Select Columns", all_columns)
            sems = st.session_state.my_dframe3[sel_mns]
            result = sems.apply(lambda i:
                                ((i.max() - i) / (i.max() - i.min())).round(2))

            st.dataframe(result)
            st.download_button(label='download CSV', data=result.to_csv(), mime='text/csv')

        if options == "Standardisation":
            st.subheader("Standardisation")

            all_columns = st.session_state.my_dframe3.columns.to_list()
            sel_mns = st.multiselect("Select Columns", all_columns)
            semsi = df[sel_mns]
            resulti = semsi.apply(lambda d: ((d - np.mean(d)) / np.std(d)))
            st.dataframe(resulti)
            st.download_button(label='download CSV', data=resulti.to_csv(), mime='text/csv')
    elif choice == '3.plots':
        st.subheader("plots    :bar_chart:")
        p_tasks = ("area chart","bar chart","hist","lineplot","kde plot")
        p_options = st.sidebar.selectbox("Plots",p_tasks)
        if data is not None:
            data.seek(0)
            if p_options == "area chart":
                st.subheader("Area chart")

                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig = plt.figure()
                df[selected_cs].value_counts().plot(kind="area", color=user_colour)
                st.pyplot(fig)
            if p_options == "bar chart":
                st.subheader("Bar chart")

                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = st.session_state.my_dframe3.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig = plt.figure()
                df[selected_cs].value_counts().plot(kind="bar", color=user_colour)
                st.pyplot(fig)
            if p_options == "hist":
                st.subheader("Histogram")

                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig1 = plt.figure()
                df[selected_cs].value_counts().plot(kind="hist", color=user_colour)
                st.pyplot(fig1)
            if p_options == "lineplot":
                st.subheader("Lineplot")

                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig = plt.figure()
                df[selected_cs].value_counts().plot(kind="line", color=user_colour)
                st.pyplot(fig)
            if p_options == "kde plot":
                st.subheader("Kernel Density Estimation PLot")

                st.header('Streamlit Colour Picker for Charts')
                user_colour = st.color_picker(label='Choose a colour for your plot')
                all_cs = df.columns.to_list()
                selected_cs = st.multiselect("Select Columns", all_cs)
                fig = plt.figure()
                st.session_state.my_dframe3[selected_cs].value_counts().plot(kind="kde", color=user_colour)
                st.pyplot(fig)

    elif choice == "4.relationship analysis":
        st.subheader("analyzing the relationship between field")
        r_tasks = ("correlation analysis","relation plot")
        r_option = st.sidebar.selectbox("relationship analysis",r_tasks)
        if data is not None:
            data.seek(0)
        if r_option == 'correlation analysis':
            st.subheader("Correlation Analysis")

            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            nef = df.select_dtypes(include=numerics)
            fig, ax = plt.subplots()
            sns.heatmap(nef.corr(), ax=ax)
            st.write(fig)
        if r_option == 'relation plot':
            st.subheader("Relation Plot")

            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            nef = df.select_dtypes(include=numerics)
            fig = sns.pairplot(nef.corr())
            st.pyplot(fig)



if __name__ == '__main__':
    main()