import streamlit as st
import pandas as pd
import pickle

# Load model
def load_model(model_name):
    if model_name == 'Logistic Regression':
        model = pickle.load(open('./no_resampling_logistic_regression_model.pkl', 'rb'))
        # model = pickle.load(open('models/no_resampling_logistic_regression_model.pkl', 'rb'))        
    elif model_name == 'SVM':
        model = pickle.load(open('models/no_resampling_svm_model.pkl', 'rb'))
    return model

# prediksi status
def predict_status(model, data):
    predictions = model.predict(data)
    return predictions

# highlight prediksi
def color_predictions(val):
    color = 'red' if val == 'Dropout' else 'green'
    return f'color: {color}'

def main():
    st.title('Prediksi Status Mahasiswa')

    st.sidebar.title("Petunjuk Penggunaan:")
    st.sidebar.write("1. Upload CSV file yang berisi data Mahasiswa")
    st.sidebar.markdown(">*Untuk format upload csv filenya bisa mencontoh student_test.csv yang terdapat pada proyek ini*")
    st.sidebar.write("2. Klik tombol 'Prediksi' untuk melihat hasil prediksi.")
    st.sidebar.write("3. Hasil prediksi bisa di download dengan klik tombol 'Download (.csv)'")                 

    model_name = "Logistic Regression"

    # Upload File
    uploaded_file = st.sidebar.file_uploader("Upload file CSV untuk prediksi", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("Pratinjau Data test:")
        preview_rows = st.slider("Geser slider untuk menampilkan jumlah baris yang diinginkan", 1, len(data), 10)
        st.write(data.head(preview_rows))

        # Load model yang dipilih
        model = load_model(model_name)

        # Button untuk trigger prediski
        if st.button('Prediksi'):
            # Melakukan prediksi
            predictions = predict_status(model, data)

            # Mengubah value agar mudah dipahami
            prediction_labels = ['Graduate' if pred == 1 else 'Dropout' for pred in predictions]

            # Menampilkan hasilnya
            result_df = pd.DataFrame({
                 'Status Prediction': prediction_labels
            })

            # Menampilkan hasil prediksi 
            st.write("Hasil Prediksi:")
            st.dataframe(result_df.style.applymap(color_predictions, subset=['Status Prediction']))

            # Download hasil prediksi
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download (.csv)",
                data=csv,
                file_name='hasil-prediksi-student.csv',
                mime='text/csv'
            )

if __name__ == '__main__':
    main()
