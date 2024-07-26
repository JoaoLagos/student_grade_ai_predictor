import joblib
import pandas as pd

def predict_grade_class(model, new_data):
    """
    Função para prever a classe de nota de um estudante.
    
    Parâmetros:
    - model: Modelo treinado.
    - new_data: DataFrame contendo as novas entradas para previsão.
    
    Retorna:
    - Predições de classe de nota.
    """
    prediction = model.predict(new_data)
    return prediction


# Exemplo de uso:
# Carregar modelo
model = joblib.load("model/students_model_random_forest.pkl")
# Suponha que new_student_data seja um DataFrame com as mesmas colunas que X (sem 'GradeClass')
new_student_data = pd.DataFrame({
    'Age': [15],
    'Gender': [0],
    'Ethnicity': [2],
    'ParentalEducation': [3],
    'StudyTimeWeekly': [4.2],
    'Absences': [26],
    'Tutoring': [0],
    'ParentalSupport': [2],
    'Extracurricular': [0],
    'Sports': [0],
    'Music': [0],
    'Volunteering': [0],
    'GPA': [0.11]
})

# Previsão
grade_class_prediction = predict_grade_class(model, new_student_data)
print(f'Predição da classe de nota: {grade_class_prediction[0]}')