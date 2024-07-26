import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

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

def on_predict():
    try:
        # Obter valores dos campos de entrada
        age = int(entry_age.get())
        gender = gender_combobox.current()
        ethnicity = ethnicity_combobox.current()
        parental_education = parental_education_combobox.current()
        study_time_weekly = float(entry_study_time_weekly.get())
        absences = int(entry_absences.get())
        tutoring = tutoring_combobox.current()
        parental_support = parental_support_combobox.current()
        extracurricular = extracurricular_combobox.current()
        sports = sports_combobox.current()
        music = music_combobox.current()
        volunteering = volunteering_combobox.current()
        
        # Criar DataFrame com os dados do estudante
        new_student_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Ethnicity': [ethnicity],
            'ParentalEducation': [parental_education],
            'StudyTimeWeekly': [study_time_weekly],
            'Absences': [absences],
            'Tutoring': [tutoring],
            'ParentalSupport': [parental_support],
            'Extracurricular': [extracurricular],
            'Sports': [sports],
            'Music': [music],
            'Volunteering': [volunteering]
        })
        
        # Fazer a previsão
        grade_class_prediction = predict_grade_class(model, new_student_data)
        print(grade_class_prediction)
        grade_class_mapping = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
        result = f'Predição da classe de nota desse aluno: {grade_class_mapping[grade_class_prediction[0]]}'
        
        # Mostrar resultado na interface gráfica
        messagebox.showinfo("Resultado da Predição", result)
    
    except ValueError:
        messagebox.showerror("Erro de Entrada", "Por favor, insira valores válidos.")

# Carregar modelo
model = joblib.load("model/students_model_random_forest.pkl")

# Configurar interface gráfica
root = tk.Tk()
root.title("Previsão de Classe de Notas de Estudantes")

# Configurar o grid para centralizar o conteúdo
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Adicionar um título principal
title = ttk.Label(root, text="Previsão de Classe de Notas de Estudantes", font=("Helvetica", 16, "bold"))
title.grid(row=0, column=0, columnspan=2, pady=10, padx=20, sticky='n')

# Criar um frame para organizar os widgets
frame = ttk.Frame(root, padding="10")
frame.grid(row=1, column=0, columnspan=2, sticky='nsew')

labels = [
    "Idade",
    "Gênero",
    "Etnia",
    "Educação dos Pais",
    "Tempo de Estudo Semanal (horas)",
    "Faltas",
    "Tutoria",
    "Apoio Parental",
    "Atividades Extracurriculares",
    "Esportes",
    "Música",
    "Voluntariado",
]

options = {
    "Gênero": ["Masculino", "Feminino"],
    "Etnia": ["Caucasiano", "Afro-americano", "Asiático", "Outro"],
    "Educação dos Pais": ["Nenhum", "Ensino Médio", "Alguma Faculdade", "Bacharelado", "Superior"],
    "Tutoria": ["Não", "Sim"],
    "Apoio Parental": ["Nenhum", "Baixo", "Moderado", "Alto", "Muito Alto"],
    "Atividades Extracurriculares": ["Não", "Sim"],
    "Esportes": ["Não", "Sim"],
    "Música": ["Não", "Sim"],
    "Voluntariado": ["Não", "Sim"]
}

entries = []
combobox_vars = {}

for i, label in enumerate(labels):
    ttk.Label(frame, text=f"{label}:").grid(row=i, column=0, padx=(0, 10), pady=5, sticky='E')
    if label in options: # Se for do MENU SELECIONÁVEL
        combobox = ttk.Combobox(frame, values=options[label], state="readonly", width=30)
        combobox.grid(row=i, column=1, pady=5, sticky='ew')
        entries.append(combobox)
        combobox_vars[label] = combobox
    else:
        entry = ttk.Entry(frame, width=30)
        entry.grid(row=i, column=1, pady=5, sticky='ew')
        entries.append(entry)

# Associar entradas às variáveis
entry_age, gender_combobox, ethnicity_combobox, parental_education_combobox, entry_study_time_weekly, \
entry_absences, tutoring_combobox, parental_support_combobox, extracurricular_combobox, sports_combobox, \
music_combobox, volunteering_combobox = entries

# Botão para prever
predict_button = ttk.Button(root, text='Prever', command=on_predict)
predict_button.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()
