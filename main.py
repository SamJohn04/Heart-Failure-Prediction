import pickle
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from tkinter import *

root = None

def is_numeric(value: str):
    try:
        float(value)
        return True
    except:
        return False

# def get_input(prompt: str):
#     result = []
#     input_frame = Frame(root, padx=50, pady=50)
#     input_label = Label(input_frame, text=prompt)
#     input_label.pack()
#     input_entry = Entry(input_frame)
#     input_entry.pack()
#     input_frame.pack()

#     def submit():
#         result.append(input_entry.get())
#         input_frame.destroy()

#     submit_button = Button(input_frame, text='Submit', command=submit, state=['disabled' if is_numeric(input_entry) else 'normal'])
#     submit_button.pack()
#     print(result[0])
#     return result[0]

def load_data():
    data = pd.read_csv('data.csv')
    return data.drop('DEATH_EVENT', axis=1)

def get_input_data(data: DataFrame):
    root = Tk()
    results = []
    entries = []
    input_frame = Frame(root, padx=50, pady=50)
    input_frame.pack()
    for column in data.columns:
        input_label = Label(input_frame, text=f'{column}: ')
        input_label.pack()
        input_entry = Entry(input_frame)
        input_entry.pack()
        entries.append(input_entry)

    def submit():
        results.append([float(entry.get()) for entry in entries])
        root.destroy()
    
    submit_button = Button(input_frame, text='Submit', command=submit)
    submit_button.pack()
    root.mainloop()
    return DataFrame(results, columns=data.columns)

def scale_data(data: DataFrame, input_data: DataFrame):

    scaler = StandardScaler()

    scaler.fit(data)
    scaled_input_data = scaler.transform(input_data)
    scaled_input_data = DataFrame(scaled_input_data)
    scaled_input_data.columns = input_data.columns

    return scaled_input_data

def main():
    model = pickle.load(open('final_model.sav', 'rb'))
    data = load_data()
    input_data = get_input_data(data)
    scaled_input_data = scale_data(data, input_data)
    prediction = model.predict(scaled_input_data)
    return prediction

if __name__ == '__main__':
    prediction = main()
    root = Tk(padx=50, pady=50)
    result_label = Label(root, text=f'Prediction: {"High probability of death" if prediction else "Low probability of death"}')
    result_label.pack()
    root.mainloop()