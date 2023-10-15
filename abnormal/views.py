from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse,HttpResponse
from .model_utils import model, tokenizer
import torch
from .SetModel import Model
from .VniAcronym import Acronym
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
from .Pho_Chat import Chat

def analyze_sentiment(request):
    if request.method == 'POST':
        text = request.POST['text']
        tokenized_text = tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor(tokenized_text).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_ids)
            predictions = torch.argmax(outputs.logits, dim=-1)
            sentiment = "Positive" if predictions.item() == 1 else "Negative"

        return render(request, 'sentiment_analysis/result.html', {'text': text, 'sentiment': sentiment})

    return render(request, 'sentiment_analysis/analyze.html')


def analyze_sentiment1(request):
    M = Model('fine_tuned_model_best')
    A = Acronym()
    if request.method == 'POST':
        input_text = request.POST['text']
        tokenized_text = A.Solve_Acr(input_text)

        label, cof = M.Predict(tokenized_text)
        with open("HisFeedBack/"+label+'.txt','a',encoding = 'utf8') as f:
            f.write(tokenized_text+'\n')
        # Update dictionary with input_text as key and label as value
        result_dict = {input_text: label}

        # Write dictionary to JSON file
        with open("HisFeedBack/hisfeedback.json", 'a', encoding='utf8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)

        with open("HisFeedBack/"+label+'.txt', 'a', encoding='utf8') as f:
            f.write(input_text+'\n')
            
        return render(request, 'sentiment_analysis/result.html', {'text': input_text, 'sentiment': label})

    return render(request, 'sentiment_analysis/analyze.html')


def home_page(request):
    return render(request, 'sentiment_analysis/index.html')

def members(request):
    return render(request, 'sentiment_analysis/members.html')

def feeback(request):
    return render(request, 'sentiment_analysis/feedback.html')



def feedback_analysis(request):
    with open("HisFeedBack/Positive.txt", 'r', encoding='utf8') as f:
        num_pos = len(f.readlines())
    with open("HisFeedBack/Negative.txt", 'r', encoding='utf8') as f:
        num_neg = len(f.readlines())

    data = {
        'Category': ['Positive', 'Negative'],
        'Values': [num_pos, num_neg]
    }

    df = pd.DataFrame(data)

    categories = data['Category']
    values = data['Values']

    plt.bar(categories, values, color=['green', 'red'])
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Positive vs Negative Values')

    # Create a temporary buffer to store the plot as an image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    chart_image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return render(request, 'sentiment_analysis/feedback_analysis.html', {'chart_image': chart_image_base64})


def chatbot_view(request):
    chatbot_response = ""

    if request.method == 'POST':
        user_input = request.POST.get('input_text', '')
        chatbot_response = Chat(user_input)  # Use your chatbot logic here
        result_dict = {user_input: chatbot_response}
        # Write dictionary to JSON file
        with open("HisFeedBack/chatbot.json", 'a', encoding='utf8') as json_file:
            json.dump(result_dict, json_file, ensure_ascii=False, indent=4)
        

    return render(request, 'sentiment_analysis/chatbot.html', {'chatbot_response': chatbot_response})


def hisfeedback(request):
    if request.method == 'GET':
        # Read the data from the hisfeedback.json file into a variable
        with open("HisFeedBack/hisfeedback.json", 'r', encoding='utf8') as json_file:
            json_data = json_file.read()

        # Clear the hisfeedback.json file by opening it in write mode
        with open("HisFeedBack/hisfeedback.json", 'w') as json_file:
            json_file.write("")

        # Create a JsonResponse with the variable's data
        response_data = {'data': json_data}  # You can add more keys to the dictionary if needed
        return JsonResponse(response_data)
    
def hischatbot(request):
    if request.method == 'GET':
        # Read the data from the chatbot.json file into a variable
        with open("HisFeedBack/chatbot.json", 'r', encoding='utf8') as json_file:
            json_data = json_file.read()

        # Clear the hisfeedback.json file by opening it in write mode
        with open("HisFeedBack/chatbot.json", 'w') as json_file:
            json_file.write("")

        # Create a JsonResponse with the variable's data
        response_data = {'data': json_data}  # You can add more keys to the dictionary if needed
        return JsonResponse(response_data)