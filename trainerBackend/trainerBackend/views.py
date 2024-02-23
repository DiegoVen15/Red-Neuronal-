import time
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import json
import torch
import sys
from .models import GestureModel
from django.utils import timezone
from decimal import Decimal


class DeepNN(torch.nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, 4) 

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path):
    model = DeepNN()
    model.load_state_dict(torch.load(model_path))
    model.eval() 
    return model

def preprocess_input(value):
    return torch.tensor(value, dtype=torch.float32).view(-1, 1)

def classify_input(model, input_value):
    with torch.no_grad():
        model.eval()
        input_tensor = preprocess_input(input_value)
        output = model(input_tensor)
        predicted_class = torch.argmax(output).item()
    return predicted_class

def run_prediction(input_value):

    model_path = './best_model.pth'
    model = load_model(model_path)
    predicted_class = classify_input(model, input_value)

    print(f'The predicted class for input value {input_value} is: {predicted_class}')
    return predicted_class

@csrf_exempt
def GetPredictedGesture(request):
    if request.method == 'POST':

        data = json.loads(request.body)
        
        start_time = time.time()
        result = run_prediction(data['value'])
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(f"Elapsed time: {elapsed_time_ms:.2f} milliseconds")
        
        value = Decimal(data['value'])

        GestureModel.objects.create(value=value, classification=result, timestamp=timezone.now())

        post_data = {'result': result}

        print(post_data)

        response = requests.post('http://172.20.10.6:80/ReceiveResult/', json=post_data, headers={'Content-Type': 'application/json'}) #post data a  la segunda mano

        print('Second ESP Response')
        print(response.content)

        return JsonResponse({'status': 'success'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})