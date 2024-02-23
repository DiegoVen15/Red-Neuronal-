from django.db import models

class GestureModel(models.Model):
    
    value = models.FloatField()
    classification = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.timestamp}: Input value - {self.value}, Result - {self.classification}"