from django.db import models

# create model for save data json as a string 


class VideoData(models.Model):
    user_id = models.IntegerField(default=0, null=True)
    data = models.TextField(default="", null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)
