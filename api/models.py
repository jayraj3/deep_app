from django.db import models
from string import Template
from django.utils.safestring import mark_safe
from django_resized import ResizedImageField


class OImage(models.Model): 
	Image =  ResizedImageField(size=[1024, 768], upload_to='images/', blank=True, null=True, quality=50)
	CHOICES = (
		('1', 'Remove'),
		('2', 'Blur'),
		('3', 'Black and White'),
	)
	action = models.CharField(max_length=100, default=3, choices=CHOICES)