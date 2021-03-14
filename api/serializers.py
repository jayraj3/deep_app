from rest_framework import serializers
from deep.models import OImage

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = OImage
        fields = ['Image', 'action']