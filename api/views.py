from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from deep.u2net_test import main
from deep.models import OImage
from api.serializers import ImageSerializer
from rest_framework import generics



@api_view(['POST'])
def api_post(request):
    serializer = ImageSerializer(data=request.data)
    if serializer.is_valid(): 
        serializer.save()
        im = OImage.objects.last()
        main(im)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)