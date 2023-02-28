from rest_framework.views import APIView 
from rest_framework.response import Response
from rest_framework import status
from .models import VideoData
from rest_framework.permissions import AllowAny, IsAuthenticated
from .serializers import VideoDataSerializer
from .producer import publish


# Create api view for inserting data into video_data table
class VideoDataAPIView(APIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        serializer = VideoDataSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            publish(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        video_data = VideoData.objects.all()
        serializer = VideoDataSerializer(video_data, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
