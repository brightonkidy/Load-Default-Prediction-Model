from django.urls import path
from .views import predict_view, docs_view


urlpatterns = [
    path('', predict_view, name='predict'),
    path('docs/', docs_view, name='docs'),
]


