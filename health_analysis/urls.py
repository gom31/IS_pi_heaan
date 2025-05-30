from django.urls import path
from . import views

app_name = 'health_analysis'

urlpatterns = [
    # main page
    path('', views.HomeView.as_view(), name='home'),
    
    # API end point
    path('api/analyze/', views.HealthAnalysisView.as_view(), name='analyze'),
    path('api/form/', views.HealthFormView.as_view(), name='form'),
    path('api/health-check/', views.HealthAnalysisView.as_view(), name='health_check'),
    path('api/test/', views.HealthTestView.as_view(), name='test'),
    
    # info page
    path('privacy/', views.PrivacyPolicyView.as_view(), name='privacy'),
    path('about/', views.AboutView.as_view(), name='about'),
]
