from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('save_simulation/', views.save_simulation, name='save_simulation'),
    path('download_results/', views.download_results, name='download_results'),
    path('documentation/', views.documentation, name='documentation'),
    path('saved_simulations/', views.saved_simulations, name='saved_simulations'),
    path('delete_simulation/', views.delete_simulation, name='delete_simulation'),
    path('compare/<uuid:sim_id>/', views.compare_simulation, name='compare_simulation'),
]