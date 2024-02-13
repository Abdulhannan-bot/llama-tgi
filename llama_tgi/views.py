from django.shortcuts import render
from .llm_loader import summarize
# Create your views here.

def home(request):
  msg = ""
  if request.method == "POST":
    print(request.POST)
    msg = summarize(request.POST.get("prompt"))

  return render(request, "home.html", context={'msg':msg})